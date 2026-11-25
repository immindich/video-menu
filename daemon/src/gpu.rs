//! GPU (wgpu/Vulkan) implementation of the frame pipeline in
//! [`crate::pipeline`]: the camera frame is uploaded, unpacked to planar RGB,
//! run through effects and packed as YUY2 by compute shaders
//! (`pipeline.wgsl`), and the result is read back.
//!
//! Effects that use the matting model split the frame's GPU work into
//! stages around the model (NPU): [`ingest`](GpuProcessor::ingest) makes
//! the model's small input straight from the camera's bytes and reads it
//! back, while the full-resolution unpack follows on the GPU;
//! [`blur`](GpuProcessor::blur) runs while the model works; and
//! [`finish`](GpuProcessor::finish) uploads the model's output, composites
//! and packs. The full-resolution frame stays on the GPU throughout, in one
//! of a few slots, so stages of different frames can overlap.

use std::{error::Error, sync::mpsc};

use bytemuck::{Pod, Zeroable};
use half::f16;

use crate::{
    profile::GpuProfiler,
    pipeline::{self, Background, BlurGeometry, InputFormat, MAX_BLUR_TAPS},
};

const WORKGROUP: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    width: u32,
    height: u32,
    src_words: u32,
    dst_words: u32,
    r_shift: u32,
    g_shift: u32,
    b_shift: u32,
    _pad0: u32,
    small_w: u32,
    small_h: u32,
    _pad1: [u32; 2],
    bg: [f32; 4],
    blur_w: u32,
    blur_h: u32,
    blur_factor: u32,
    blur_taps: u32,
    /// The blur kernel's weights for offsets 0..=blur_taps.
    blur_weights: [f32; MAX_BLUR_TAPS + 1],
}

/// Buffers sized for one frame size and shared by the slots: the camera
/// frame as uploaded, and the YUY2 result with its CPU readback copy.
/// Frames use them one after another, in queue order.
struct Shared {
    width: usize,
    height: usize,
    input: wgpu::Buffer,
    output: wgpu::Buffer,
    readback: wgpu::Buffer,
}

/// A frame's own place on the GPU, so several frames can be in flight: its
/// full-resolution planar copy, its blurred background, the parameters it
/// was ingested with, and a bind group tying them to the shared buffers
/// (which keeps the slot's own buffers alive).
struct Slot {
    /// Samples the blur buffers hold; they grow (while the slot is free)
    /// when a smaller blur needs a bigger copy of the frame.
    blur_len: usize,
    params: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// The frame in the slot, from [`ingest`](GpuProcessor::ingest) until
    /// [`finish`](GpuProcessor::finish).
    job: Option<Job>,
}

/// What a frame in a slot was ingested with. Settings changed afterwards
/// apply to later frames.
#[derive(Clone, Copy, Debug)]
struct Job {
    width: usize,
    height: usize,
    background: Option<Background>,
    blur: Option<BlurGeometry>,
    /// Its blur has been submitted.
    blurred: bool,
}

/// How many frames can be on the GPU at once.
pub const SLOTS: usize = 3;

pub struct GpuProcessor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    unpack_packed: wgpu::ComputePipeline,
    unpack_yuy2: wgpu::ComputePipeline,
    downsample_packed: wgpu::ComputePipeline,
    downsample_yuy2: wgpu::ComputePipeline,
    unpack_coef: wgpu::ComputePipeline,
    composite_color: wgpu::ComputePipeline,
    blur_down: wgpu::ComputePipeline,
    blur_h: wgpu::ComputePipeline,
    blur_v: wgpu::ComputePipeline,
    composite_blur: wgpu::ComputePipeline,
    pack_yuy2: wgpu::ComputePipeline,
    /// The frame at model resolution (f16): the model's input, made from the
    /// camera's bytes by `downsample_packed` / `downsample_yuy2`.
    small: wgpu::Buffer,
    small_readback: wgpu::Buffer,
    /// The model's output (f16 A then b), read by the composite passes.
    coef: wgpu::Buffer,
    /// The coefficients unpacked to f32 and interleaved, A then b for each
    /// model pixel, by `unpack_coef`: the composite passes sample them at
    /// every frame pixel, and one load of a vec4 beats four of packed f16.
    coef4: wgpu::Buffer,
    /// Just the alpha coefficients (A and b), which `blur_down` needs: those
    /// of the last finished frame, as a frame's blur runs before its own
    /// result is in.
    coef_alpha: wgpu::Buffer,
    /// Zero `coef_alpha` before the next blur: there is no finished frame
    /// since the last reset.
    clear_alpha: bool,
    shared: Option<Shared>,
    slots: Vec<Slot>,
    /// Tightly packed copy of the input, used when its stride has padding.
    staging: Vec<u8>,
    /// The model input, copied out of `small_readback`.
    small_host: Vec<f16>,
    /// Resolution of the matting model, which sizes `small` and `coef`.
    model_size: (usize, usize),
    adapter_name: String,
    /// Records a timeline of each frame, if `VIDEO_MENU_GPU_PROFILE` is set.
    profiler: Option<GpuProfiler>,
}

impl GpuProcessor {
    /// Set up a hardware Vulkan device. Fails if there is
    /// none (a software rasterizer such as llvmpipe doesn't count).
    pub fn new() -> Result<Self, Box<dyn Error>> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(format!("only a software GPU is available ({})", info.name).into());
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("video-menu"),
            // Planar f32 frames are large (24 MB at 1080p); allow what the
            // hardware allows rather than wgpu's conservative defaults.
            required_limits: adapter.limits(),
            // Only for profiling, which needs timestamps.
            required_features: if std::env::var_os("VIDEO_MENU_GPU_PROFILE").is_some() {
                adapter.features() & wgpu::Features::TIMESTAMP_QUERY
            } else {
                wgpu::Features::empty()
            },
            ..Default::default()
        }))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pipeline.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("pipeline.wgsl").into()),
        });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pipeline"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, false),
                storage(3, false),
                storage(4, false),
                storage(5, true),
                storage(6, false),
                storage(7, false),
                storage(8, false),
                storage(9, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        // Placeholders until set_model_size gives the model's resolution.
        let model_size = (2, 1);
        let [small, small_readback, coef, coef4, coef_alpha] = model_buffers(&device, model_size);

        let profiler = GpuProfiler::from_env(&device, &queue);
        Ok(Self {
            profiler,
            unpack_packed: pipeline("unpack_packed"),
            unpack_yuy2: pipeline("unpack_yuy2"),
            downsample_packed: pipeline("downsample_packed"),
            downsample_yuy2: pipeline("downsample_yuy2"),
            unpack_coef: pipeline("unpack_coef"),
            composite_color: pipeline("composite_color"),
            blur_down: pipeline("blur_down"),
            blur_h: pipeline("blur_h"),
            blur_v: pipeline("blur_v"),
            composite_blur: pipeline("composite_blur"),
            pack_yuy2: pipeline("pack_yuy2"),
            small,
            small_readback,
            coef,
            coef4,
            coef_alpha,
            clear_alpha: true,
            device,
            queue,
            layout,
            shared: None,
            slots: Vec::new(),
            staging: Vec::new(),
            small_host: Vec::new(),
            model_size,
            adapter_name: info.name,
        })
    }

    /// Size the model's buffers for a model working at `size`, and forget
    /// the last frame's alpha. Only call with no frames in flight.
    pub fn set_model_size(&mut self, size: (usize, usize)) {
        if size != self.model_size {
            [self.small, self.small_readback, self.coef, self.coef4, self.coef_alpha] =
                model_buffers(&self.device, size);
            self.small_host = vec![f16::ZERO; 3 * size.0 * size.1];
            self.model_size = size;
            // The slots' bind groups refer to the old buffers.
            self.slots.clear();
        }
        self.clear_alpha = true;
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Forget the last frame's alpha, e.g. after switching cameras.
    pub fn reset_alpha(&mut self) {
        self.clear_alpha = true;
    }

    /// Upload a camera frame into `slot` and convert it to planar RGB there,
    /// ready for [`blur`](Self::blur) and [`finish`](Self::finish). With a
    /// `background` (an effect that runs the model), also make the model's
    /// input (unless `model_input` gives it already, e.g. from a scaled JPEG
    /// decode), leave it in [`model_input`](Self::model_input) and return
    /// true. Waits only for the model input.
    #[allow(clippy::too_many_arguments)]
    pub fn ingest(
        &mut self,
        slot: usize,
        input: InputFormat,
        src: &[u8],
        src_stride: usize,
        width: usize,
        height: usize,
        background: Option<Background>,
        model_input: Option<&[f16]>,
    ) -> Result<bool, String> {
        if !width.is_multiple_of(2) || !pipeline::frame_fits(input, src.len(), src_stride, width, height) {
            return Err(format!("short or odd-sized frame ({} bytes)", src.len()));
        }
        let blur = match background {
            Some(Background::Blur { radius }) => Some(BlurGeometry::new(width, height, radius)),
            _ => None,
        };
        self.ensure_shared(width, height);
        self.ensure_slot(slot, blur.map_or(0, |g| g.len()));
        if let Some(p) = &self.profiler {
            p.begin(slot, format!("{width}x{height} {background:?}"));
        }

        // Upload, dropping any row padding so the shader can index rows tightly.
        let row_bytes = input.row_bytes(width);
        let tight: &[u8] = if src_stride == row_bytes {
            &src[..row_bytes * height]
        } else {
            self.staging.clear();
            for row in src.chunks(src_stride).take(height) {
                self.staging.extend_from_slice(&row[..row_bytes]);
            }
            &self.staging
        };
        let shared = self.shared.as_ref().unwrap();
        self.queue.write_buffer(&shared.input, 0, tight);
        let params = self.params(input, width, height, background, blur.as_ref());
        self.queue.write_buffer(&self.slots[slot].params, 0, bytemuck::bytes_of(&params));
        self.slots[slot].job = Some(Job { width, height, background, blur, blurred: false });
        self.mark(slot, "uploaded");

        // The model's input first, straight from the camera's bytes, and its
        // readback; then the full-resolution unpack, which runs while the
        // model input is read back and the model works.
        if let (Some(_), Some(given)) = (background, model_input) {
            if given.len() != self.small_host.len() {
                return Err(format!("model input has {} values, expected {}", given.len(), self.small_host.len()));
            }
            self.small_host.copy_from_slice(given);
        }
        let model_input = background.filter(|_| model_input.is_none()).map(|_| {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            let (mw, mh) = self.model_size;
            let (downsample, name) = match input {
                InputFormat::Packed { .. } => (&self.downsample_packed, "downsample_packed"),
                InputFormat::Yuy2 => (&self.downsample_yuy2, "downsample_yuy2"),
            };
            self.dispatch(&mut encoder, slot, downsample, name, (mw / 2) as u32, mh as u32);
            let bytes = self.small.size();
            encoder.copy_buffer_to_buffer(&self.small, 0, &self.small_readback, 0, bytes);
            let commands = encoder.finish();
            self.submitted(slot, "model input");
            let index = self.queue.submit([commands]);
            self.mark(slot, "submit returned");
            (index, bytes)
        });
        let (w, h) = (width as u32, height as u32);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        match input {
            InputFormat::Packed { .. } => self.dispatch(&mut encoder, slot, &self.unpack_packed, "unpack_packed", w, h),
            InputFormat::Yuy2 => self.dispatch(&mut encoder, slot, &self.unpack_yuy2, "unpack_yuy2", w / 2, h),
        }
        let commands = encoder.finish();
        self.submitted(slot, "unpack");
        self.queue.submit([commands]);
        self.mark(slot, "submit returned");
        let Some((index, bytes)) = model_input else {
            return Ok(background.is_some());
        };
        self.wait_mapped(&self.small_readback, bytes, index)?;
        self.mark(slot, "model input waited");
        {
            let mapped = self
                .small_readback
                .slice(..)
                .get_mapped_range()
                .map_err(|e| format!("GPU readback failed: {e}"))?;
            bytemuck::cast_slice_mut(&mut self.small_host).copy_from_slice(&mapped);
        }
        self.small_readback.unmap();
        Ok(true)
    }

    /// The model input the last [`ingest`](Self::ingest) made.
    pub fn model_input(&self) -> &[f16] {
        &self.small_host
    }

    /// Start blurring the background of the frame in `slot`, if its effect
    /// is a blur, weighing pixels by the alpha of the last finished frame.
    /// Doesn't wait; [`finish`](Self::finish) blurs if this wasn't called.
    /// Returns whether there was anything to blur.
    pub fn blur(&mut self, slot: usize) -> bool {
        let Some(job) = self.slots.get_mut(slot).and_then(|s| s.job.as_mut()) else { return false };
        let Some(g) = job.blur.filter(|_| !job.blurred) else { return false };
        job.blurred = true;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.record_blur(&mut encoder, slot, &g);
        let commands = encoder.finish();
        self.submitted(slot, "blur");
        self.queue.submit([commands]);
        self.mark(slot, "submit returned");
        true
    }

    fn record_blur(&mut self, encoder: &mut wgpu::CommandEncoder, slot: usize, g: &BlurGeometry) {
        if std::mem::take(&mut self.clear_alpha) {
            encoder.clear_buffer(&self.coef_alpha, 0, None);
        }
        let (bw, bh) = (g.width as u32, g.height as u32);
        self.dispatch(encoder, slot, &self.blur_down, "blur_down", bw, bh);
        self.dispatch(encoder, slot, &self.blur_h, "blur_h", bw, bh);
        self.dispatch(encoder, slot, &self.blur_v, "blur_v", bw, bh);
    }

    /// Composite the frame in `slot` with the model's coefficients (for an
    /// effect that runs the model), pack it as YUY2 into `dst` (rows
    /// `dst_stride` apart), and free the slot.
    pub fn finish(
        &mut self,
        slot: usize,
        coefficients: Option<(&[f16], &[f16])>,
        dst: &mut [u8],
        dst_stride: usize,
    ) -> Result<(), String> {
        let job = self
            .slots
            .get_mut(slot)
            .and_then(|s| s.job.take())
            .ok_or_else(|| format!("no frame in slot {slot}"))?;
        let (w, h) = (job.width as u32, job.height as u32);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        if let Some(background) = job.background {
            let (a, b) = coefficients.ok_or("the effect needs the model's coefficients")?;
            if let Some(g) = job.blur.filter(|_| !job.blurred) {
                self.record_blur(&mut encoder, slot, &g);
            }
            self.queue.write_buffer(&self.coef, 0, bytemuck::cast_slice(a));
            self.queue.write_buffer(&self.coef, (a.len() * 2) as u64, bytemuck::cast_slice(b));
            // unpack_coef replaces the alpha coefficients with this frame's.
            self.clear_alpha = false;
            let (mw, mh) = self.model_size;
            self.dispatch(&mut encoder, slot, &self.unpack_coef, "unpack_coef", mw as u32, mh as u32);
            let (composite, name) = match background {
                Background::Color(_) => (&self.composite_color, "composite_color"),
                Background::Blur { .. } => (&self.composite_blur, "composite_blur"),
            };
            self.dispatch(&mut encoder, slot, composite, name, w, h);
        }
        self.dispatch(&mut encoder, slot, &self.pack_yuy2, "pack_yuy2", w / 2, h);

        let shared = self.shared.as_ref().unwrap();
        let out_bytes = (job.width * 2 * job.height) as u64;
        encoder.copy_buffer_to_buffer(&shared.output, 0, &shared.readback, 0, out_bytes);
        if let Some(p) = &self.profiler {
            p.resolve(slot, &mut encoder);
        }
        let commands = encoder.finish();
        self.submitted(slot, "output");
        let index = self.queue.submit([commands]);
        self.mark(slot, "submit returned");
        self.wait_mapped(&shared.readback, out_bytes, index)?;
        self.mark(slot, "output waited");

        {
            let mapped = shared
                .readback
                .slice(..out_bytes)
                .get_mapped_range()
                .map_err(|e| format!("GPU readback failed: {e}"))?;
            let out_row = job.width * 2;
            for (dst_row, src_row) in dst.chunks_mut(dst_stride).zip(mapped.chunks(out_row)) {
                dst_row[..out_row].copy_from_slice(src_row);
            }
        }
        shared.readback.unmap();
        if let Some(p) = &self.profiler {
            p.mark(slot, "output copied");
            p.finish(slot, &self.device);
        }
        Ok(())
    }

    /// The shader parameters for a frame.
    fn params(
        &self,
        input: InputFormat,
        width: usize,
        height: usize,
        background: Option<Background>,
        blur: Option<&BlurGeometry>,
    ) -> Params {
        let (r, g, b) = match input {
            InputFormat::Packed { r, g, b, .. } => (r, g, b),
            InputFormat::Yuy2 => (0, 0, 0),
        };
        let mut params = Params {
            width: width as u32,
            height: height as u32,
            src_words: (input.row_bytes(width) / 4) as u32,
            dst_words: (width / 2) as u32,
            r_shift: 8 * r as u32,
            g_shift: 8 * g as u32,
            b_shift: 8 * b as u32,
            _pad0: 0,
            small_w: self.model_size.0 as u32,
            small_h: self.model_size.1 as u32,
            _pad1: [0; 2],
            bg: match background {
                Some(Background::Color(c)) => [c[0], c[1], c[2], 1.0],
                _ => [0.0; 4],
            },
            blur_w: 0,
            blur_h: 0,
            blur_factor: 0,
            blur_taps: 0,
            blur_weights: [0.0; MAX_BLUR_TAPS + 1],
        };
        if let Some(g) = blur {
            (params.blur_w, params.blur_h) = (g.width as u32, g.height as u32);
            (params.blur_factor, params.blur_taps) = (g.factor as u32, g.taps as u32);
            let weights = g.weights();
            params.blur_weights[..weights.len()].copy_from_slice(&weights);
        }
        params
    }

    /// Note an event for the frame in `slot` in the profile, if profiling.
    pub fn mark(&self, slot: usize, name: &'static str) {
        if let Some(p) = &self.profiler {
            p.mark(slot, name);
        }
    }

    /// Note a number about the frame in `slot` in the profile, if profiling.
    pub fn note(&self, slot: usize, name: &'static str, value: u64) {
        if let Some(p) = &self.profiler {
            p.note(slot, name, value);
        }
    }

    /// Note a submission for the frame in `slot`. Called just before
    /// `submit()`, as the GPU may start on it before the call returns;
    /// "submit returned" is marked after, showing what the call costs.
    fn submitted(&self, slot: usize, name: &'static str) {
        if let Some(p) = &self.profiler {
            p.submitted(slot, name);
        }
    }

    /// Record `pipeline` (called `name`, for profiling) over a `x`x`y` grid
    /// of invocations, on the frame in `slot`.
    fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        slot: usize,
        pipeline: &wgpu::ComputePipeline,
        name: &'static str,
        x: u32,
        y: u32,
    ) {
        let timestamp_writes = self.profiler.as_ref().and_then(|p| p.pass(slot, name));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(name), timestamp_writes });
        pass.set_bind_group(0, &self.slots[slot].bind_group, &[]);
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(x.div_ceil(WORKGROUP), y.div_ceil(WORKGROUP), 1);
    }

    /// Wait for the submission `index` (and those before it, as the queue
    /// runs in order), then map the first `len` bytes of `buffer`.
    fn wait_mapped(&self, buffer: &wgpu::Buffer, len: u64, index: wgpu::SubmissionIndex) -> Result<(), String> {
        let (tx, rx) = mpsc::channel();
        buffer.slice(..len).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait { submission_index: Some(index), timeout: None })
            .map_err(|e| format!("GPU poll failed: {e}"))?;
        rx.recv()
            .map_err(|_| "GPU readback was dropped".to_string())?
            .map_err(|e| format!("GPU readback failed: {e}"))
    }

    /// Make sure the shared buffers fit a `width`x`height` frame. A new size
    /// means new buffers, so all slots must be free.
    fn ensure_shared(&mut self, width: usize, height: usize) {
        if self.shared.as_ref().is_some_and(|s| (s.width, s.height) == (width, height)) {
            return;
        }
        debug_assert!(self.slots.iter().all(|s| s.job.is_none()), "resizing with frames in flight");
        let pixels = (width * height) as u64;
        use wgpu::BufferUsages as U;
        self.shared = Some(Shared {
            width,
            height,
            // Packed RGB is the largest input: 4 bytes per pixel.
            input: self.buffer("input", pixels * 4, U::STORAGE | U::COPY_DST),
            output: self.buffer("output", pixels * 2, U::STORAGE | U::COPY_SRC),
            readback: self.buffer("readback", pixels * 2, U::MAP_READ | U::COPY_DST),
        });
        // Their bind groups refer to the old buffers.
        self.slots.clear();
    }

    /// Make sure `slot` exists, fits the current frame size, and its blur
    /// buffers hold `blur_len` samples. Only call while the slot is free.
    fn ensure_slot(&mut self, slot: usize, blur_len: usize) {
        assert!(slot < SLOTS, "there are {SLOTS} slots");
        while self.slots.len() <= slot {
            let slot = self.new_slot(1);
            self.slots.push(slot);
        }
        if self.slots[slot].blur_len < blur_len {
            self.slots[slot] = self.new_slot(blur_len);
        }
    }

    fn new_slot(&self, blur_len: usize) -> Slot {
        let shared = self.shared.as_ref().expect("shared buffers before slots");
        let pixels = (shared.width * shared.height) as u64;
        use wgpu::BufferUsages as U;
        let planar = self.buffer("planar", pixels * 3 * 4, U::STORAGE);
        let blur_a = self.buffer("blur a", (blur_len * 4) as u64, U::STORAGE);
        let blur_b = self.buffer("blur b", (blur_len * 4) as u64, U::STORAGE);
        let params = self.buffer("params", size_of::<Params>() as u64, U::UNIFORM | U::COPY_DST);
        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pipeline"),
            layout: &self.layout,
            entries: &[
                entry(0, &params),
                entry(1, &shared.input),
                entry(2, &planar),
                entry(3, &shared.output),
                entry(4, &self.small),
                entry(5, &self.coef),
                entry(6, &blur_a),
                entry(7, &blur_b),
                entry(8, &self.coef4),
                entry(9, &self.coef_alpha),
            ],
        });
        Slot { blur_len, params, bind_group, job: None }
    }

    fn buffer(&self, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
    }
}

/// Buffers for a matting model working at `(w, h)`: the frame at that
/// resolution (f16, written by `downsample`), its CPU readback copy, and the
/// model's coefficients (f16 A then b, as the model returns them), the same
/// unpacked for the composite passes, and just alpha's for `blur_down`. The
/// shaders pack two f16 per word, so `w` must be even.
fn model_buffers(device: &wgpu::Device, (w, h): (usize, usize)) -> [wgpu::Buffer; 5] {
    assert!(w.is_multiple_of(2), "model width must be even");
    let buffer = |label, size, usage| {
        device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false })
    };
    use wgpu::BufferUsages as U;
    let small_bytes = (3 * w * h * 2) as u64;
    [
        buffer("small", small_bytes, U::STORAGE | U::COPY_SRC),
        buffer("small readback", small_bytes, U::MAP_READ | U::COPY_DST),
        buffer("coef", (8 * w * h * 2) as u64, U::STORAGE | U::COPY_DST),
        buffer("coef4", (8 * w * h * 4) as u64, U::STORAGE),
        buffer("coef alpha", (2 * w * h * 4) as u64, U::STORAGE | U::COPY_DST),
    ]
}

#[cfg(test)]
mod tests {
    use half::slice::HalfFloatSliceExt;
    use pipewire::spa::param::video::VideoFormat;

    use super::*;
    use crate::pipeline::{PlanarFrame, resample, unpack};

    /// The model input the GPU makes from the camera's bytes matches the
    /// CPU's (unpack, then resample) to within one f16 step.
    #[test]
    fn model_input_from_camera_bytes_matches_cpu() {
        let mut gpu = match GpuProcessor::new() {
            Ok(gpu) => gpu,
            Err(e) => return eprintln!("skipping: no GPU ({e})"),
        };
        let formats = [
            ("BGRx", InputFormat::from_video_format(VideoFormat::BGRx).unwrap()),
            ("RGBx", InputFormat::from_video_format(VideoFormat::RGBx).unwrap()),
            ("YUY2", InputFormat::Yuy2),
        ];
        for (mw, mh) in [(480, 270), (256, 144)] {
            gpu.set_model_size((mw, mh));
            for (w, h) in [(200, 120), (70, 37), (640, 360)] {
                for (name, format) in formats {
                    let stride = format.row_bytes(w) + 8;
                    let src: Vec<u8> = (0..stride * h)
                        .map(|i| {
                            let v = (i * 7 + (i / stride) * 13) % 256;
                            // Keep YUY2 chroma near neutral so colours stay in gamut.
                            match format {
                                InputFormat::Yuy2 if i % 2 == 1 => 108 + (v % 40) as u8,
                                _ => v as u8,
                            }
                        })
                        .collect();
                    let bg = Some(Background::Color([0.0; 3]));
                    assert!(gpu.ingest(0, format, &src, stride, w, h, bg, None).unwrap());
                    let got = gpu.model_input().to_vec();
                    let mut dst = vec![0u8; w * 2 * h];
                    let (a, b) = (vec![f16::ZERO; 4 * mw * mh], vec![f16::ZERO; 4 * mw * mh]);
                    gpu.finish(0, Some((&a, &b)), &mut dst, w * 2).unwrap();

                    let mut frame = PlanarFrame::default();
                    frame.resize(w, h);
                    assert!(unpack(format, &src, stride, &mut frame));
                    let planes: Vec<f32> = frame.planes().concat();
                    let mut small = vec![0.0; 3 * mw * mh];
                    resample(&planes, w, h, &mut small, mw, mh, 3);
                    let mut want = vec![f16::ZERO; small.len()];
                    want.convert_from_f32_slice(&small);

                    let worst = got.iter().zip(&want).map(|(g, w)| g.to_bits().abs_diff(w.to_bits())).max().unwrap();
                    assert!(worst <= 1, "{name} {w}x{h} -> {mw}x{mh}: off by {worst} f16 steps");
                }
            }
        }
    }
}
