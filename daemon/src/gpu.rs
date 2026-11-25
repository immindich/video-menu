//! GPU (wgpu/Vulkan) implementation of the frame pipeline in
//! [`crate::pipeline`]: the camera frame is uploaded, unpacked to planar RGB,
//! run through effects and packed as YUY2 by compute shaders
//! (`pipeline.wgsl`), and the result is read back.
//!
//! Effects that use the matting model split the frame's GPU work in two:
//! the GPU downsamples the frame to the model's input, which is read back
//! and run on the model (NPU), and the model's small output is uploaded for
//! the second half. The full-resolution frame stays on the GPU throughout.

use std::{
    error::Error,
    sync::mpsc,
    time::{Duration, Instant},
};

use bytemuck::{Pod, Zeroable};
use half::f16;

use crate::{
    model::Matting,
    pipeline::{self, Background, BlurGeometry, InputFormat, MAX_BLUR_TAPS},
};

const WORKGROUP: u32 = 16;

/// An effect that runs on the GPU.
pub enum GpuEffect {
    /// Replace the background, using a matting model to find the person
    /// (see [`pipeline::ReplaceBackground`]).
    ReplaceBackground(Matting, Background),
}

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

/// GPU buffers sized for one frame size; reallocated when it changes.
struct Frame {
    width: usize,
    height: usize,
    input: wgpu::Buffer,
    output: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// Samples the blur buffers hold; they grow when a smaller blur needs
    /// a bigger copy of the frame.
    blur_len: usize,
    bind_group: wgpu::BindGroup,
}

pub struct GpuProcessor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    unpack_packed: wgpu::ComputePipeline,
    unpack_yuy2: wgpu::ComputePipeline,
    downsample: wgpu::ComputePipeline,
    unpack_coef: wgpu::ComputePipeline,
    composite_color: wgpu::ComputePipeline,
    blur_down: wgpu::ComputePipeline,
    blur_h: wgpu::ComputePipeline,
    blur_v: wgpu::ComputePipeline,
    composite_blur: wgpu::ComputePipeline,
    pack_yuy2: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    /// The frame at model resolution (f16), written by `downsample`.
    small: wgpu::Buffer,
    small_readback: wgpu::Buffer,
    /// The model's output (f16 A then b), read by the composite passes.
    coef: wgpu::Buffer,
    /// The coefficients unpacked to f32 and interleaved, A then b for each
    /// model pixel, by `unpack_coef`: the composite passes sample them at
    /// every frame pixel, and one load of a vec4 beats four of packed f16.
    coef4: wgpu::Buffer,
    /// Just the alpha coefficients (A and b), which `blur_down` needs. It
    /// runs while the model works on the frame, so it reads the previous
    /// frame's.
    coef_alpha: wgpu::Buffer,
    /// Zero `coef_alpha` before the next frame: the previous frame's matte
    /// is of something else (after a reset) or of nothing (at first).
    clear_alpha: bool,
    effects: Vec<GpuEffect>,
    frame: Option<Frame>,
    /// Size of the last frame; models' temporal state is reset when it changes.
    last_size: (usize, usize),
    /// Tightly packed copy of the input, used when its stride has padding.
    staging: Vec<u8>,
    /// The model input, copied out of `small_readback`.
    small_host: Vec<f16>,
    /// Resolution of the matting model, which sizes `small` and `coef`.
    model_size: (usize, usize),
    adapter_name: String,
}

/// Where the time for one frame went.
struct Times {
    upload: Duration,
    gpu: Duration,
    model: Duration,
}

impl GpuProcessor {
    /// Set up a hardware Vulkan device, with no effects. Fails if there is
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
        let buffer = |label, size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        use wgpu::BufferUsages as U;
        // Placeholders until set_effects knows the model's resolution.
        let model_size = (2, 1);
        let [small, small_readback, coef, coef4, coef_alpha] = model_buffers(&device, model_size);

        Ok(Self {
            unpack_packed: pipeline("unpack_packed"),
            unpack_yuy2: pipeline("unpack_yuy2"),
            downsample: pipeline("downsample"),
            unpack_coef: pipeline("unpack_coef"),
            composite_color: pipeline("composite_color"),
            blur_down: pipeline("blur_down"),
            blur_h: pipeline("blur_h"),
            blur_v: pipeline("blur_v"),
            composite_blur: pipeline("composite_blur"),
            pack_yuy2: pipeline("pack_yuy2"),
            params: buffer("params", size_of::<Params>() as u64, U::UNIFORM | U::COPY_DST),
            small,
            small_readback,
            coef,
            coef4,
            coef_alpha,
            clear_alpha: true,
            device,
            queue,
            layout,
            effects: Vec::new(),
            frame: None,
            last_size: (0, 0),
            staging: Vec::new(),
            small_host: Vec::new(),
            model_size,
            adapter_name: info.name,
        })
    }

    pub fn set_effects(&mut self, effects: Vec<GpuEffect>) {
        let model_size = effects.iter().map(|e| match e {
            GpuEffect::ReplaceBackground(matte, _) => matte.size(),
        }).next();
        if let Some(size) = model_size
            && size != self.model_size
        {
            [self.small, self.small_readback, self.coef, self.coef4, self.coef_alpha] =
                model_buffers(&self.device, size);
            self.clear_alpha = true;
            self.small_host = vec![f16::ZERO; 3 * size.0 * size.1];
            self.model_size = size;
            // The bind group refers to the old buffers.
            self.frame = None;
        }
        self.effects = effects;
        self.last_size = (0, 0);
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// The matting model the effects run, if any.
    pub fn matting_mut(&mut self) -> Option<&mut Matting> {
        self.effects.iter_mut().map(|e| match e {
            GpuEffect::ReplaceBackground(matte, _) => matte,
        }).next()
    }

    /// Change what the effects put behind the person.
    pub fn set_background(&mut self, background: Background) {
        for effect in &mut self.effects {
            let GpuEffect::ReplaceBackground(_, bg) = effect;
            *bg = background;
        }
    }

    /// Forget effects' temporal state, e.g. after switching cameras.
    pub fn reset(&mut self) {
        self.clear_alpha = true;
        for effect in &mut self.effects {
            let GpuEffect::ReplaceBackground(matte, _) = effect;
            matte.reset();
        }
    }

    /// Process one `width`x`height` camera frame into YUY2 in `dst` (rows
    /// `dst_stride` apart). Returns the time spent uploading, waiting for the
    /// GPU, running models, and copying the result out.
    #[allow(clippy::too_many_arguments)]
    pub fn process(
        &mut self,
        input: InputFormat,
        src: &[u8],
        src_stride: usize,
        width: usize,
        height: usize,
        dst: &mut [u8],
        dst_stride: usize,
    ) -> Result<[Duration; 4], String> {
        if !width.is_multiple_of(2) || !pipeline::frame_fits(input, src.len(), src_stride, width, height) {
            return Err(format!("short or odd-sized frame ({} bytes)", src.len()));
        }
        if self.last_size != (width, height) {
            self.reset();
            self.last_size = (width, height);
        }
        let t0 = Instant::now();
        // Only one background effect runs, so its settings can live in
        // the shared parameters.
        let background = self.effects.iter().map(|e| match e {
            GpuEffect::ReplaceBackground(_, bg) => *bg,
        }).next();
        let bg = match background {
            Some(Background::Color(c)) => [c[0], c[1], c[2], 1.0],
            _ => [0.0; 4],
        };
        let blur = match background {
            Some(Background::Blur { radius }) => Some(BlurGeometry::new(width, height, radius)),
            _ => None,
        };
        self.ensure_frame(width, height, blur.map_or(0, |g| g.len()));

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
        let frame = self.frame.as_ref().unwrap();
        self.queue.write_buffer(&frame.input, 0, tight);
        let (r, g, b) = match input {
            InputFormat::Packed { r, g, b, .. } => (r, g, b),
            InputFormat::Yuy2 => (0, 0, 0),
        };
        let mut params = Params {
            width: width as u32,
            height: height as u32,
            src_words: (row_bytes / 4) as u32,
            dst_words: (width / 2) as u32,
            r_shift: 8 * r as u32,
            g_shift: 8 * g as u32,
            b_shift: 8 * b as u32,
            _pad0: 0,
            small_w: self.model_size.0 as u32,
            small_h: self.model_size.1 as u32,
            _pad1: [0; 2],
            bg,
            blur_w: 0,
            blur_h: 0,
            blur_factor: 0,
            blur_taps: 0,
            blur_weights: [0.0; MAX_BLUR_TAPS + 1],
        };
        if let Some(g) = &blur {
            (params.blur_w, params.blur_h) = (g.width as u32, g.height as u32);
            (params.blur_factor, params.blur_taps) = (g.factor as u32, g.taps as u32);
            let weights = g.weights();
            params.blur_weights[..weights.len()].copy_from_slice(&weights);
        }
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let mut times = Times { upload: t0.elapsed(), gpu: Duration::ZERO, model: Duration::ZERO };

        let (w, h) = (width as u32, height as u32);
        let pairs = w / 2;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        if std::mem::take(&mut self.clear_alpha) {
            encoder.clear_buffer(&self.coef_alpha, 0, None);
        }
        match input {
            InputFormat::Packed { .. } => self.dispatch(&mut encoder, &self.unpack_packed, w, h),
            InputFormat::Yuy2 => self.dispatch(&mut encoder, &self.unpack_yuy2, pairs, h),
        }
        // Take the effects out so they can be borrowed alongside `self`.
        let mut effects = std::mem::take(&mut self.effects);
        let mut result = Ok(());
        for effect in &mut effects {
            match effect {
                GpuEffect::ReplaceBackground(matte, background) => {
                    // The blur weighs pixels by the previous frame's matte, so
                    // it can run on the GPU while the model works.
                    let blur = match background {
                        Background::Blur { .. } => blur,
                        Background::Color(_) => None,
                    };
                    result = self.run_model(&mut encoder, matte, &mut times, blur);
                    if result.is_err() {
                        break;
                    }
                    let (mw, mh) = self.model_size;
                    self.dispatch(&mut encoder, &self.unpack_coef, mw as u32, mh as u32);
                    let composite = match background {
                        Background::Color(_) => &self.composite_color,
                        Background::Blur { .. } => &self.composite_blur,
                    };
                    self.dispatch(&mut encoder, composite, w, h);
                }
            }
        }
        self.effects = effects;
        result?;
        self.dispatch(&mut encoder, &self.pack_yuy2, pairs, h);

        let frame = self.frame.as_ref().unwrap();
        let out_bytes = (width * 2 * height) as u64;
        encoder.copy_buffer_to_buffer(&frame.output, 0, &frame.readback, 0, out_bytes);
        let t1 = Instant::now();
        self.queue.submit([encoder.finish()]);
        self.wait_mapped(&frame.readback, out_bytes)?;
        times.gpu += t1.elapsed();

        let t2 = Instant::now();
        {
            let mapped = frame
                .readback
                .slice(..out_bytes)
                .get_mapped_range()
                .map_err(|e| format!("GPU readback failed: {e}"))?;
            let out_row = width * 2;
            for (dst_row, src_row) in dst.chunks_mut(dst_stride).zip(mapped.chunks(out_row)) {
                dst_row[..out_row].copy_from_slice(src_row);
            }
        }
        frame.readback.unmap();
        Ok([times.upload, times.gpu, times.model, t2.elapsed()])
    }

    /// Finish the GPU work recorded in `encoder` with a downsample to the
    /// model's input, run the model on it, and upload its output to `coef`.
    /// With `blur`, the blur of the background is started on the GPU before
    /// the model runs. `encoder` is replaced with a fresh one for the rest
    /// of the frame.
    fn run_model(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        matte: &mut Matting,
        times: &mut Times,
        blur: Option<BlurGeometry>,
    ) -> Result<(), String> {
        let (mw, mh) = self.model_size;
        self.dispatch(encoder, &self.downsample, (mw / 2) as u32, mh as u32);
        let bytes = self.small.size();
        encoder.copy_buffer_to_buffer(&self.small, 0, &self.small_readback, 0, bytes);
        let fresh = self.device.create_command_encoder(&Default::default());
        let finished = std::mem::replace(encoder, fresh).finish();
        let t = Instant::now();
        self.queue.submit([finished]);
        self.wait_mapped(&self.small_readback, bytes)?;
        times.gpu += t.elapsed();
        {
            let mapped = self
                .small_readback
                .slice(..)
                .get_mapped_range()
                .map_err(|e| format!("GPU readback failed: {e}"))?;
            bytemuck::cast_slice_mut(&mut self.small_host).copy_from_slice(&mapped);
        }
        self.small_readback.unmap();

        if let Some(g) = blur {
            let mut blur_encoder = self.device.create_command_encoder(&Default::default());
            let (bw, bh) = (g.width as u32, g.height as u32);
            self.dispatch(&mut blur_encoder, &self.blur_down, bw, bh);
            self.dispatch(&mut blur_encoder, &self.blur_h, bw, bh);
            self.dispatch(&mut blur_encoder, &self.blur_v, bw, bh);
            self.queue.submit([blur_encoder.finish()]);
        }
        let (a, b) = matte.run(&self.small_host)?;
        let t = Instant::now();
        self.queue.write_buffer(&self.coef, 0, bytemuck::cast_slice(a));
        self.queue.write_buffer(&self.coef, (a.len() * 2) as u64, bytemuck::cast_slice(b));
        times.upload += t.elapsed();
        times.model += matte.last_time();
        Ok(())
    }

    /// Record `pipeline` over a `x`x`y` grid of invocations.
    fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline, x: u32, y: u32) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_bind_group(0, &self.frame.as_ref().unwrap().bind_group, &[]);
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(x.div_ceil(WORKGROUP), y.div_ceil(WORKGROUP), 1);
    }

    /// Wait for submitted work, then map the first `len` bytes of `buffer`.
    fn wait_mapped(&self, buffer: &wgpu::Buffer, len: u64) -> Result<(), String> {
        let (tx, rx) = mpsc::channel();
        buffer.slice(..len).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("GPU poll failed: {e}"))?;
        rx.recv()
            .map_err(|_| "GPU readback was dropped".to_string())?
            .map_err(|e| format!("GPU readback failed: {e}"))
    }

    /// Make sure the frame buffers fit a `width`x`height` frame and the
    /// blur buffers hold `blur_len` samples.
    fn ensure_frame(&mut self, width: usize, height: usize, blur_len: usize) {
        if self
            .frame
            .as_ref()
            .is_some_and(|f| (f.width, f.height) == (width, height) && f.blur_len >= blur_len)
        {
            return;
        }
        let pixels = (width * height) as u64;
        let buffer = |label, size, usage| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        use wgpu::BufferUsages as U;
        // Packed RGB is the largest input: 4 bytes per pixel.
        let input = buffer("input", pixels * 4, U::STORAGE | U::COPY_DST);
        let planar = buffer("planar", pixels * 3 * 4, U::STORAGE);
        let output = buffer("output", pixels * 2, U::STORAGE | U::COPY_SRC);
        let readback = buffer("readback", pixels * 2, U::MAP_READ | U::COPY_DST);
        // Keep what blur buffers there were, if bigger. (Buffers can't be
        // empty.)
        let same_size = self.frame.as_ref().filter(|f| (f.width, f.height) == (width, height));
        let blur_len = blur_len.max(same_size.map_or(0, |f| f.blur_len)).max(1);
        let blur_bytes = (blur_len * 4) as u64;
        let blur_a = buffer("blur a", blur_bytes, U::STORAGE);
        let blur_b = buffer("blur b", blur_bytes, U::STORAGE);
        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pipeline"),
            layout: &self.layout,
            entries: &[
                entry(0, &self.params),
                entry(1, &input),
                entry(2, &planar),
                entry(3, &output),
                entry(4, &self.small),
                entry(5, &self.coef),
                entry(8, &self.coef4),
                entry(9, &self.coef_alpha),
                entry(6, &blur_a),
                entry(7, &blur_b),
            ],
        });
        self.frame = Some(Frame { width, height, input, output, readback, blur_len, bind_group });
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
