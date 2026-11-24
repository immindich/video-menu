//! GPU (wgpu/Vulkan) implementation of the frame pipeline in
//! [`crate::pipeline`]: the camera frame is uploaded, unpacked to planar RGB,
//! run through effects and packed as YUY2 by compute shaders
//! (`pipeline.wgsl`), and the result is read back.

use std::{
    error::Error,
    sync::mpsc,
    time::{Duration, Instant},
};

use bytemuck::{Pod, Zeroable};

use crate::pipeline::{self, InputFormat};

const WORKGROUP: u32 = 16;

/// An effect that runs on the GPU.
#[derive(Clone, Copy, Debug)]
pub enum GpuEffect {
    VerticalFlip,
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
    _pad: u32,
}

/// GPU buffers sized for one frame size; reallocated when it changes.
struct Frame {
    width: usize,
    height: usize,
    input: wgpu::Buffer,
    output: wgpu::Buffer,
    readback: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

pub struct GpuProcessor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    unpack_packed: wgpu::ComputePipeline,
    unpack_yuy2: wgpu::ComputePipeline,
    vertical_flip: wgpu::ComputePipeline,
    pack_yuy2: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    effects: Vec<GpuEffect>,
    frame: Option<Frame>,
    /// Tightly packed copy of the input, used when its stride has padding.
    staging: Vec<u8>,
    adapter_name: String,
}

impl GpuProcessor {
    /// Set up a hardware Vulkan device. Fails if there is none (a software
    /// rasterizer such as llvmpipe doesn't count).
    pub fn new(effects: Vec<GpuEffect>) -> Result<Self, Box<dyn Error>> {
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
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            unpack_packed: pipeline("unpack_packed"),
            unpack_yuy2: pipeline("unpack_yuy2"),
            vertical_flip: pipeline("vertical_flip"),
            pack_yuy2: pipeline("pack_yuy2"),
            device,
            queue,
            layout,
            params,
            effects,
            frame: None,
            staging: Vec::new(),
            adapter_name: info.name,
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Process one `width`x`height` camera frame into YUY2 in `dst` (rows
    /// `dst_stride` apart). Returns the time spent uploading, waiting for the
    /// GPU, and copying the result out.
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
    ) -> Result<[Duration; 3], String> {
        if !width.is_multiple_of(2) || !pipeline::frame_fits(input, src.len(), src_stride, width, height) {
            return Err(format!("short or odd-sized frame ({} bytes)", src.len()));
        }
        let t0 = Instant::now();
        self.ensure_frame(width, height);
        let frame = self.frame.as_ref().unwrap();

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
        self.queue.write_buffer(&frame.input, 0, tight);
        let (r, g, b) = match input {
            InputFormat::Packed { r, g, b, .. } => (r, g, b),
            InputFormat::Yuy2 => (0, 0, 0),
        };
        let params = Params {
            width: width as u32,
            height: height as u32,
            src_words: (row_bytes / 4) as u32,
            dst_words: (width / 2) as u32,
            r_shift: 8 * r as u32,
            g_shift: 8 * g as u32,
            b_shift: 8 * b as u32,
            _pad: 0,
        };
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, &frame.bind_group, &[]);
            let (w, h) = (width as u32, height as u32);
            let pairs = w / 2;
            match input {
                InputFormat::Packed { .. } => {
                    pass.set_pipeline(&self.unpack_packed);
                    pass.dispatch_workgroups(w.div_ceil(WORKGROUP), h.div_ceil(WORKGROUP), 1);
                }
                InputFormat::Yuy2 => {
                    pass.set_pipeline(&self.unpack_yuy2);
                    pass.dispatch_workgroups(pairs.div_ceil(WORKGROUP), h.div_ceil(WORKGROUP), 1);
                }
            }
            for effect in &self.effects {
                match effect {
                    GpuEffect::VerticalFlip => {
                        pass.set_pipeline(&self.vertical_flip);
                        pass.dispatch_workgroups(
                            w.div_ceil(WORKGROUP),
                            (h / 2).div_ceil(WORKGROUP),
                            1,
                        );
                    }
                }
            }
            pass.set_pipeline(&self.pack_yuy2);
            pass.dispatch_workgroups(pairs.div_ceil(WORKGROUP), h.div_ceil(WORKGROUP), 1);
        }
        let out_bytes = (width * 2 * height) as u64;
        encoder.copy_buffer_to_buffer(&frame.output, 0, &frame.readback, 0, out_bytes);
        let t1 = Instant::now();

        self.queue.submit([encoder.finish()]);
        let (tx, rx) = mpsc::channel();
        frame
            .readback
            .slice(..out_bytes)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("GPU poll failed: {e}"))?;
        rx.recv()
            .map_err(|_| "GPU readback was dropped".to_string())?
            .map_err(|e| format!("GPU readback failed: {e}"))?;
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
        Ok([t1 - t0, t2 - t1, t2.elapsed()])
    }

    fn ensure_frame(&mut self, width: usize, height: usize) {
        if self
            .frame
            .as_ref()
            .is_some_and(|f| (f.width, f.height) == (width, height))
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
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pipeline"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: planar.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: output.as_entire_binding() },
            ],
        });
        self.frame = Some(Frame { width, height, input, output, readback, bind_group });
    }
}
