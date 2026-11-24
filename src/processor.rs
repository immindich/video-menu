//! Chooses where frames are processed: on the GPU ([`crate::gpu`]) when one
//! is available, otherwise on the CPU ([`crate::pipeline`]). Both produce the
//! same YUY2 output (to within rounding).
//!
//! Set `VIDEO_MENU_BACKEND=cpu` or `=gpu` to force one; `gpu` fails instead
//! of falling back.

use std::time::{Duration, Instant};

use crate::{
    gpu::{GpuEffect, GpuProcessor},
    pipeline::{self, Effect, InputFormat, PlanarFrame, VerticalFlip},
};

pub struct CpuProcessor {
    planar: PlanarFrame,
    effects: Vec<Box<dyn Effect>>,
}

pub enum Processor {
    Cpu(CpuProcessor),
    Gpu(Box<GpuProcessor>),
}

impl Processor {
    pub fn cpu() -> Self {
        Self::Cpu(CpuProcessor {
            planar: PlanarFrame::default(),
            effects: vec![Box::new(VerticalFlip)],
        })
    }

    pub fn gpu() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::Gpu(Box::new(GpuProcessor::new(vec![GpuEffect::VerticalFlip])?)))
    }

    /// The backend chosen by `VIDEO_MENU_BACKEND`, defaulting to the GPU with
    /// a fallback to the CPU.
    pub fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        match std::env::var("VIDEO_MENU_BACKEND").as_deref() {
            Ok("cpu") => Ok(Self::cpu()),
            Ok("gpu") => Self::gpu(),
            Ok(other) if !other.is_empty() => {
                Err(format!("VIDEO_MENU_BACKEND must be cpu or gpu, not {other:?}").into())
            }
            _ => Ok(Self::gpu().unwrap_or_else(|e| {
                eprintln!("processor: no GPU ({e}); using the CPU");
                Self::cpu()
            })),
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Self::Cpu(_) => "CPU".into(),
            Self::Gpu(gpu) => format!("GPU ({})", gpu.adapter_name()),
        }
    }

    /// Names of the three stages [`process`](Self::process) times.
    pub fn stage_names(&self) -> [&'static str; 3] {
        match self {
            Self::Cpu(_) => ["unpack", "effects", "pack"],
            Self::Gpu(_) => ["upload", "gpu", "readback"],
        }
    }

    /// Turn one `width`x`height` camera frame into YUY2 in `dst` (rows
    /// `dst_stride` apart, `dst_stride * height` bytes). Returns the time of
    /// each stage named by [`stage_names`](Self::stage_names).
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
        match self {
            Self::Gpu(gpu) => gpu.process(input, src, src_stride, width, height, dst, dst_stride),
            Self::Cpu(cpu) => {
                let t0 = Instant::now();
                cpu.planar.resize(width, height);
                if !pipeline::unpack(input, src, src_stride, &mut cpu.planar) {
                    return Err(format!("short frame ({} bytes)", src.len()));
                }
                let t1 = Instant::now();
                for effect in &mut cpu.effects {
                    effect.apply(&mut cpu.planar);
                }
                let t2 = Instant::now();
                pipeline::pack_yuy2_bt601(&cpu.planar, dst, dst_stride);
                Ok([t1 - t0, t2 - t1, t2.elapsed()])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pipewire::spa::param::video::VideoFormat;

    /// A deterministic test pattern with gradients and noise-like detail.
    fn pattern(format: InputFormat, width: usize, height: usize, stride: usize) -> Vec<u8> {
        let mut src = vec![0u8; stride * height];
        for (y, row) in src.chunks_mut(stride).enumerate() {
            let bytes = format.row_bytes(width);
            for (x, b) in row[..bytes].iter_mut().enumerate() {
                let v = (x * 7 + y * 13 + (x * y) % 31) % 256;
                *b = match format {
                    // Keep YUY2 chroma near neutral so colours stay in gamut.
                    InputFormat::Yuy2 if x % 2 == 1 => 108 + (v % 40) as u8,
                    InputFormat::Yuy2 => 30 + (v % 190) as u8,
                    _ => v as u8,
                };
            }
        }
        src
    }

    fn gpu_or_skip() -> Option<Processor> {
        match Processor::gpu() {
            Ok(gpu) => Some(gpu),
            Err(e) => {
                eprintln!("skipping: no GPU ({e})");
                None
            }
        }
    }

    fn formats() -> [(&'static str, InputFormat); 3] {
        [
            ("BGRx", InputFormat::from_video_format(VideoFormat::BGRx).unwrap()),
            ("RGBx", InputFormat::from_video_format(VideoFormat::RGBx).unwrap()),
            ("YUY2", InputFormat::Yuy2),
        ]
    }

    #[test]
    fn gpu_matches_cpu() {
        let Some(mut gpu) = gpu_or_skip() else { return };
        let mut cpu = Processor::cpu();
        // Odd sizes in workgroup terms, and padded input rows.
        let (w, h) = (70, 37);
        for (name, format) in formats() {
            let stride = format.row_bytes(w) + 12;
            let src = pattern(format, w, h, stride);
            let (mut a, mut b) = (vec![0u8; w * 2 * h], vec![0u8; w * 2 * h]);
            cpu.process(format, &src, stride, w, h, &mut a, w * 2).unwrap();
            gpu.process(format, &src, stride, w, h, &mut b, w * 2).unwrap();
            let worst = a.iter().zip(&b).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
            let off = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            assert!(worst <= 1, "{name}: GPU differs from CPU by up to {worst}");
            eprintln!("{name}: {off} of {} bytes differ by 1", a.len());
        }
    }

    /// Compare CPU and GPU speed on 1080p frames. Run with
    /// `cargo test --release -- --ignored --nocapture bench`.
    #[test]
    #[ignore]
    fn bench_cpu_vs_gpu() {
        const FRAMES: u32 = 100;
        let (w, h) = (1920, 1080);
        let mut backends = vec![("CPU", Processor::cpu())];
        backends.extend(gpu_or_skip().map(|g| ("GPU", g)));
        let mut dst = vec![0u8; w * 2 * h];
        for (name, format) in formats().into_iter().filter(|(n, _)| *n != "RGBx") {
            let stride = format.row_bytes(w);
            let src = pattern(format, w, h, stride);
            for (backend, proc) in &mut backends {
                proc.process(format, &src, stride, w, h, &mut dst, w * 2).unwrap(); // warm up
                let mut stages = [Duration::ZERO; 3];
                let start = Instant::now();
                for _ in 0..FRAMES {
                    let t = proc.process(format, &src, stride, w, h, &mut dst, w * 2).unwrap();
                    for (s, t) in stages.iter_mut().zip(t) {
                        *s += t;
                    }
                }
                let ms = |d: Duration| d.as_secs_f64() * 1e3 / FRAMES as f64;
                let names = proc.stage_names();
                eprintln!(
                    "{name} 1080p {backend}: {:.2} ms/frame ({} {:.2}, {} {:.2}, {} {:.2})",
                    ms(start.elapsed()),
                    names[0],
                    ms(stages[0]),
                    names[1],
                    ms(stages[1]),
                    names[2],
                    ms(stages[2]),
                );
            }
        }
    }
}
