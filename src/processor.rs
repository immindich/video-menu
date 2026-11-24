//! Turns camera frames into processed YUY2 frames. MJPG frames are decoded
//! on the CPU (libjpeg-turbo); the rest of the pipeline runs on the GPU
//! ([`crate::gpu`]) when one is available, otherwise on the CPU
//! ([`crate::pipeline`]). Both backends produce the same output.
//!
//! Set `VIDEO_MENU_BACKEND=cpu` or `=gpu` to force a backend; `gpu` fails
//! instead of falling back.

use std::time::{Duration, Instant};

use pipewire::spa::param::video::VideoFormat;

use crate::{
    gpu::{GpuEffect, GpuProcessor},
    pipeline::{self, Effect, InputFormat, PlanarFrame, VerticalFlip},
};

/// The encoding of an incoming camera frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameFormat {
    Raw(InputFormat),
    Mjpg,
}

struct CpuBackend {
    planar: PlanarFrame,
    effects: Vec<Box<dyn Effect>>,
}

enum Backend {
    Cpu(CpuBackend),
    Gpu(Box<GpuProcessor>),
}

pub struct Processor {
    backend: Backend,
    /// JPEG decoder, created on the first MJPG frame.
    jpeg: Option<turbojpeg::Decompressor>,
    /// The last decoded MJPG frame, as RGBx.
    decoded: Vec<u8>,
}

impl Processor {
    fn with_backend(backend: Backend) -> Self {
        Self { backend, jpeg: None, decoded: Vec::new() }
    }

    pub fn cpu() -> Self {
        Self::with_backend(Backend::Cpu(CpuBackend {
            planar: PlanarFrame::default(),
            effects: vec![Box::new(VerticalFlip)],
        }))
    }

    pub fn gpu() -> Result<Self, Box<dyn std::error::Error>> {
        let gpu = GpuProcessor::new(vec![GpuEffect::VerticalFlip])?;
        Ok(Self::with_backend(Backend::Gpu(Box::new(gpu))))
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
        match &self.backend {
            Backend::Cpu(_) => "CPU".into(),
            Backend::Gpu(gpu) => format!("GPU ({})", gpu.adapter_name()),
        }
    }

    /// Names of the four stages [`process`](Self::process) times. The first,
    /// `decode`, is zero for raw frames.
    pub fn stage_names(&self) -> [&'static str; 4] {
        match &self.backend {
            Backend::Cpu(_) => ["decode", "unpack", "effects", "pack"],
            Backend::Gpu(_) => ["decode", "upload", "gpu", "readback"],
        }
    }

    /// Turn one `width`x`height` camera frame into YUY2 in `dst` (rows
    /// `dst_stride` apart, `dst_stride * height` bytes). `src_stride` is
    /// ignored for MJPG. Returns the time of each stage named by
    /// [`stage_names`](Self::stage_names).
    #[allow(clippy::too_many_arguments)]
    pub fn process(
        &mut self,
        format: FrameFormat,
        src: &[u8],
        src_stride: usize,
        width: usize,
        height: usize,
        dst: &mut [u8],
        dst_stride: usize,
    ) -> Result<[Duration; 4], String> {
        let (input, src, src_stride, decode) = match format {
            FrameFormat::Raw(input) => (input, src, src_stride, Duration::ZERO),
            FrameFormat::Mjpg => {
                let start = Instant::now();
                self.decode_jpeg(src, width, height)?;
                let rgbx = InputFormat::from_video_format(VideoFormat::RGBx).unwrap();
                (rgbx, self.decoded.as_slice(), width * 4, start.elapsed())
            }
        };
        let [a, b, c] = match &mut self.backend {
            Backend::Gpu(gpu) => gpu.process(input, src, src_stride, width, height, dst, dst_stride)?,
            Backend::Cpu(cpu) => {
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
                [t1 - t0, t2 - t1, t2.elapsed()]
            }
        };
        Ok([decode, a, b, c])
    }

    /// Decode a JPEG frame into `self.decoded` as RGBx, checking it is the
    /// size the camera negotiated.
    fn decode_jpeg(&mut self, jpeg: &[u8], width: usize, height: usize) -> Result<(), String> {
        let decoder = match &mut self.jpeg {
            Some(decoder) => decoder,
            None => self
                .jpeg
                .insert(turbojpeg::Decompressor::new().map_err(|e| format!("JPEG decoder: {e}"))?),
        };
        let header = decoder.read_header(jpeg).map_err(|e| format!("bad JPEG frame: {e}"))?;
        if (header.width, header.height) != (width, height) {
            return Err(format!(
                "JPEG frame is {}x{}, expected {width}x{height}",
                header.width, header.height
            ));
        }
        self.decoded.resize(width * 4 * height, 0);
        let image = turbojpeg::Image {
            pixels: self.decoded.as_mut_slice(),
            width,
            pitch: width * 4,
            height,
            format: turbojpeg::PixelFormat::RGBX,
        };
        decoder
            .decompress(jpeg, image)
            .map_err(|e| format!("bad JPEG frame: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic test pattern with gradients and fine detail.
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

    /// A smooth RGBx image, which JPEG reproduces closely.
    fn smooth_rgbx(width: usize, height: usize) -> Vec<u8> {
        let mut px = Vec::with_capacity(width * height * 4);
        for y in 0..height {
            for x in 0..width {
                let (fx, fy) = (x as f32 / width as f32, y as f32 / height as f32);
                px.extend([(255.0 * fx) as u8, (255.0 * fy) as u8, (255.0 * (1.0 - fx * fy)) as u8, 0]);
            }
        }
        px
    }

    fn jpeg(rgbx: &[u8], width: usize, height: usize, quality: i32, subsamp: turbojpeg::Subsamp) -> Vec<u8> {
        let image = turbojpeg::Image {
            pixels: rgbx,
            width,
            pitch: width * 4,
            height,
            format: turbojpeg::PixelFormat::RGBX,
        };
        turbojpeg::compress(image, quality, subsamp).unwrap().to_vec()
    }

    fn rgbx() -> InputFormat {
        InputFormat::from_video_format(VideoFormat::RGBx).unwrap()
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

    fn raw_formats() -> [(&'static str, InputFormat); 3] {
        [
            ("BGRx", InputFormat::from_video_format(VideoFormat::BGRx).unwrap()),
            ("RGBx", rgbx()),
            ("YUY2", InputFormat::Yuy2),
        ]
    }

    #[test]
    fn gpu_matches_cpu() {
        let Some(mut gpu) = gpu_or_skip() else { return };
        let mut cpu = Processor::cpu();
        // Odd sizes in workgroup terms, and padded input rows.
        let (w, h) = (70, 37);
        let mut cases: Vec<(&str, FrameFormat, Vec<u8>, usize)> = raw_formats()
            .into_iter()
            .map(|(name, format)| {
                let stride = format.row_bytes(w) + 12;
                (name, FrameFormat::Raw(format), pattern(format, w, h, stride), stride)
            })
            .collect();
        let mjpg = jpeg(&pattern(rgbx(), w, h, w * 4), w, h, 90, turbojpeg::Subsamp::Sub2x1);
        cases.push(("MJPG", FrameFormat::Mjpg, mjpg, 0));
        for (name, format, src, stride) in cases {
            let (mut a, mut b) = (vec![0u8; w * 2 * h], vec![0u8; w * 2 * h]);
            cpu.process(format, &src, stride, w, h, &mut a, w * 2).unwrap();
            gpu.process(format, &src, stride, w, h, &mut b, w * 2).unwrap();
            let worst = a.iter().zip(&b).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
            let off = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            assert!(worst <= 1, "{name}: GPU differs from CPU by up to {worst}");
            eprintln!("{name}: {off} of {} bytes differ by 1", a.len());
        }
    }

    #[test]
    fn mjpg_matches_raw() {
        let (w, h) = (64, 48);
        let px = smooth_rgbx(w, h);
        let mjpg = jpeg(&px, w, h, 100, turbojpeg::Subsamp::None);
        let mut cpu = Processor::cpu();
        let (mut from_raw, mut from_jpeg) = (vec![0u8; w * 2 * h], vec![0u8; w * 2 * h]);
        let t = cpu.process(FrameFormat::Raw(rgbx()), &px, w * 4, w, h, &mut from_raw, w * 2).unwrap();
        assert_eq!(t[0], Duration::ZERO, "raw frames have no decode stage");
        cpu.process(FrameFormat::Mjpg, &mjpg, 0, w, h, &mut from_jpeg, w * 2).unwrap();
        let worst = from_raw.iter().zip(&from_jpeg).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
        assert!(worst <= 3, "decoded JPEG differs from the raw frame by up to {worst}");
    }

    #[test]
    fn mjpg_rejects_bad_frames() {
        let (w, h) = (32, 16);
        let mjpg = jpeg(&smooth_rgbx(w, h), w, h, 90, turbojpeg::Subsamp::Sub2x1);
        let mut cpu = Processor::cpu();
        let mut dst = vec![0u8; 64 * 2 * 16];
        let err = cpu.process(FrameFormat::Mjpg, &mjpg, 0, 64, 16, &mut dst, 128).unwrap_err();
        assert!(err.contains("32x16"), "{err}");
        let err = cpu.process(FrameFormat::Mjpg, b"not a jpeg", 0, w, h, &mut dst, w * 2).unwrap_err();
        assert!(err.contains("bad JPEG"), "{err}");
        // A truncated frame, as a camera might send after a USB hiccup.
        assert!(cpu.process(FrameFormat::Mjpg, &mjpg[..mjpg.len() / 2], 0, w, h, &mut dst, w * 2).is_err());
    }

    /// A realistic 1080p webcam MJPG frame: `testimage.jpg` re-encoded at
    /// quality 85 with 4:2:2 chroma, or a synthetic pattern without it.
    fn bench_jpeg(w: usize, h: usize) -> Vec<u8> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/testimage.jpg");
        let px = match std::fs::read(path) {
            Ok(jpg) => {
                let img = turbojpeg::decompress(&jpg, turbojpeg::PixelFormat::RGBX).unwrap();
                assert_eq!((img.width, img.height), (w, h), "testimage.jpg must be {w}x{h}");
                img.pixels
            }
            Err(_) => {
                eprintln!("no testimage.jpg; benchmarking MJPG on a synthetic pattern");
                pattern(rgbx(), w, h, w * 4)
            }
        };
        jpeg(&px, w, h, 85, turbojpeg::Subsamp::Sub2x1)
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
        let mjpg = bench_jpeg(w, h);
        eprintln!("MJPG frame: {} KiB", mjpg.len() / 1024);
        let cases = [
            ("BGRx", FrameFormat::Raw(raw_formats()[0].1), pattern(raw_formats()[0].1, w, h, w * 4), w * 4),
            ("YUY2", FrameFormat::Raw(InputFormat::Yuy2), pattern(InputFormat::Yuy2, w, h, w * 2), w * 2),
            ("MJPG", FrameFormat::Mjpg, mjpg, 0),
        ];
        for (name, format, src, stride) in &cases {
            for (backend, proc) in &mut backends {
                proc.process(*format, src, *stride, w, h, &mut dst, w * 2).unwrap(); // warm up
                let mut stages = [Duration::ZERO; 4];
                let start = Instant::now();
                for _ in 0..FRAMES {
                    let t = proc.process(*format, src, *stride, w, h, &mut dst, w * 2).unwrap();
                    for (s, t) in stages.iter_mut().zip(t) {
                        *s += t;
                    }
                }
                let ms = |d: Duration| d.as_secs_f64() * 1e3 / FRAMES as f64;
                let stages: Vec<String> = proc
                    .stage_names()
                    .iter()
                    .zip(stages)
                    .map(|(n, t)| format!("{n} {:.2}", ms(t)))
                    .collect();
                eprintln!(
                    "{name} 1080p {backend}: {:.2} ms/frame ({})",
                    ms(start.elapsed()),
                    stages.join(", ")
                );
            }
        }
    }
}
