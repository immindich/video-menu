//! Turns camera frames into processed YUY2 frames. MJPG frames are decoded
//! on the CPU (libjpeg-turbo); the rest of the pipeline runs on the GPU
//! ([`crate::gpu`]) when one is available, otherwise on the CPU
//! ([`crate::pipeline`]). Both backends produce the same output.
//!
//! Set `VIDEO_MENU_BACKEND=cpu` or `=gpu` to force a backend; `gpu` fails
//! instead of falling back.

use std::time::{Duration, Instant};

use pipewire::spa::param::video::VideoFormat;

use video_menu_protocol::ModelChoice;

use crate::{
    gpu::{GpuEffect, GpuProcessor},
    model::{self, Matting},
    pipeline::{self, Background, Effect, InputFormat, PlanarFrame, ReplaceBackground},
};

/// What to do to each frame.
pub enum Effects {
    /// Nothing: frames pass through unchanged.
    None,
    /// Replace the background, finding it with a matting model.
    Background(Matting, Background),
}

impl Effects {
    /// The effects for `effect`, loading the matting model if it needs one.
    /// `background` is what a background effect starts out with.
    pub fn load(
        background: Option<Background>,
        model: ModelChoice,
        reuse: bool,
        smoothing: f32,
    ) -> Result<(Self, String), String> {
        let Some(background) = background else {
            return Ok((Self::None, "none".into()));
        };
        let (matte, description) = model::load(model, smoothing)?;
        let reuse_note = if reuse { ", reusing each matte for the next frame" } else { "" };
        let what = match background {
            Background::Color(_) => "solid colour background",
            Background::Blur { .. } => "blurred background",
        };
        Ok((
            Self::Background(Matting::new(matte, reuse), background),
            format!("{what} with {description}{reuse_note}"),
        ))
    }

    fn for_cpu(self) -> Vec<Box<dyn Effect>> {
        match self {
            Self::None => Vec::new(),
            Self::Background(matte, bg) => vec![Box::new(ReplaceBackground::new(matte, bg))],
        }
    }

    fn for_gpu(self) -> Vec<GpuEffect> {
        match self {
            Self::None => Vec::new(),
            Self::Background(matte, bg) => vec![GpuEffect::ReplaceBackground(matte, bg)],
        }
    }
}

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
    /// Makes the model's input from MJPG frames, before the full decode.
    early: EarlyInput,
}

impl Processor {
    fn with_backend(backend: Backend) -> Self {
        Self { backend, jpeg: None, decoded: Vec::new(), early: EarlyInput::default() }
    }

    pub fn cpu(effects: Effects) -> Self {
        Self::with_backend(Backend::Cpu(CpuBackend {
            planar: PlanarFrame::default(),
            effects: effects.for_cpu(),
        }))
    }

    #[cfg(test)]
    pub fn gpu(effects: Effects) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self::on_gpu(GpuProcessor::new()?, effects))
    }

    fn on_gpu(mut gpu: GpuProcessor, effects: Effects) -> Self {
        gpu.set_effects(effects.for_gpu());
        Self::with_backend(Backend::Gpu(Box::new(gpu)))
    }

    /// The backend chosen by `VIDEO_MENU_BACKEND`, defaulting to the GPU with
    /// a fallback to the CPU.
    pub fn from_env(effects: Effects) -> Result<Self, Box<dyn std::error::Error>> {
        let gpu = match std::env::var("VIDEO_MENU_BACKEND").as_deref() {
            Ok("cpu") => None,
            Ok("gpu") => Some(GpuProcessor::new()?),
            Ok(other) if !other.is_empty() => {
                return Err(format!("VIDEO_MENU_BACKEND must be cpu or gpu, not {other:?}").into());
            }
            _ => GpuProcessor::new()
                .map_err(|e| eprintln!("processor: no GPU ({e}); using the CPU"))
                .ok(),
        };
        Ok(match gpu {
            Some(gpu) => Self::on_gpu(gpu, effects),
            None => Self::cpu(effects),
        })
    }

    /// Replace the effects, e.g. to switch models.
    pub fn set_effects(&mut self, effects: Effects) {
        match &mut self.backend {
            Backend::Cpu(cpu) => cpu.effects = effects.for_cpu(),
            Backend::Gpu(gpu) => gpu.set_effects(effects.for_gpu()),
        }
    }

    /// The matting model the effects run, if any, to change its settings.
    pub fn matting_mut(&mut self) -> Option<&mut Matting> {
        match &mut self.backend {
            Backend::Cpu(cpu) => cpu.effects.iter_mut().find_map(|e| e.matting_mut()),
            Backend::Gpu(gpu) => gpu.matting_mut(),
        }
    }

    /// Change what the background effect, if any, puts behind the person.
    pub fn set_background(&mut self, background: Background) {
        match &mut self.backend {
            Backend::Cpu(cpu) => cpu.effects.iter_mut().for_each(|e| e.set_background(background)),
            Backend::Gpu(gpu) => gpu.set_background(background),
        }
    }

    /// Forget effects' temporal state, e.g. after switching cameras.
    pub fn reset(&mut self) {
        match &mut self.backend {
            Backend::Cpu(cpu) => cpu.effects.iter_mut().for_each(|e| e.reset()),
            Backend::Gpu(gpu) => gpu.reset(),
        }
    }

    pub fn describe(&self) -> String {
        match &self.backend {
            Backend::Cpu(_) => "CPU".into(),
            Backend::Gpu(gpu) => format!("GPU ({})", gpu.adapter_name()),
        }
    }

    /// Names of the stages [`process`](Self::process) times. `decode` is
    /// zero for raw frames and `model` without a model-based effect.
    pub fn stage_names(&self) -> [&'static str; 5] {
        match &self.backend {
            Backend::Cpu(_) => ["decode", "unpack", "effects", "model", "pack"],
            Backend::Gpu(_) => ["decode", "upload", "gpu", "model", "readback"],
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
    ) -> Result<[Duration; 5], String> {
        let mut early = false;
        let (input, src, src_stride, decode) = match format {
            FrameFormat::Raw(input) => (input, src, src_stride, Duration::ZERO),
            FrameFormat::Mjpg => {
                let start = Instant::now();
                // The model's input first, from a scaled-down decode, so a
                // pipelined model can start before the full decode.
                if let Some(size) = self.model_size() {
                    early = self.early.make(src, (width, height), size)?;
                }
                self.decode_jpeg(src, width, height)?;
                let rgbx = InputFormat::from_video_format(VideoFormat::RGBx).unwrap();
                (rgbx, self.decoded.as_slice(), width * 4, start.elapsed())
            }
        };
        let model_input = early.then_some(self.early.input.as_slice());
        let [a, b, model, c] = match &mut self.backend {
            Backend::Gpu(gpu) => gpu.process(input, src, src_stride, width, height, model_input, dst, dst_stride)?,
            Backend::Cpu(cpu) => {
                let t0 = Instant::now();
                cpu.planar.resize(width, height);
                if !pipeline::unpack(input, src, src_stride, &mut cpu.planar) {
                    return Err(format!("short frame ({} bytes)", src.len()));
                }
                let t1 = Instant::now();
                let mut model = Duration::ZERO;
                for effect in &mut cpu.effects {
                    effect.apply_with_input(&mut cpu.planar, model_input)?;
                    model += effect.model_time();
                }
                let t2 = Instant::now();
                pipeline::pack_yuy2_bt601(&cpu.planar, dst, dst_stride);
                [t1 - t0, (t2 - t1).saturating_sub(model), model, t2.elapsed()]
            }
        };
        Ok([decode, a, b, model, c])
    }

    /// The resolution of the model the effects run, if any.
    fn model_size(&mut self) -> Option<(usize, usize)> {
        match &mut self.backend {
            Backend::Cpu(cpu) => cpu.effects.iter().find_map(|e| e.model_size()),
            Backend::Gpu(gpu) => gpu.matting_mut().map(|m| m.size()),
        }
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

/// Makes a model's input from a JPEG frame by decoding it scaled down:
/// libjpeg-turbo scales in the DCT, skipping most of the decode, so the
/// model can start long before the full-resolution decode is done.
#[derive(Default)]
struct EarlyInput {
    decoder: Option<turbojpeg::Decompressor>,
    rgbx: Vec<u8>,
    frame: PlanarFrame,
    small: Vec<f32>,
    input: Vec<half::f16>,
}

impl EarlyInput {
    /// The scale to decode a `size` frame at for a model of `model` size:
    /// the smallest at least as big as the model, or `None` if only the
    /// full size is.
    fn scale(size: (usize, usize), model: (usize, usize)) -> Option<turbojpeg::ScalingFactor> {
        turbojpeg::Decompressor::supported_scaling_factors()
            .into_iter()
            .filter(|f| f.num() < f.denom())
            .filter(|f| f.scale(size.0) >= model.0 && f.scale(size.1) >= model.1)
            .min_by(|a, b| (a.num() * b.denom()).cmp(&(b.num() * a.denom())))
    }

    /// Make the model input for a `size` JPEG frame in `self.input`.
    /// Returns false if it can't be made smaller than a full decode (the
    /// backend makes it from the full frame then).
    fn make(&mut self, jpeg: &[u8], size: (usize, usize), model: (usize, usize)) -> Result<bool, String> {
        let Some(factor) = Self::scale(size, model) else {
            return Ok(false);
        };
        let decoder = match &mut self.decoder {
            Some(decoder) => decoder,
            None => self.decoder.insert(turbojpeg::Decompressor::new().map_err(|e| format!("JPEG decoder: {e}"))?),
        };
        decoder.set_scaling_factor(factor).map_err(|e| format!("JPEG decoder: {e}"))?;
        let header = decoder.read_header(jpeg).map_err(|e| format!("bad JPEG frame: {e}"))?;
        if (header.width, header.height) != size {
            return Err(format!("JPEG frame is {}x{}, expected {}x{}", header.width, header.height, size.0, size.1));
        }
        let (w, h) = (factor.scale(size.0), factor.scale(size.1));
        self.rgbx.resize(w * 4 * h, 0);
        let image = turbojpeg::Image {
            pixels: self.rgbx.as_mut_slice(),
            width: w,
            pitch: w * 4,
            height: h,
            format: turbojpeg::PixelFormat::RGBX,
        };
        decoder.decompress(jpeg, image).map_err(|e| format!("bad JPEG frame: {e}"))?;
        let rgbx = InputFormat::from_video_format(VideoFormat::RGBx).unwrap();
        self.frame.resize(w, h);
        pipeline::unpack(rgbx, &self.rgbx, w * 4, &mut self.frame);
        let (mw, mh) = model;
        self.small.resize(3 * mw * mh, 0.0);
        pipeline::resample(&self.frame.planes().concat(), w, h, &mut self.small, mw, mh, 3);
        self.input.resize(self.small.len(), half::f16::ZERO);
        half::slice::HalfFloatSliceExt::convert_from_f32_slice(self.input.as_mut_slice(), &self.small);
        Ok(true)
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
        gpu_with(Effects::None)
    }

    fn gpu_with(effects: Effects) -> Option<Processor> {
        match Processor::gpu(effects) {
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
        let mut cpu = Processor::cpu(Effects::None);
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
        let mut cpu = Processor::cpu(Effects::None);
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
        let mut cpu = Processor::cpu(Effects::None);
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
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../testimage.jpg");
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
        let mut backends = vec![("CPU", Processor::cpu(Effects::None))];
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
                let mut stages = [Duration::ZERO; 5];
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

    const GREEN: Background = Background::Color([0.0, 177.0 / 255.0, 64.0 / 255.0]);

    #[test]
    fn background_gpu_matches_cpu() {
        use crate::model::fake::FakeMatte;
        // RVM's and the selfie model's resolutions, with each background
        // (including a blur strong enough to clamp at the frame's edges).
        let cases = [
            ((480, 270), GREEN),
            ((256, 144), GREEN),
            ((480, 270), Background::Blur { radius: 2.0 }),
            ((256, 144), Background::Blur { radius: 8.0 }),
        ];
        for ((mw, mh), bg) in cases {
            let fake = || Effects::Background(Matting::new(Box::new(FakeMatte::with_size(mw, mh)), false), bg);
            let Some(gpu) = gpu_with(fake()) else { return };
            background_matches(gpu, Processor::cpu(fake()), (mw, mh), bg);
        }
    }

    #[test]
    fn background_can_change_in_place() {
        use crate::model::fake::FakeMatte;
        let fake = || Effects::Background(Matting::new(Box::new(FakeMatte::with_size(256, 144)), false), GREEN);
        let Some(mut gpu) = gpu_with(fake()) else { return };
        let mut cpu = Processor::cpu(fake());
        let blur = Background::Blur { radius: 3.0 };
        gpu.set_background(blur);
        cpu.set_background(blur);
        background_matches(gpu, cpu, (256, 144), blur);
    }

    fn background_matches(mut gpu: Processor, mut cpu: Processor, (mw, mh): (usize, usize), bg: Background) {
        // Not the model's aspect ratio, and not a multiple of the workgroup;
        // the last is big enough that the blur scales it down. Two different
        // frames at each size, as the blur weighs by the previous frame's
        // matte.
        for (w, h) in [(200, 120), (70, 37), (642, 722)] {
            let first = pattern(rgbx(), w, h, w * 4);
            let src: Vec<u8> = first.iter().map(|v| 255 - v).collect();
            let (mut a, mut b) = (vec![0u8; w * 2 * h], vec![0u8; w * 2 * h]);
            for frame in [&first, &src] {
                cpu.process(FrameFormat::Raw(rgbx()), frame, w * 4, w, h, &mut a, w * 2).unwrap();
                gpu.process(FrameFormat::Raw(rgbx()), frame, w * 4, w, h, &mut b, w * 2).unwrap();
            }
            let diffs: Vec<u8> = a.iter().zip(&b).map(|(x, y)| x.abs_diff(*y)).collect();
            let worst = *diffs.iter().max().unwrap();
            let off = diffs.iter().filter(|&&d| d > 0).count();
            eprintln!("model {mw}x{mh}, {bg:?}, frame {w}x{h}: {off} of {} bytes differ, by up to {worst}", a.len());
            // The model input is f16, which the GPU and CPU may round
            // differently in the last bit; allow that to show through.
            assert!(worst <= 2, "{w}x{h}: GPU differs from CPU by up to {worst}");
            assert!(off * 100 < a.len(), "{w}x{h}: {off} bytes differ");
        }
    }

    /// GPU time of each background at 1080p with a stand-in model (so only
    /// the GPU work is measured). Run with
    /// `cargo test --release -- --ignored --nocapture bench_backgrounds`.
    #[test]
    #[ignore]
    fn bench_backgrounds() {
        use crate::model::fake::FakeMatte;
        const FRAMES: u32 = 200;
        let (w, h) = (1920, 1080);
        let src = pattern(rgbx(), w, h, w * 4);
        let mut dst = vec![0u8; w * 2 * h];
        let cases = [("none", None), ("color", Some(GREEN)), ("blur 2%", Some(Background::Blur { radius: 2.0 })), ("blur 8%", Some(Background::Blur { radius: 8.0 }))];
        for (mw, mh) in [(480, 270), (256, 144)] {
            for (name, bg) in cases {
                let effects = match bg {
                    None => Effects::None,
                    Some(bg) => Effects::Background(Matting::new(Box::new(FakeMatte::with_size(mw, mh)), false), bg),
                };
                let Some(mut gpu) = gpu_with(effects) else { return };
                for _ in 0..20 {
                    gpu.process(FrameFormat::Raw(rgbx()), &src, w * 4, w, h, &mut dst, w * 2).unwrap();
                }
                let mut total = Duration::ZERO;
                for _ in 0..FRAMES {
                    total += gpu.process(FrameFormat::Raw(rgbx()), &src, w * 4, w, h, &mut dst, w * 2).unwrap()[2];
                }
                eprintln!("model {mw}x{mh}, {name}: gpu {:.2} ms/frame", total.as_secs_f64() * 1e3 / FRAMES as f64);
            }
        }
    }

    #[test]
    fn early_input_picks_the_smallest_scale_that_covers_the_model() {
        let scale = |size, model| EarlyInput::scale(size, model).map(|f| (f.num(), f.denom()));
        assert_eq!(scale((1920, 1080), (480, 270)), Some((1, 4)));
        assert_eq!(scale((1280, 720), (480, 270)), Some((3, 8)));
        assert_eq!(scale((640, 360), (480, 270)), Some((3, 4)));
        assert_eq!(scale((1920, 1080), (256, 144)), Some((1, 4)));
        // Nothing smaller than the frame itself covers the model.
        assert_eq!(scale((200, 120), (480, 270)), None);
    }

    #[test]
    fn mjpg_background_gpu_matches_cpu() {
        use crate::model::fake::FakeMatte;
        for bg in [GREEN, Background::Blur { radius: 2.0 }] {
            let fake = || Effects::Background(Matting::new(Box::new(FakeMatte::with_size(480, 270)), false), bg);
            let Some(mut gpu) = gpu_with(fake()) else { return };
            let mut cpu = Processor::cpu(fake());
            // Early input from a 3/4 and a 1/2 scale decode.
            for (w, h) in [(640, 360), (960, 540)] {
                let mjpg = jpeg(&smooth_rgbx(w, h), w, h, 90, turbojpeg::Subsamp::Sub2x1);
                let (mut a, mut b) = (vec![0u8; w * 2 * h], vec![0u8; w * 2 * h]);
                for _ in 0..2 {
                    cpu.process(FrameFormat::Mjpg, &mjpg, 0, w, h, &mut a, w * 2).unwrap();
                    gpu.process(FrameFormat::Mjpg, &mjpg, 0, w, h, &mut b, w * 2).unwrap();
                }
                let worst = a.iter().zip(&b).map(|(x, y)| x.abs_diff(*y)).max().unwrap();
                assert!(worst <= 2, "{bg:?} {w}x{h}: GPU differs from CPU by up to {worst}");
            }
        }
    }

    /// How much the MJPG early input (a scaled-down decode) changes the
    /// real model's result: `testimage.jpg` as MJPG (early input) against
    /// the same frame decoded and passed as raw RGBx (the model input made
    /// from the full frame), green background, on the GPU. Set
    /// `VIDEO_MENU_TEST_OUT` to save both.
    ///
    ///     cargo test --release -- --ignored --nocapture early_input_matte
    #[test]
    #[ignore]
    fn early_input_matte_matches_full_decode() {
        let jpg = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../testimage.jpg")).expect("testimage.jpg");
        let img = turbojpeg::decompress(&jpg, turbojpeg::PixelFormat::RGBX).unwrap();
        let (w, h) = (img.width, img.height);
        let choice = model::choice_from_env().unwrap();
        let effects = || Effects::Background(Matting::new(model::load(choice, 0.5).unwrap().0, false), GREEN);
        let run = |format: FrameFormat, src: &[u8], stride: usize| {
            let mut proc = Processor::gpu(effects()).unwrap();
            let mut dst = vec![0u8; w * 2 * h];
            // A still image as a short clip, so the recurrent state settles.
            for _ in 0..8 {
                proc.process(format, src, stride, w, h, &mut dst, w * 2).unwrap();
            }
            dst
        };
        let early = run(FrameFormat::Mjpg, &jpg, 0);
        let full = run(FrameFormat::Raw(rgbx()), &img.pixels, w * 4);
        if let Ok(dir) = std::env::var("VIDEO_MENU_TEST_OUT") {
            std::fs::write(format!("{dir}/early_{w}x{h}.yuv"), &early).unwrap();
            std::fs::write(format!("{dir}/full_{w}x{h}.yuv"), &full).unwrap();
        }
        let diffs: Vec<u8> = early.iter().zip(&full).map(|(a, b)| a.abs_diff(*b)).collect();
        let over = |t: u8| diffs.iter().filter(|&&d| d > t).count() as f64 * 100.0 / diffs.len() as f64;
        let mean = diffs.iter().map(|&d| d as f64).sum::<f64>() / diffs.len() as f64;
        eprintln!(
            "{choice:?}, {w}x{h}: mean diff {mean:.3}, >8: {:.3}%, >32: {:.3}%, >64: {:.3}%, max {}",
            over(8),
            over(32),
            over(64),
            diffs.iter().max().unwrap()
        );
    }

    /// Run the real model (chosen as in the daemon, with
    /// `VIDEO_MENU_MATTE_MODEL`) on `testimage.jpg` on both backends. Needs
    /// the model files and the NPU (or `VIDEO_MENU_MODEL_DEVICE`). Set
    /// `VIDEO_MENU_TEST_OUT` to a directory to save the YUY2 results.
    ///
    ///     cargo test --release -- --ignored --nocapture real_model
    #[test]
    #[ignore]
    fn real_model_on_testimage() {
        let jpg = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../testimage.jpg"))
            .expect("testimage.jpg");
        let header = turbojpeg::read_header(&jpg).unwrap();
        let (w, h) = (header.width, header.height);
        let choice = model::choice_from_env().unwrap();
        for (label, bg) in [("color", GREEN), ("blur", Background::Blur { radius: 2.0 })] {
            real_model_with(choice, label, bg, &jpg, (w, h));
        }
    }

    fn real_model_with(choice: ModelChoice, label: &str, bg: Background, jpg: &[u8], (w, h): (usize, usize)) {
        const FRAMES: usize = 8;
        let matte = || Effects::Background(Matting::new(model::load(choice, 0.5).unwrap().0, false), bg);
        let mut outputs = Vec::new();
        for (name, mut proc) in [("cpu", Processor::cpu(matte())), ("gpu", Processor::gpu(matte()).unwrap())] {
            let mut dst = vec![0u8; w * 2 * h];
            // A still image as a short clip, so the recurrent state settles.
            let mut last = [Duration::ZERO; 5];
            for _ in 0..FRAMES {
                last = proc.process(FrameFormat::Mjpg, jpg, 0, w, h, &mut dst, w * 2).unwrap();
            }
            let stages: Vec<String> = proc
                .stage_names()
                .iter()
                .zip(last)
                .map(|(n, t)| format!("{n} {:.2}", t.as_secs_f64() * 1e3))
                .collect();
            eprintln!("{label} {name}: last frame {}", stages.join(", "));
            if let Ok(dir) = std::env::var("VIDEO_MENU_TEST_OUT") {
                std::fs::write(format!("{dir}/{label}_{name}_{w}x{h}.yuv"), &dst).unwrap();
            }
            outputs.push(dst);
        }
        let diffs: Vec<u8> = outputs[0].iter().zip(&outputs[1]).map(|(a, b)| a.abs_diff(*b)).collect();
        let big = diffs.iter().filter(|&&d| d > 8).count();
        eprintln!(
            "{label} cpu vs gpu: max diff {}, {} of {} bytes differ by more than 8",
            diffs.iter().max().unwrap(),
            big,
            diffs.len()
        );
        // f16 rounding differences in the model input can nudge the matte,
        // but the results must agree almost everywhere.
        assert!(big * 1000 < diffs.len(), "CPU and GPU results disagree");
    }
}
