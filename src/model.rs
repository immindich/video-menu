//! Matting models, run with OpenVINO (by default on the NPU).
//!
//! A model sees a small copy of the frame, at a resolution it chooses
//! ([`Matte::size`]), and returns per-pixel coefficients `A` and `b` (4
//! channels each: foreground residual R, G, B and alpha) at that resolution.
//! The caller upsamples them to the frame's resolution and computes
//! `A * [R, G, B, gray] + b` there; see [`crate::pipeline::GreenScreen`] and
//! `green_screen` in `pipeline.wgsl`. The model is [`RvmMatte`]:
//! RobustVideoMatting without its full-resolution refiner stage (see
//! `tools/split_refiner.py`). The network itself produces `A` and `b` (a
//! learned guided filter) and carries recurrent state between frames.
//!
//! [`Matting`] schedules the model. It can either run it on each frame
//! before that frame is composited, or reuse the previous frame's result
//! while the model works on the current frame in the background, which takes
//! the model off the path from camera to output.

use std::time::{Duration, Instant};

use half::f16;
use openvino::{
    Core, DeviceType, ElementType, InferRequest, PartialShape, RwPropertyKey, Shape, Tensor,
};

/// Resolution RVM runs at.
pub const RVM_SIZE: (usize, usize) = (480, 270);

/// File names looked for in the current directory when the model's path
/// variable is unset or wrong.
const RVM_FILE: &str = "rvm_mobilenetv3_fp16_core.onnx";

/// Recurrent state tensors: name prefix, channels, and downscale factor
/// relative to the model resolution (stride-2 convolutions round up).
const RECURRENT: [(&str, i64, i64); 4] = [("r1", 16, 2), ("r2", 20, 4), ("r3", 40, 8), ("r4", 64, 16)];

/// Something that turns a downsampled frame into refiner coefficients,
/// possibly in the background.
pub trait Matte {
    /// The resolution (width, height) the model works at.
    fn size(&self) -> (usize, usize);

    /// Start working on `src`: the frame at [`size`](Self::size) as three
    /// planes (R, G, B) of values in 0..1. At most one frame is in flight;
    /// call [`finish`](Self::finish) before starting another.
    fn start(&mut self, src: &[f16]) -> Result<(), String>;

    /// Wait for the started frame and make its result the current
    /// [`coefficients`](Self::coefficients).
    fn finish(&mut self) -> Result<(), String>;

    /// The latest finished result: `A` and `b`, each four planes (fgr
    /// residual R, G, B, then alpha) at the model's resolution. Unaffected by
    /// [`start`](Self::start), so it can be used while a frame is in flight.
    fn coefficients(&self) -> (&[f16], &[f16]);

    /// Forget temporal state, e.g. after switching cameras. Must not be
    /// called while a frame is in flight.
    fn reset(&mut self);
}

/// Runs a [`Matte`] for each frame, either in step with the frames or one
/// frame behind.
pub struct Matting {
    matte: Box<dyn Matte>,
    /// Composite each frame with the previous frame's result, so the model
    /// runs alongside the rest of the pipeline instead of before it.
    reuse: bool,
    /// A frame is in flight (reuse mode only).
    pending: bool,
    /// The matte has a result to reuse.
    ready: bool,
    last: Duration,
}

impl Matting {
    pub fn new(matte: Box<dyn Matte>, reuse: bool) -> Self {
        Self { matte, reuse, pending: false, ready: false, last: Duration::ZERO }
    }

    /// The resolution the model works at.
    pub fn size(&self) -> (usize, usize) {
        self.matte.size()
    }

    /// The coefficients to composite the frame whose model input is `src`
    /// with: that frame's own, or in reuse mode the previous frame's (the
    /// model then starts on `src` in the background). The first frame after
    /// a reset always waits for its own result.
    pub fn run(&mut self, src: &[f16]) -> Result<(&[f16], &[f16]), String> {
        let start = Instant::now();
        if self.pending {
            self.pending = false;
            self.matte.finish()?;
        } else if !self.reuse || !self.ready {
            self.matte.start(src)?;
            self.matte.finish()?;
        }
        if self.reuse {
            // The first frame was just run synchronously; running it again
            // here only gives the recurrent state one more look at it.
            self.matte.start(src)?;
            self.pending = true;
        }
        self.ready = true;
        self.last = start.elapsed();
        Ok(self.matte.coefficients())
    }

    /// Forget temporal state, e.g. after switching cameras or frame sizes.
    pub fn reset(&mut self) {
        if std::mem::take(&mut self.pending) {
            let _ = self.matte.finish();
        }
        self.matte.reset();
        self.ready = false;
    }

    /// How long the last [`run`](Self::run) kept the caller waiting.
    pub fn last_time(&self) -> Duration {
        self.last
    }
}

impl Drop for Matting {
    fn drop(&mut self) {
        // Don't free the model's buffers while it may still be using them.
        if self.pending {
            let _ = self.matte.finish();
        }
    }
}

/// Which model to use, from `VIDEO_MENU_MATTE_MODEL` (currently only `rvm`,
/// the default). Returns the model and a description for the log.
pub fn from_env() -> Result<(Box<dyn Matte>, String), String> {
    let device = std::env::var("VIDEO_MENU_MODEL_DEVICE").unwrap_or_else(|_| "NPU".into());
    match std::env::var("VIDEO_MENU_MATTE_MODEL").as_deref() {
        Ok("rvm") | Ok("") | Err(_) => {
            let path = find_model("VIDEO_MENU_MODEL", RVM_FILE, "nix build .#rvm-model")?;
            let matte = RvmMatte::load(&path, &device)?;
            Ok((Box::new(matte), format!("RVM on the {device}")))
        }
        Ok(other) => Err(format!("VIDEO_MENU_MATTE_MODEL must be rvm, not {other:?}")),
    }
}

/// RVM's network on OpenVINO.
pub struct RvmMatte {
    request: InferRequest,
    src: Tensor,
    /// Recurrent inputs, fed from the previous frame's outputs.
    state: Vec<Tensor>,
    a: Vec<f16>,
    b: Vec<f16>,
}

impl RvmMatte {
    pub fn load(path: &str, device: &str) -> Result<Self, String> {
        let (w, h) = (RVM_SIZE.0 as i64, RVM_SIZE.1 as i64);
        let mut shapes = vec![("src_sm".to_string(), vec![1, 3, h, w])];
        for (name, channels, f) in RECURRENT {
            shapes.push((format!("{name}i"), vec![1, channels, (h + f - 1) / f, (w + f - 1) / f]));
        }
        let request = compile(path, device, &shapes)?;
        let src = zero_tensor(ElementType::F16, &shapes[0].1)?;
        let state = shapes[1..]
            .iter()
            .map(|(_, s)| zero_tensor(ElementType::F16, s))
            .collect::<Result<_, _>>()?;
        let pixels = RVM_SIZE.0 * RVM_SIZE.1;
        Ok(Self {
            request,
            src,
            state,
            a: vec![f16::ZERO; 4 * pixels],
            b: vec![f16::ZERO; 4 * pixels],
        })
    }

    fn start_inner(&mut self, src: &[f16]) -> Result<(), openvino::InferenceError> {
        self.src
            .get_raw_data_mut()?
            .copy_from_slice(bytemuck::cast_slice(src));
        self.request.set_tensor("src_sm", &self.src)?;
        for ((name, _, _), state) in RECURRENT.iter().zip(&self.state) {
            self.request.set_tensor(&format!("{name}i"), state)?;
        }
        self.request.infer_async()
    }

    fn finish_inner(&mut self) -> Result<(), openvino::InferenceError> {
        wait(&mut self.request)?;
        let copy_out = |name: &str, dst: &mut [f16]| -> Result<(), openvino::InferenceError> {
            let t = self.request.get_tensor(name)?;
            bytemuck::cast_slice_mut(dst).copy_from_slice(t.get_raw_data()?);
            Ok(())
        };
        copy_out("A", &mut self.a)?;
        copy_out("b", &mut self.b)?;
        // Outputs become the next frame's recurrent inputs. Copy them: the
        // request reuses its output buffers on the next inference.
        for ((name, _, _), state) in RECURRENT.iter().zip(&mut self.state) {
            let out = self.request.get_tensor(&format!("{name}o"))?;
            state.get_raw_data_mut()?.copy_from_slice(out.get_raw_data()?);
        }
        Ok(())
    }
}

impl Matte for RvmMatte {
    fn size(&self) -> (usize, usize) {
        RVM_SIZE
    }

    fn start(&mut self, src: &[f16]) -> Result<(), String> {
        check_input(src, RVM_SIZE)?;
        self.start_inner(src).map_err(|e| format!("starting inference: {e}"))
    }

    fn finish(&mut self) -> Result<(), String> {
        self.finish_inner().map_err(|e| format!("inference failed: {e}"))
    }

    fn coefficients(&self) -> (&[f16], &[f16]) {
        (&self.a, &self.b)
    }

    fn reset(&mut self) {
        for state in &mut self.state {
            if let Ok(data) = state.get_raw_data_mut() {
                data.fill(0);
            }
        }
    }
}

fn check_input(src: &[f16], (w, h): (usize, usize)) -> Result<(), String> {
    if src.len() != 3 * w * h {
        return Err(format!("model input has {} values, expected {}", src.len(), 3 * w * h));
    }
    Ok(())
}

/// Wait for an asynchronous inference. The C API doesn't report a timeout;
/// this one only guards against a hung device.
fn wait(request: &mut InferRequest) -> Result<(), openvino::InferenceError> {
    request.wait(10_000)
}

fn zero_tensor(ty: ElementType, dims: &[i64]) -> Result<Tensor, String> {
    let shape = Shape::new(dims).map_err(ov_err("shape"))?;
    let mut t = Tensor::new(ty, &shape).map_err(ov_err("tensor"))?;
    t.get_raw_data_mut().map_err(ov_err("tensor"))?.fill(0);
    Ok(t)
}

/// Read the model at `path`, fix its input shapes, and compile it for
/// `device`, returning an inference request.
fn compile(path: &str, device: &str, shapes: &[(String, Vec<i64>)]) -> Result<InferRequest, String> {
    if !std::path::Path::new(path).exists() {
        return Err(format!("model {path} not found"));
    }
    let mut core = Core::new().map_err(|e| format!("OpenVINO: {e}"))?;
    let dev: DeviceType = device.parse().map_err(|e| format!("device {device}: {e}"))?;
    let mut turbo = false;
    if dev == DeviceType::NPU {
        // nixpkgs' openvino doesn't ship the plugin compiler
        // (libopenvino_intel_npu_compiler.so); use the NPU driver's.
        core.set_property(&dev, &RwPropertyKey::Other("NPU_COMPILER_TYPE".into()), "DRIVER")
            .map_err(ov_err("NPU compiler"))?;
        // Between camera frames the NPU idles and clocks down, so an
        // inference paced at 30 fps takes ~15 ms against ~10.5 ms back to
        // back (for RVM). Turbo avoids that, but the driver then busy-waits
        // for the NPU: inference costs ~12 ms of CPU instead of ~2 ms.
        // Measured with RVM at 1080p:
        // - NexiGo (MJPG): the model drops from ~17.6 to ~11.3 ms, but MJPG
        //   decoding slows by ~2 ms (the spinning core competes with it), so
        //   frames get ~3.6 ms (~9%) faster for ~1/3 of a core.
        // - Built-in camera (~50 fps): the NPU never idles long enough to
        //   slow down, so turbo gains nothing and still costs ~45% of a core.
        // Opt in with VIDEO_MENU_NPU_TURBO=1.
        turbo = std::env::var("VIDEO_MENU_NPU_TURBO").as_deref() == Ok("1");
        core.set_property(
            &dev,
            &RwPropertyKey::Other("NPU_TURBO".into()),
            if turbo { "YES" } else { "NO" },
        )
        .map_err(ov_err("NPU turbo"))?;
    }
    // Compiling for the NPU takes ~5-10 s; cache the result between runs.
    if let Some(cache) = cache_dir()
        && let Err(e) = core.set_property(&dev, &RwPropertyKey::CacheDir, &cache)
    {
        eprintln!("model: not caching compiled model: {e}");
    }

    let mut model = core.read_model_from_file(path, "").map_err(ov_err("reading model"))?;
    let partial: Vec<(String, PartialShape)> = shapes
        .iter()
        .map(|(n, s)| Ok((n.clone(), PartialShape::new_static(s.len() as i64, s)?)))
        .collect::<Result<_, openvino::InferenceError>>()
        .map_err(ov_err("model shapes"))?;
    let partial: Vec<(&str, &PartialShape)> = partial.iter().map(|(n, s)| (n.as_str(), s)).collect();
    model.reshape(&partial).map_err(ov_err("reshaping model"))?;

    let start = Instant::now();
    let mut compiled = core.compile_model(&model, dev).map_err(ov_err("compiling model"))?;
    let mode = if turbo { " (turbo)" } else { "" };
    eprintln!("model: compiled for {device}{mode} in {:.1}s", start.elapsed().as_secs_f64());
    compiled.create_infer_request().map_err(ov_err("infer request"))
}

/// Where to load a model from: the path in `var` if it is set and exists,
/// otherwise `file` in the current directory.
fn find_model(var: &str, file: &str, build_hint: &str) -> Result<String, String> {
    if let Ok(path) = std::env::var(var) {
        if std::path::Path::new(&path).exists() {
            return Ok(path);
        }
        eprintln!("model: {var}={path} does not exist; trying ./{file}");
    }
    if std::path::Path::new(file).exists() {
        return Ok(file.into());
    }
    Err(format!(
        "no matting model: set {var} or put {file} in the current directory \
         (build it with `{build_hint}`)"
    ))
}

/// Format an OpenVINO error with what we were doing.
fn ov_err(what: &'static str) -> impl Fn(openvino::InferenceError) -> String {
    move |e| format!("{what}: {e}")
}

fn cache_dir() -> Option<String> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?;
    let dir = base.join("video-menu").join("openvino");
    std::fs::create_dir_all(&dir).ok()?;
    dir.to_str().map(str::to_owned)
}

#[cfg(test)]
pub mod fake {
    //! A stand-in model for testing the code around inference without an NPU.

    use super::*;

    /// Produces coefficients that depend on every input channel, so tests
    /// notice if the frame is resampled or laid out differently.
    pub struct FakeMatte {
        size: (usize, usize),
        a: Vec<f16>,
        b: Vec<f16>,
        /// The started frame's result, published by `finish`.
        next: Option<(Vec<f16>, Vec<f16>)>,
        pub frames: usize,
    }

    impl FakeMatte {
        pub fn with_size(w: usize, h: usize) -> Self {
            Self { size: (w, h), a: Vec::new(), b: Vec::new(), next: None, frames: 0 }
        }
    }

    impl Default for FakeMatte {
        fn default() -> Self {
            Self::with_size(RVM_SIZE.0, RVM_SIZE.1)
        }
    }

    /// The fake's coefficients for one frame of `n` pixels.
    pub fn coefficients_for(src: &[f16], n: usize) -> (Vec<f16>, Vec<f16>) {
        assert_eq!(src.len(), 3 * n);
        let (mut a, mut b) = (vec![f16::ZERO; 4 * n], vec![f16::ZERO; 4 * n]);
        for i in 0..n {
            let rgb = [src[i].to_f32(), src[n + i].to_f32(), src[2 * n + i].to_f32()];
            for c in 0..4 {
                // Residuals in -0.2..0.2; alpha (c = 3) spanning 0..1.
                let (ac, bc) = match c {
                    3 => (2.0 * rgb[0] - rgb[2], 0.5 * rgb[1] - 0.3),
                    _ => (0.2 * rgb[(c + 1) % 3] - 0.1, 0.1 - 0.2 * rgb[c]),
                };
                a[c * n + i] = f16::from_f32(ac);
                b[c * n + i] = f16::from_f32(bc);
            }
        }
        (a, b)
    }

    impl Matte for FakeMatte {
        fn size(&self) -> (usize, usize) {
            self.size
        }

        fn start(&mut self, src: &[f16]) -> Result<(), String> {
            assert!(self.next.is_none(), "started a frame while one was in flight");
            self.next = Some(coefficients_for(src, self.size.0 * self.size.1));
            self.frames += 1;
            Ok(())
        }

        fn finish(&mut self) -> Result<(), String> {
            (self.a, self.b) = self.next.take().expect("finished without starting");
            Ok(())
        }

        fn coefficients(&self) -> (&[f16], &[f16]) {
            (&self.a, &self.b)
        }

        fn reset(&mut self) {
            assert!(self.next.is_none(), "reset while a frame was in flight");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fake::{FakeMatte, coefficients_for};
    use super::*;

    const N: usize = RVM_SIZE.0 * RVM_SIZE.1;

    fn frame(v: f32) -> Vec<f16> {
        vec![f16::from_f32(v); 3 * N]
    }

    fn a_of(m: &mut Matting, src: &[f16]) -> Vec<f16> {
        m.run(src).unwrap().0.to_vec()
    }

    #[test]
    fn in_step_uses_each_frames_own_result() {
        let mut m = Matting::new(Box::<FakeMatte>::default(), false);
        for v in [0.1, 0.5, 0.9] {
            assert_eq!(a_of(&mut m, &frame(v)), coefficients_for(&frame(v), N).0);
        }
    }

    #[test]
    fn reuse_lags_one_frame_after_the_first() {
        let mut m = Matting::new(Box::<FakeMatte>::default(), true);
        let (f1, f2, f3) = (frame(0.1), frame(0.5), frame(0.9));
        // The first frame has nothing to reuse, so it waits for its own.
        assert_eq!(a_of(&mut m, &f1), coefficients_for(&f1, N).0);
        // Later frames get the previous frame's result.
        assert_eq!(a_of(&mut m, &f2), coefficients_for(&f1, N).0);
        assert_eq!(a_of(&mut m, &f3), coefficients_for(&f2, N).0);
        // A reset drops the in-flight frame and starts over.
        m.reset();
        assert_eq!(a_of(&mut m, &f1), coefficients_for(&f1, N).0);
    }
}

