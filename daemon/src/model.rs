//! Matting models, run with OpenVINO (by default on the NPU).
//!
//! A model sees a small copy of the frame, at a resolution it chooses
//! ([`Matte::size`]), and returns per-pixel coefficients `A` and `b` (4
//! channels each: foreground residual R, G, B and alpha) at that resolution.
//! The caller upsamples them to the frame's resolution and computes
//! `A * [R, G, B, gray] + b` there; see [`crate::pipeline::ReplaceBackground`]
//! and `matte_at` in `pipeline.wgsl`. Two models are available:
//!
//! - [`RvmMatte`]: RobustVideoMatting without its full-resolution refiner
//!   stage (see `tools/split_refiner.py`). The network itself produces `A`
//!   and `b` (a learned guided filter) and carries recurrent state between
//!   frames. GPL-3.0.
//! - [`SelfieMatte`]: MediaPipe's selfie segmenter (landscape), converted
//!   with `tools/convert_selfie.py`. It outputs only a 256x144 person mask; a
//!   classic guided filter, guided by the frame's brightness, turns that into
//!   `A` and `b`, and the foreground colour is taken to be the frame's own.
//!   It is single-frame, so its mask is smoothed over time where the image
//!   is still ([`TemporalSmoother`]). Very small (~250K weights);
//!   Apache-2.0.
//!
//! [`Matting`] schedules the model. It can either run it on each frame
//! before that frame is composited, or reuse the previous frame's result
//! while the model works on the current frame in the background, which takes
//! the model off the path from camera to output.

#[cfg(test)]
use std::time::Duration;
use std::time::Instant;

use half::{f16, slice::HalfFloatSliceExt};
use video_menu_protocol::ModelChoice;
use openvino::{
    Core, DeviceType, ElementType, InferRequest, PartialShape, RwPropertyKey, Shape, Tensor,
};

/// Resolution RVM runs at.
pub const RVM_SIZE: (usize, usize) = (480, 270);

/// Resolution of MediaPipe's landscape selfie segmenter.
pub const SELFIE_SIZE: (usize, usize) = (256, 144);

/// File names looked for in the current directory when the model's path
/// variable is unset or wrong.
const RVM_FILE: &str = "rvm_mobilenetv3_fp16_core.onnx";
const SELFIE_FILE: &str = "selfie_segmenter_landscape.xml";

/// Guided filter applied to alpha-only models' masks: window radius (at
/// model resolution) and regularisation. Smaller `eps` follows the frame's
/// edges more closely, larger smooths more.
const GUIDED_RADIUS: usize = 2;
const GUIDED_EPS: f32 = 1e-3;

/// Default temporal smoothing for the selfie model's mask: the weight given
/// to the previous frame's mask where the image hasn't changed.
pub const SELFIE_SMOOTHING: f32 = 0.5;

/// Brightness change (0..1, averaged over a small window) at which temporal
/// smoothing fades out completely, so moving parts follow the new mask.
const MOTION_THRESHOLD: f32 = 0.08;

/// Recurrent state tensors: name prefix, channels, and downscale factor
/// relative to the model resolution (stride-2 convolutions round up).
const RECURRENT: [(&str, i64, i64); 4] = [("r1", 16, 2), ("r2", 20, 4), ("r3", 40, 8), ("r4", 64, 16)];

/// Something that turns a downsampled frame into refiner coefficients,
/// possibly in the background.
/// `Send`, so the model can run on a thread of its own.
pub trait Matte: Send {
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

    /// Change how strongly the result is smoothed over time, for models that
    /// do that (0 disables it). Others ignore it.
    fn set_smoothing(&mut self, _strength: f32) {}
}

/// Runs a [`Matte`] for each frame, either in step with the frames or one
/// frame behind: the serial counterpart of the worker's model thread, used
/// by the serial [`crate::processor::Processor`].
#[cfg(test)]
pub struct Matting {
    matte: Box<dyn Matte>,
    /// Composite each frame with the previous frame's result, so the model
    /// runs alongside the rest of the pipeline instead of before it.
    reuse: bool,
    /// A frame is in flight whose result the next frame will use (reuse
    /// mode only).
    pending: bool,
    /// [`begin`](Self::begin) started this frame's own result, which
    /// [`end`](Self::end) waits for.
    started: bool,
    /// In reuse mode, the first frame's input, which the model is started on
    /// again once it has that frame's own result.
    first: Vec<f16>,
    /// The matte has a result to reuse.
    ready: bool,
    last: Duration,
}

#[cfg(test)]
impl Matting {
    pub fn new(matte: Box<dyn Matte>, reuse: bool) -> Self {
        Self { matte, reuse, pending: false, started: false, first: Vec::new(), ready: false, last: Duration::ZERO }
    }

    /// The resolution the model works at.
    pub fn size(&self) -> (usize, usize) {
        self.matte.size()
    }

    /// Switch between waiting for each frame's own result and reusing the
    /// previous frame's. Turning reuse off waits for any frame in flight and
    /// starts over, so the next frame gets its own result.
    pub fn set_reuse(&mut self, reuse: bool) {
        if reuse == self.reuse {
            return;
        }
        self.settle();
        self.ready = false;
        self.reuse = reuse;
    }


    /// The coefficients to composite the frame whose model input is `src`
    /// with: that frame's own, or in reuse mode the previous frame's (the
    /// model then starts on `src` in the background). The first frame after
    /// a reset always waits for its own result.
    #[cfg(test)]
    pub fn run(&mut self, src: &[f16]) -> Result<(&[f16], &[f16]), String> {
        self.begin(src)?;
        self.end()
    }

    /// The first half of [`run`](Self::run): start the model on `src`, so
    /// the caller can do other work while it runs, then call
    /// [`end`](Self::end) for the coefficients.
    pub fn begin(&mut self, src: &[f16]) -> Result<(), String> {
        let start = Instant::now();
        if self.reuse && self.ready {
            // Composite with the result in flight; start on this frame.
            if std::mem::take(&mut self.pending) {
                self.matte.finish()?;
            }
            self.matte.start(src)?;
            self.pending = true;
        } else {
            self.matte.start(src)?;
            self.started = true;
            if self.reuse {
                self.first.clear();
                self.first.extend_from_slice(src);
            }
        }
        self.last = start.elapsed();
        Ok(())
    }

    /// Wait for the frame [`begin`](Self::begin) started, if this frame
    /// needs its own result, and return the coefficients to composite with.
    pub fn end(&mut self) -> Result<(&[f16], &[f16]), String> {
        let start = Instant::now();
        if std::mem::take(&mut self.started) {
            self.matte.finish()?;
            if self.reuse {
                // The first frame after a reset waited for its own result;
                // keep the model busy with it, so the next frame has one
                // in flight to reuse (the recurrent state gets one more look
                // at this frame).
                self.matte.start(&self.first)?;
                self.pending = true;
            }
        }
        self.ready = true;
        self.last += start.elapsed();
        Ok(self.matte.coefficients())
    }

    /// Wait for any frame in flight, so the model is idle.
    fn settle(&mut self) {
        if std::mem::take(&mut self.pending) | std::mem::take(&mut self.started) {
            let _ = self.matte.finish();
        }
    }

    /// Forget temporal state, e.g. after switching cameras or frame sizes.
    pub fn reset(&mut self) {
        self.settle();
        self.matte.reset();
        self.ready = false;
    }

    /// How long the last [`run`](Self::run) kept the caller waiting.
    pub fn last_time(&self) -> Duration {
        self.last
    }
}

#[cfg(test)]
impl Drop for Matting {
    fn drop(&mut self) {
        // Don't free the model's buffers while it may still be using them.
        self.settle();
    }
}

/// The model to start with, from `VIDEO_MENU_MATTE_MODEL` (`rvm`, the
/// default, or `selfie`).
pub fn choice_from_env() -> Result<ModelChoice, String> {
    match std::env::var("VIDEO_MENU_MATTE_MODEL") {
        Ok(s) if !s.is_empty() => s.parse(),
        _ => Ok(ModelChoice::default()),
    }
}

/// The selfie model's starting temporal smoothing, from
/// `VIDEO_MENU_SELFIE_SMOOTHING` (0..1, default [`SELFIE_SMOOTHING`]).
pub fn smoothing_from_env() -> Result<f32, String> {
    match std::env::var("VIDEO_MENU_SELFIE_SMOOTHING") {
        Ok(s) => s
            .parse::<f32>()
            .ok()
            .filter(|v| (0.0..1.0).contains(v))
            .ok_or_else(|| format!("VIDEO_MENU_SELFIE_SMOOTHING must be in 0..1, not {s:?}")),
        Err(_) => Ok(SELFIE_SMOOTHING),
    }
}

/// Load `choice` from its usual location (see [`find_model`]), on the device
/// named by `VIDEO_MENU_MODEL_DEVICE` (default `NPU`). `smoothing` applies to
/// the selfie model. Returns the model and a description for the log.
pub fn load(choice: ModelChoice, smoothing: f32) -> Result<(Box<dyn Matte>, String), String> {
    let device = std::env::var("VIDEO_MENU_MODEL_DEVICE").unwrap_or_else(|_| "NPU".into());
    match choice {
        ModelChoice::Rvm => {
            let path = find_model("VIDEO_MENU_MODEL", RVM_FILE, "nix build .#rvm-model")?;
            let matte = RvmMatte::load(&path, &device)?;
            Ok((Box::new(matte), format!("RVM on the {device}")))
        }
        ModelChoice::Selfie => {
            let path = find_model("VIDEO_MENU_SELFIE_MODEL", SELFIE_FILE, "nix build .#selfie-model")?;
            let matte = SelfieMatte::load(&path, &device, smoothing)?;
            Ok((
                Box::new(matte),
                format!("MediaPipe selfie (landscape) on the {device}, smoothing {smoothing}"),
            ))
        }
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

/// MediaPipe's selfie segmenter on OpenVINO, with a guided filter to make
/// refiner coefficients from its mask. The converted model takes the frame
/// as f16 planes directly (see `tools/convert_selfie.py`).
pub struct SelfieMatte {
    request: InferRequest,
    input: Tensor,
    /// Brightness of the frame in flight, which guides the filter.
    guide: Vec<f32>,
    alpha: Vec<f32>,
    smoother: TemporalSmoother,
    filter: GuidedFilter,
    a: Vec<f16>,
    b: Vec<f16>,
}

impl SelfieMatte {
    /// `smoothing` is the [`TemporalSmoother`] strength (0 disables it).
    pub fn load(path: &str, device: &str, smoothing: f32) -> Result<Self, String> {
        let (w, h) = SELFIE_SIZE;
        let shape = vec![1, 3, h as i64, w as i64];
        let request = compile(path, device, &[("input_1".into(), shape.clone())])?;
        let input = zero_tensor(ElementType::F16, &shape)?;
        let pixels = w * h;
        Ok(Self {
            request,
            input,
            guide: vec![0.0; pixels],
            alpha: vec![0.0; pixels],
            smoother: TemporalSmoother::new(w, h, smoothing, MOTION_THRESHOLD),
            filter: GuidedFilter::new(w, h, GUIDED_RADIUS, GUIDED_EPS),
            a: vec![f16::ZERO; 4 * pixels],
            b: vec![f16::ZERO; 4 * pixels],
        })
    }

    fn start_inner(&mut self, src: &[f16]) -> Result<(), openvino::InferenceError> {
        let n = SELFIE_SIZE.0 * SELFIE_SIZE.1;
        self.input
            .get_raw_data_mut()?
            .copy_from_slice(bytemuck::cast_slice(src));
        for (i, g) in self.guide.iter_mut().enumerate() {
            *g = (src[i].to_f32() + src[n + i].to_f32() + src[2 * n + i].to_f32()) * (1.0 / 3.0);
        }
        self.request.set_tensor("input_1", &self.input)?;
        self.request.infer_async()
    }

    fn finish_inner(&mut self) -> Result<(), openvino::InferenceError> {
        wait(&mut self.request)?;
        // [1, 144, 256, 1]: one channel, so the same layout as a plane.
        let out = self.request.get_output_tensor_by_index(0)?;
        self.alpha.copy_from_slice(bytemuck::cast_slice(out.get_raw_data()?));
        self.smoother.apply(&self.guide, &mut self.alpha);
        self.filter.apply(&self.guide, &self.alpha);
        // The foreground colour is the frame's own (zero residual), and alpha
        // follows the guide: alpha = A * gray + b, which the refine step
        // evaluates per pixel at full resolution.
        let n = SELFIE_SIZE.0 * SELFIE_SIZE.1;
        self.a[..3 * n].fill(f16::ZERO);
        self.b[..3 * n].fill(f16::ZERO);
        self.a[3 * n..].convert_from_f32_slice(&self.filter.a);
        self.b[3 * n..].convert_from_f32_slice(&self.filter.b);
        Ok(())
    }
}

impl Matte for SelfieMatte {
    fn size(&self) -> (usize, usize) {
        SELFIE_SIZE
    }

    fn start(&mut self, src: &[f16]) -> Result<(), String> {
        check_input(src, SELFIE_SIZE)?;
        self.start_inner(src).map_err(|e| format!("starting inference: {e}"))
    }

    fn finish(&mut self) -> Result<(), String> {
        self.finish_inner().map_err(|e| format!("inference failed: {e}"))
    }

    fn coefficients(&self) -> (&[f16], &[f16]) {
        (&self.a, &self.b)
    }

    fn reset(&mut self) {
        self.smoother.reset();
    }

    fn set_smoothing(&mut self, strength: f32) {
        self.smoother.set_strength(strength);
    }
}

/// Smooths a single-frame model's mask over time: each new mask is blended
/// with the previous (smoothed) one, `mask = (1 - k) * new + k * previous`.
/// `k` is `strength` where the image is still and falls to 0 where its
/// brightness changed by `threshold` or more (averaged over a small window,
/// so camera noise doesn't count as motion), so moving parts follow the new
/// mask at once and only still areas are stabilised.
pub struct TemporalSmoother {
    w: usize,
    h: usize,
    strength: f32,
    threshold: f32,
    prev_mask: Vec<f32>,
    prev_guide: Vec<f32>,
    has_prev: bool,
    diff: Vec<f32>,
    motion: Vec<f32>,
    rows: Vec<f32>,
}

impl TemporalSmoother {
    pub fn new(w: usize, h: usize, strength: f32, threshold: f32) -> Self {
        let buf = || vec![0f32; w * h];
        Self {
            w,
            h,
            strength,
            threshold,
            prev_mask: buf(),
            prev_guide: buf(),
            has_prev: false,
            diff: buf(),
            motion: buf(),
            rows: buf(),
        }
    }

    /// Smooth `mask` in place, given the frame's brightness `guide`.
    pub fn apply(&mut self, guide: &[f32], mask: &mut [f32]) {
        if self.strength > 0.0 && self.has_prev {
            for ((d, g), p) in self.diff.iter_mut().zip(guide).zip(&self.prev_guide) {
                *d = (g - p).abs();
            }
            box_mean(&self.diff, self.w, self.h, 2, &mut self.rows, &mut self.motion);
            let still_scale = 1.0 / self.threshold;
            for ((m, p), motion) in mask.iter_mut().zip(&self.prev_mask).zip(&self.motion) {
                let k = self.strength * (1.0 - motion * still_scale).max(0.0);
                *m += k * (p - *m);
            }
        }
        self.prev_mask.copy_from_slice(mask);
        self.prev_guide.copy_from_slice(guide);
        self.has_prev = true;
    }

    /// Forget the previous frame, e.g. after switching cameras.
    pub fn reset(&mut self) {
        self.has_prev = false;
    }

    /// Change the weight given to the previous mask (0 disables smoothing).
    pub fn set_strength(&mut self, strength: f32) {
        self.strength = strength;
    }
}

/// Guided filter (He et al.): smoothed linear coefficients `(a, b)` such
/// that `a * guide + b` approximates `p` while following the guide's edges.
/// Computing them at low resolution and applying them to a full-resolution
/// guide is the "fast guided filter", which is exactly what the refine step
/// does. Holds its scratch buffers so per-frame use doesn't allocate.
pub struct GuidedFilter {
    w: usize,
    h: usize,
    r: usize,
    eps: f32,
    mean_i: Vec<f32>,
    mean_p: Vec<f32>,
    mean_ip: Vec<f32>,
    mean_ii: Vec<f32>,
    tmp: Vec<f32>,
    rows: Vec<f32>,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

impl GuidedFilter {
    pub fn new(w: usize, h: usize, r: usize, eps: f32) -> Self {
        let n = w * h;
        let buf = || vec![0f32; n];
        Self {
            w,
            h,
            r,
            eps,
            mean_i: buf(),
            mean_p: buf(),
            mean_ip: buf(),
            mean_ii: buf(),
            tmp: buf(),
            rows: buf(),
            a: buf(),
            b: buf(),
        }
    }

    /// Filter `p` guided by `guide` (both `w`x`h`); the result is in
    /// [`a`](Self::a) and [`b`](Self::b).
    pub fn apply(&mut self, guide: &[f32], p: &[f32]) {
        let (w, h, r) = (self.w, self.h, self.r);
        box_mean(guide, w, h, r, &mut self.rows, &mut self.mean_i);
        box_mean(p, w, h, r, &mut self.rows, &mut self.mean_p);
        for ((t, g), v) in self.tmp.iter_mut().zip(guide).zip(p) {
            *t = g * v;
        }
        box_mean(&self.tmp, w, h, r, &mut self.rows, &mut self.mean_ip);
        for (t, g) in self.tmp.iter_mut().zip(guide) {
            *t = g * g;
        }
        box_mean(&self.tmp, w, h, r, &mut self.rows, &mut self.mean_ii);
        for i in 0..w * h {
            let (mi, mp) = (self.mean_i[i], self.mean_p[i]);
            let cov = self.mean_ip[i] - mi * mp;
            let var = self.mean_ii[i] - mi * mi;
            let a = cov / (var + self.eps);
            self.mean_ip[i] = a; // reuse as scratch for a and b
            self.mean_ii[i] = mp - a * mi;
        }
        box_mean(&self.mean_ip, w, h, r, &mut self.rows, &mut self.a);
        box_mean(&self.mean_ii, w, h, r, &mut self.rows, &mut self.b);
    }
}

/// Convenience wrapper allocating a [`GuidedFilter`] for one use.
#[cfg(test)]
pub fn guided_filter(
    guide: &[f32],
    p: &[f32],
    w: usize,
    h: usize,
    r: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut f = GuidedFilter::new(w, h, r, eps);
    f.apply(guide, p);
    (f.a, f.b)
}

/// Mean of `x` (`w`x`h`) over a (2r+1)x(2r+1) window into `out`, shrinking
/// the window at the borders. `rows` is scratch of the same size. The
/// vertical pass adds and removes whole rows, so it runs over contiguous
/// memory and vectorises.
fn box_mean(x: &[f32], w: usize, h: usize, r: usize, rows: &mut [f32], out: &mut [f32]) {
    // Horizontal: running sum along each row.
    for (src, dst) in x.chunks_exact(w).zip(rows.chunks_exact_mut(w)) {
        let mut sum: f32 = src[..r.min(w - 1) + 1].iter().sum();
        let (mut lo, mut hi) = (0usize, r.min(w - 1) + 1); // window [lo, hi)
        for (i, d) in dst.iter_mut().enumerate() {
            let (want_lo, want_hi) = (i.saturating_sub(r), (i + r + 1).min(w));
            while hi < want_hi {
                sum += src[hi];
                hi += 1;
            }
            while lo < want_lo {
                sum -= src[lo];
                lo += 1;
            }
            *d = sum / (hi - lo) as f32;
        }
    }
    // Vertical: keep a running sum of rows, writing each output row from it.
    let mut acc = vec![0f32; w];
    let (mut lo, mut hi) = (0usize, 0usize); // rows [lo, hi)
    for y in 0..h {
        let (want_lo, want_hi) = (y.saturating_sub(r), (y + r + 1).min(h));
        while hi < want_hi {
            for (a, v) in acc.iter_mut().zip(&rows[hi * w..][..w]) {
                *a += v;
            }
            hi += 1;
        }
        while lo < want_lo {
            for (a, v) in acc.iter_mut().zip(&rows[lo * w..][..w]) {
                *a -= v;
            }
            lo += 1;
        }
        let inv = 1.0 / (hi - lo) as f32;
        for (o, a) in out[y * w..][..w].iter_mut().zip(&acc) {
            *o = a * inv;
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

    #[test]
    fn box_mean_shrinks_at_borders() {
        // 5x1, radius 1: windows [0,1], [0,2], [1,3], [2,4], [3,4].
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = |w, h| {
            let (mut rows, mut out) = ([0.0; 5], [0.0; 5]);
            box_mean(&x, w, h, 1, &mut rows, &mut out);
            out
        };
        assert_eq!(mean(5, 1), [1.5, 2.0, 3.0, 4.0, 4.5]);
        // Vertical pass on a 1x5 column gives the same.
        assert_eq!(mean(1, 5), [1.5, 2.0, 3.0, 4.0, 4.5]);
    }

    #[test]
    fn guided_filter_reproduces_linear_inputs() {
        let (w, h) = (16, 12);
        let guide: Vec<f32> = (0..w * h).map(|i| ((i % w) as f32 / w as f32) * 0.8 + 0.1).collect();
        // A constant passes through with no dependence on the guide.
        let (a, b) = guided_filter(&guide, &vec![0.7; w * h], w, h, 2, 1e-3);
        assert!(a.iter().all(|v| v.abs() < 1e-4) && b.iter().all(|v| (v - 0.7).abs() < 1e-4));
        // An input that is a linear function of the guide is followed exactly
        // (up to the small regularisation).
        let p: Vec<f32> = guide.iter().map(|g| 0.5 * g + 0.2).collect();
        let (a, b) = guided_filter(&guide, &p, w, h, 2, 1e-6);
        for i in 0..w * h {
            let q = a[i] * guide[i] + b[i];
            assert!((q - p[i]).abs() < 1e-3, "pixel {i}: {q} vs {}", p[i]);
        }
    }

    #[test]
    fn smoothing_damps_flicker_where_the_image_is_still() {
        let (w, h) = (16, 8);
        let guide = vec![0.5; w * h];
        let mut sm = TemporalSmoother::new(w, h, 0.5, 0.08);
        // A mask flickering between 0 and 1 on a still image.
        let mut out = Vec::new();
        for i in 0..10 {
            let mut mask = vec![(i % 2) as f32; w * h];
            sm.apply(&guide, &mut mask);
            out.push(mask[0]);
        }
        // The first frame passes through; after that the swing shrinks from
        // 1 to about 1/3 (the fixed point of the alternating blend).
        assert_eq!(out[0], 0.0);
        let swing = (out[9] - out[8]).abs();
        assert!(swing < 0.4, "still flickering by {swing}");
    }

    #[test]
    fn smoothing_follows_moving_parts_and_resets() {
        let (w, h) = (16, 8);
        let mut sm = TemporalSmoother::new(w, h, 0.9, 0.08);
        let mut mask = vec![0.0; w * h];
        sm.apply(&vec![0.2; w * h], &mut mask);
        // The image changed a lot: the new mask is taken as is.
        let mut mask = vec![1.0; w * h];
        sm.apply(&vec![0.8; w * h], &mut mask);
        assert!(mask.iter().all(|&m| m == 1.0));
        // After a reset there is no previous frame to blend with.
        sm.reset();
        let mut mask = vec![0.0; w * h];
        sm.apply(&vec![0.8; w * h], &mut mask);
        assert!(mask.iter().all(|&m| m == 0.0));
    }

    #[test]
    fn turning_reuse_off_finishes_the_frame_in_flight() {
        let mut m = Matting::new(Box::<FakeMatte>::default(), true);
        let (f1, f2) = (frame(0.1), frame(0.5));
        a_of(&mut m, &f1); // leaves f1 in flight
        m.set_reuse(false); // FakeMatte asserts if a frame is started twice
        assert_eq!(a_of(&mut m, &f2), coefficients_for(&f2, N).0);
        // And back on: the first frame waits for its own result again.
        m.set_reuse(true);
        assert_eq!(a_of(&mut m, &f1), coefficients_for(&f1, N).0);
    }

    #[test]
    fn smoothing_strength_zero_passes_masks_through() {
        let (w, h) = (8, 4);
        let guide = vec![0.5; w * h];
        let mut sm = TemporalSmoother::new(w, h, 0.8, 0.08);
        let mut mask = vec![1.0; w * h];
        sm.apply(&guide, &mut mask);
        sm.set_strength(0.0);
        let mut mask = vec![0.0; w * h];
        sm.apply(&guide, &mut mask);
        assert!(mask.iter().all(|&m| m == 0.0));
    }
}
