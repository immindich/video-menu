//! The matting model: RVM's network without its full-resolution refiner
//! stage (see `tools/split_refiner.py`), run with OpenVINO, by default on the
//! NPU.
//!
//! The model sees a fixed `MODEL_WIDTH`x`MODEL_HEIGHT` copy of the frame and
//! returns per-pixel coefficients `A` and `b` (4 channels each: foreground
//! residual R, G, B and alpha). The caller upsamples them to the frame's
//! resolution and computes `A * [R, G, B, gray] + b` there; see
//! [`crate::pipeline::GreenScreen`] and `refine_composite` in
//! `pipeline.wgsl`.

use std::time::{Duration, Instant};

use half::f16;
use openvino::{Core, DeviceType, ElementType, InferRequest, PartialShape, RwPropertyKey, Shape, Tensor};

/// Resolution the model runs at. Frames of other sizes are resampled to it.
pub const MODEL_WIDTH: usize = 480;
pub const MODEL_HEIGHT: usize = 270;
pub const MODEL_PIXELS: usize = MODEL_WIDTH * MODEL_HEIGHT;

/// Default location of the split model, produced from RVM's ONNX export with
/// `tools/split_refiner.py`.
const DEFAULT_MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/rvm_mobilenetv3_fp16_core.onnx");

/// Recurrent state tensors: name prefix, channels, and downscale factor
/// relative to the model resolution (stride-2 convolutions round up).
const RECURRENT: [(&str, i64, i64); 4] = [("r1", 16, 2), ("r2", 20, 4), ("r3", 40, 8), ("r4", 64, 16)];

/// Something that turns a downsampled frame into refiner coefficients.
pub trait Matte {
    /// `src` is the frame at `MODEL_WIDTH`x`MODEL_HEIGHT` as three planes
    /// (R, G, B) of values in 0..1. Returns `A` and `b`, each four planes
    /// (fgr residual R, G, B, then alpha) at the same resolution.
    fn infer(&mut self, src: &[f16]) -> Result<(&[f16], &[f16]), String>;

    /// Forget temporal state, e.g. after switching cameras.
    fn reset(&mut self);

    /// How long the last [`infer`](Self::infer) took.
    fn last_time(&self) -> Duration;
}

/// RVM on OpenVINO.
pub struct RvmMatte {
    request: InferRequest,
    src: Tensor,
    /// Recurrent inputs, fed from the previous frame's outputs.
    state: Vec<Tensor>,
    a: Vec<f16>,
    b: Vec<f16>,
    device: String,
    last: Duration,
}

impl RvmMatte {
    /// Load and compile the model. `VIDEO_MENU_MODEL` overrides the model
    /// path and `VIDEO_MENU_MODEL_DEVICE` the device (default `NPU`).
    pub fn from_env() -> Result<Self, String> {
        let path = std::env::var("VIDEO_MENU_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
        let device = std::env::var("VIDEO_MENU_MODEL_DEVICE").unwrap_or_else(|_| "NPU".into());
        Self::load(&path, &device)
    }

    pub fn load(path: &str, device: &str) -> Result<Self, String> {
        if !std::path::Path::new(path).exists() {
            return Err(format!(
                "model {path} not found; create it with tools/split_refiner.py"
            ));
        }
        let mut core = Core::new().map_err(|e| format!("OpenVINO: {e}"))?;
        let dev: DeviceType = device.parse().map_err(|e| format!("device {device}: {e}"))?;
        if dev == DeviceType::NPU {
            // nixpkgs' openvino doesn't ship the plugin compiler
            // (libopenvino_intel_npu_compiler.so); use the NPU driver's.
            core.set_property(&dev, &RwPropertyKey::Other("NPU_COMPILER_TYPE".into()), "DRIVER")
                .map_err(ov_err("NPU compiler"))?;
        }
        // Compiling for the NPU takes ~10 s; cache the result between runs.
        if let Some(cache) = cache_dir()
            && let Err(e) = core.set_property(&dev, &RwPropertyKey::CacheDir, &cache)
        {
            eprintln!("model: not caching compiled model: {e}");
        }

        let mut model = core.read_model_from_file(path, "").map_err(ov_err("reading model"))?;
        let (h, w) = (MODEL_HEIGHT as i64, MODEL_WIDTH as i64);
        let mut shapes = vec![("src_sm".to_string(), vec![1, 3, h, w])];
        for (name, channels, f) in RECURRENT {
            shapes.push((format!("{name}i"), vec![1, channels, (h + f - 1) / f, (w + f - 1) / f]));
        }
        let partial: Vec<(String, PartialShape)> = shapes
            .iter()
            .map(|(n, s)| Ok((n.clone(), PartialShape::new_static(s.len() as i64, s)?)))
            .collect::<Result<_, openvino::InferenceError>>()
            .map_err(ov_err("model shapes"))?;
        let partial: Vec<(&str, &PartialShape)> = partial.iter().map(|(n, s)| (n.as_str(), s)).collect();
        model.reshape(&partial).map_err(ov_err("reshaping model"))?;

        let start = Instant::now();
        let mut compiled = core.compile_model(&model, dev).map_err(ov_err("compiling model"))?;
        eprintln!("model: compiled for {device} in {:.1}s", start.elapsed().as_secs_f64());
        let request = compiled.create_infer_request().map_err(ov_err("infer request"))?;

        let tensor = |dims: &[i64]| -> Result<Tensor, String> {
            let shape = Shape::new(dims).map_err(ov_err("shape"))?;
            let mut t = Tensor::new(ElementType::F16, &shape).map_err(ov_err("tensor"))?;
            t.get_raw_data_mut().map_err(ov_err("tensor"))?.fill(0);
            Ok(t)
        };
        let src = tensor(&shapes[0].1)?;
        let state = shapes[1..].iter().map(|(_, s)| tensor(s)).collect::<Result<_, _>>()?;
        Ok(Self {
            request,
            src,
            state,
            a: vec![f16::ZERO; 4 * MODEL_PIXELS],
            b: vec![f16::ZERO; 4 * MODEL_PIXELS],
            device: device.into(),
            last: Duration::ZERO,
        })
    }

    pub fn device(&self) -> &str {
        &self.device
    }

    fn run(&mut self, src: &[f16]) -> Result<(), openvino::InferenceError> {
        self.src
            .get_raw_data_mut()?
            .copy_from_slice(bytemuck::cast_slice(src));
        self.request.set_tensor("src_sm", &self.src)?;
        for ((name, _, _), state) in RECURRENT.iter().zip(&self.state) {
            self.request.set_tensor(&format!("{name}i"), state)?;
        }
        self.request.infer()?;
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
    fn infer(&mut self, src: &[f16]) -> Result<(&[f16], &[f16]), String> {
        if src.len() != 3 * MODEL_PIXELS {
            return Err(format!("model input has {} values, expected {}", src.len(), 3 * MODEL_PIXELS));
        }
        let start = Instant::now();
        self.run(src).map_err(|e| format!("inference failed: {e}"))?;
        self.last = start.elapsed();
        Ok((&self.a, &self.b))
    }

    fn reset(&mut self) {
        for state in &mut self.state {
            if let Ok(data) = state.get_raw_data_mut() {
                data.fill(0);
            }
        }
    }

    fn last_time(&self) -> Duration {
        self.last
    }
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
    #[derive(Default)]
    pub struct FakeMatte {
        a: Vec<f16>,
        b: Vec<f16>,
        pub frames: usize,
    }

    impl Matte for FakeMatte {
        fn infer(&mut self, src: &[f16]) -> Result<(&[f16], &[f16]), String> {
            let n = MODEL_PIXELS;
            assert_eq!(src.len(), 3 * n);
            self.a.resize(4 * n, f16::ZERO);
            self.b.resize(4 * n, f16::ZERO);
            for i in 0..n {
                let rgb = [src[i].to_f32(), src[n + i].to_f32(), src[2 * n + i].to_f32()];
                for c in 0..4 {
                    // Residuals in -0.2..0.2; alpha (c = 3) spanning 0..1.
                    let (a, b) = match c {
                        3 => (2.0 * rgb[0] - rgb[2], 0.5 * rgb[1] - 0.3),
                        _ => (0.2 * rgb[(c + 1) % 3] - 0.1, 0.1 - 0.2 * rgb[c]),
                    };
                    self.a[c * n + i] = f16::from_f32(a);
                    self.b[c * n + i] = f16::from_f32(b);
                }
            }
            self.frames += 1;
            Ok((&self.a, &self.b))
        }

        fn reset(&mut self) {}

        fn last_time(&self) -> Duration {
            Duration::ZERO
        }
    }
}
