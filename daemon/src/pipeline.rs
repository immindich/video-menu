//! Frame processing: camera pixels are unpacked into planar RGB, effects run
//! on that, and the result is packed as YUY2 for the virtual camera.

use std::time::Duration;

use half::{f16, slice::HalfFloatSliceExt};
use pipewire::spa::param::video::VideoFormat;

use crate::model::Matting;

/// A frame as three planes (R, G, B) of `f32` samples in `0..=1`.
#[derive(Default)]
pub struct PlanarFrame {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

impl PlanarFrame {
    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    /// Change the frame size, reusing the existing allocation. Sample values
    /// are unspecified afterwards.
    pub fn resize(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;
        self.data.resize(3 * width * height, 0.0);
    }

    pub fn planes(&self) -> [&[f32]; 3] {
        let (r, gb) = self.data.split_at(self.width * self.height);
        let (g, b) = gb.split_at(self.width * self.height);
        [r, g, b]
    }

    pub fn planes_mut(&mut self) -> [&mut [f32]; 3] {
        let (r, gb) = self.data.split_at_mut(self.width * self.height);
        let (g, b) = gb.split_at_mut(self.width * self.height);
        [r, g, b]
    }
}

/// A camera pixel format we can unpack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputFormat {
    /// 8-bit packed RGB; the fields are byte offsets within a pixel.
    Packed { r: usize, g: usize, b: usize, bpp: usize },
    /// 4:2:2 YUV as Y0 U Y1 V, BT.601 limited range.
    Yuy2,
}

impl InputFormat {
    /// Camera formats we accept, most preferred first.
    pub const SUPPORTED: &[VideoFormat] = &[
        VideoFormat::BGRx,
        VideoFormat::RGBx,
        VideoFormat::BGRA,
        VideoFormat::RGBA,
        VideoFormat::YUY2,
    ];

    pub fn from_video_format(format: VideoFormat) -> Option<Self> {
        let rgb = Self::Packed { r: 0, g: 1, b: 2, bpp: 4 };
        let bgr = Self::Packed { r: 2, g: 1, b: 0, bpp: 4 };
        match format {
            VideoFormat::RGBx | VideoFormat::RGBA => Some(rgb),
            VideoFormat::BGRx | VideoFormat::BGRA => Some(bgr),
            VideoFormat::YUY2 => Some(Self::Yuy2),
            _ => None,
        }
    }

    /// Bytes per row of pixel data (excluding stride padding).
    pub fn row_bytes(self, width: usize) -> usize {
        match self {
            Self::Packed { bpp, .. } => width * bpp,
            Self::Yuy2 => width * 2,
        }
    }
}

/// Whether `len` bytes with rows `stride` apart hold a `width`x`height` frame.
pub fn frame_fits(format: InputFormat, len: usize, stride: usize, width: usize, height: usize) -> bool {
    let row_bytes = format.row_bytes(width);
    height > 0 && stride >= row_bytes && len >= (height - 1) * stride + row_bytes
}

/// Unpack a camera frame of `frame`'s size into `frame`. Returns false if
/// `src` is too short to hold it.
pub fn unpack(format: InputFormat, src: &[u8], stride: usize, frame: &mut PlanarFrame) -> bool {
    let (width, height) = (frame.width, frame.height);
    let row_bytes = format.row_bytes(width);
    if !frame_fits(format, src.len(), stride, width, height) {
        return false;
    }
    let [rp, gp, bp] = frame.planes_mut();
    let rows = src.chunks(stride).take(height);
    let planes = rp
        .chunks_exact_mut(width)
        .zip(gp.chunks_exact_mut(width))
        .zip(bp.chunks_exact_mut(width));
    match format {
        InputFormat::Packed { r, g, b, bpp } => {
            for (src, ((rr, gr), br)) in rows.zip(planes) {
                let px = src[..row_bytes].chunks_exact(bpp);
                for (p, ((r_out, g_out), b_out)) in
                    px.zip(rr.iter_mut().zip(gr.iter_mut()).zip(br.iter_mut()))
                {
                    *r_out = p[r] as f32 * (1.0 / 255.0);
                    *g_out = p[g] as f32 * (1.0 / 255.0);
                    *b_out = p[b] as f32 * (1.0 / 255.0);
                }
            }
        }
        InputFormat::Yuy2 => {
            for (src, ((rr, gr), br)) in rows.zip(planes) {
                let pairs = src[..row_bytes].chunks_exact(4);
                let out = rr
                    .chunks_exact_mut(2)
                    .zip(gr.chunks_exact_mut(2))
                    .zip(br.chunks_exact_mut(2));
                for (p, ((r2, g2), b2)) in pairs.zip(out) {
                    let cb = (p[1] as f32 - 128.0) * (1.0 / 224.0);
                    let cr = (p[3] as f32 - 128.0) * (1.0 / 224.0);
                    let (dr, dg, db) = (1.402 * cr, -0.344136 * cb - 0.714136 * cr, 1.772 * cb);
                    for (i, y) in [p[0], p[2]].into_iter().enumerate() {
                        let y = (y as f32 - 16.0) * (1.0 / 219.0);
                        r2[i] = (y + dr).clamp(0.0, 1.0);
                        g2[i] = (y + dg).clamp(0.0, 1.0);
                        b2[i] = (y + db).clamp(0.0, 1.0);
                    }
                }
            }
        }
    }
    true
}

/// Pack `frame` as YUY2 (BT.601, limited range) into `dst` with the given
/// row stride. Chroma is taken from the average colour of each pixel pair.
/// `frame`'s width must be even and `dst` must hold `stride * height` bytes.
pub fn pack_yuy2_bt601(frame: &PlanarFrame, dst: &mut [u8], stride: usize) {
    let width = frame.width;
    debug_assert!(width.is_multiple_of(2), "YUY2 needs an even width");
    debug_assert!(stride >= width * 2 && dst.len() >= stride * frame.height);
    let to_u8 = |v: f32| (v.clamp(0.0, 255.0) + 0.5) as u8;
    let luma = |r: f32, g: f32, b: f32| to_u8(16.0 + 65.481 * r + 128.553 * g + 24.966 * b);

    let [rp, gp, bp] = frame.planes();
    let planes = rp
        .chunks_exact(width)
        .zip(gp.chunks_exact(width))
        .zip(bp.chunks_exact(width));
    for (dst, ((rr, gr), br)) in dst.chunks_mut(stride).zip(planes) {
        let pairs = rr
            .chunks_exact(2)
            .zip(gr.chunks_exact(2))
            .zip(br.chunks_exact(2));
        for (out, ((r, g), b)) in dst[..width * 2].chunks_exact_mut(4).zip(pairs) {
            let (ra, ga, ba) = ((r[0] + r[1]) * 0.5, (g[0] + g[1]) * 0.5, (b[0] + b[1]) * 0.5);
            out[0] = luma(r[0], g[0], b[0]);
            out[1] = to_u8(128.0 - 37.797 * ra - 74.203 * ga + 112.0 * ba);
            out[2] = luma(r[1], g[1], b[1]);
            out[3] = to_u8(128.0 + 112.0 * ra - 93.786 * ga - 18.214 * ba);
        }
    }
}

/// An in-place transformation of a frame.
pub trait Effect {
    fn apply(&mut self, frame: &mut PlanarFrame) -> Result<(), String>;

    /// Time the last [`apply`](Self::apply) spent waiting for a model.
    fn model_time(&self) -> Duration {
        Duration::ZERO
    }

    /// Forget any temporal state, e.g. after switching cameras.
    fn reset(&mut self) {}

    /// The matting model this effect runs, if any, so its settings can be
    /// changed in place.
    fn matting_mut(&mut self) -> Option<&mut Matting> {
        None
    }

    /// Change what goes behind the person, for effects that replace the
    /// background.
    fn set_background(&mut self, _background: Background) {}
}

/// What [`ReplaceBackground`] puts behind the person.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Background {
    /// A solid colour, as 0..1 RGB like the frame's samples.
    Color([f32; 3]),
    /// The camera's own background, blurred by a Gaussian whose standard
    /// deviation is `radius` percent of the frame height.
    Blur { radius: f32 },
}

/// The blur works on a copy of the frame scaled down to about this many
/// rows, which makes a wide blur cheap and looks no different.
const BLUR_ROWS: usize = 270;

/// How much the person counts towards the blurred background. Not zero, so
/// the weights never vanish where the person covers a whole neighbourhood
/// (the person hides the background there anyway).
const BLUR_FOREGROUND_WEIGHT: f32 = 0.01;

/// The scaled-down copy the blur runs on, for a frame of a given size.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlurGeometry {
    /// Each blur sample averages a `factor`x`factor` block of the frame.
    pub factor: usize,
    pub width: usize,
    pub height: usize,
    /// Gaussian standard deviation, in blur samples.
    pub sigma: f32,
    /// Taps either side of the centre: three standard deviations.
    pub taps: usize,
}

impl BlurGeometry {
    pub fn new(width: usize, height: usize, radius: f32) -> Self {
        let factor = ((height + BLUR_ROWS / 2) / BLUR_ROWS).max(1);
        let sigma = (radius / 100.0 * height as f32 / factor as f32).max(0.3);
        Self {
            factor,
            width: width.div_ceil(factor),
            height: height.div_ceil(factor),
            sigma,
            taps: (3.0 * sigma).ceil() as usize,
        }
    }
}

/// The weight a pixel with alpha `pha` gives its colour in the blur.
fn blur_weight(pha: f32) -> f32 {
    1.0 - pha * (1.0 - BLUR_FOREGROUND_WEIGHT)
}

/// Scale `rgb` down to `g`'s size, weighting each pixel by
/// [`blur_weight`] of its alpha: `out` gets three planes of block averages
/// of weighted colour, then one of the weights. `pipeline.wgsl`'s
/// `blur_down` computes the same.
fn blur_down(rgb: [&[f32]; 3], alpha: &[f32], width: usize, height: usize, g: &BlurGeometry, out: &mut [f32]) {
    let n = g.width * g.height;
    for by in 0..g.height {
        for bx in 0..g.width {
            let mut sum = [0.0f32; 4];
            let (x0, y0) = (bx * g.factor, by * g.factor);
            let (x1, y1) = ((x0 + g.factor).min(width), (y0 + g.factor).min(height));
            for y in y0..y1 {
                for x in x0..x1 {
                    let i = y * width + x;
                    let w = blur_weight(alpha[i]);
                    for (c, plane) in rgb.iter().enumerate() {
                        sum[c] += plane[i] * w;
                    }
                    sum[3] += w;
                }
            }
            let count = ((x1 - x0) * (y1 - y0)) as f32;
            for (c, s) in sum.into_iter().enumerate() {
                out[c * n + by * g.width + bx] = s / count;
            }
        }
    }
}

/// The normalised Gaussian kernel for `g`, `2 * taps + 1` long.
fn gaussian(g: &BlurGeometry) -> Vec<f32> {
    let t = g.taps as isize;
    let k: Vec<f32> = (-t..=t)
        .map(|d| (-((d * d) as f32) / (2.0 * g.sigma * g.sigma)).exp())
        .collect();
    let sum: f32 = k.iter().sum();
    k.into_iter().map(|v| v / sum).collect()
}

/// Blur four planes of `g`'s size from `src` into `dst` along one axis,
/// clamping at the edges. `pipeline.wgsl`'s `blur_h` and `blur_v` compute
/// the same.
fn blur_pass(src: &[f32], dst: &mut [f32], g: &BlurGeometry, kernel: &[f32], horizontal: bool) {
    let (w, h) = (g.width, g.height);
    let t = g.taps as isize;
    for c in 0..4 {
        let (src, dst) = (&src[c * w * h..][..w * h], &mut dst[c * w * h..][..w * h]);
        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0;
                for (k, d) in kernel.iter().zip(-t..=t) {
                    let i = if horizontal {
                        y * w + (x as isize + d).clamp(0, w as isize - 1) as usize
                    } else {
                        (y as isize + d).clamp(0, h as isize - 1) as usize * w + x
                    };
                    sum += k * src[i];
                }
                dst[y * w + x] = sum;
            }
        }
    }
}

/// Bilinear resampling taps from `n_in` samples to `n_out`, matching
/// PyTorch's `align_corners=False` without antialiasing (ONNX Resize with
/// `pytorch_half_pixel`), which is how RVM resamples. Each tap is
/// `(i0, i1, weight of i1)`. `pipeline.wgsl` computes the same.
fn taps(n_in: usize, n_out: usize) -> Vec<(usize, usize, f32)> {
    let scale = n_in as f32 / n_out as f32;
    (0..n_out)
        .map(|i| {
            let x = ((i as f32 + 0.5) * scale - 0.5).clamp(0.0, (n_in - 1) as f32);
            let i0 = x as usize;
            (i0, (i0 + 1).min(n_in - 1), x - i0 as f32)
        })
        .collect()
}

/// Resample `planes` planes of `w`x`h` in `src` to `dw`x`dh` in `dst`.
fn resample(src: &[f32], w: usize, h: usize, dst: &mut [f32], dw: usize, dh: usize, planes: usize) {
    let (xs, ys) = (taps(w, dw), taps(h, dh));
    let mut row = vec![0f32; w];
    for p in 0..planes {
        let (src, dst) = (&src[p * w * h..][..w * h], &mut dst[p * dw * dh..][..dw * dh]);
        for (&(y0, y1, wy), out) in ys.iter().zip(dst.chunks_exact_mut(dw)) {
            let (r0, r1) = (&src[y0 * w..][..w], &src[y1 * w..][..w]);
            for ((v, a), b) in row.iter_mut().zip(r0).zip(r1) {
                *v = a * (1.0 - wy) + b * wy;
            }
            for (o, &(x0, x1, wx)) in out.iter_mut().zip(&xs) {
                *o = row[x0] * (1.0 - wx) + row[x1] * wx;
            }
        }
    }
}

/// Replaces the background with a colour or a blur of itself, using the
/// matting model to find the person.
///
/// The frame is downsampled to the model's resolution, the model returns
/// coefficients `A` and `b` there, and these are upsampled to the frame's
/// resolution, where `A * [R, G, B, gray] + b` gives the foreground colour
/// (as a residual on the frame) and alpha. This is RVM's own refiner step.
///
/// For a blur, the background is the frame scaled down, blurred with each
/// pixel weighted by how much it is background (so the person doesn't bleed
/// into it as a halo), and sampled back up.
pub struct ReplaceBackground {
    matte: Matting,
    background: Background,
    /// Size of the last frame; the model's temporal state is reset when it
    /// changes.
    size: (usize, usize),
    small: Vec<f32>,
    small16: Vec<f16>,
    coef: Vec<f32>,
    coef_full: Vec<f32>,
    /// Blur scratch: alpha at full resolution, the two scaled-down buffers
    /// the blur passes between, and the result back at full resolution.
    alpha: Vec<f32>,
    blur: [Vec<f32>; 2],
    blurred: Vec<f32>,
}

impl ReplaceBackground {
    pub fn new(matte: Matting, background: Background) -> Self {
        let (mw, mh) = matte.size();
        Self {
            matte,
            background,
            size: (0, 0),
            small: vec![0.0; 3 * mw * mh],
            small16: vec![f16::ZERO; 3 * mw * mh],
            coef: vec![0.0; 8 * mw * mh],
            coef_full: Vec::new(),
            alpha: Vec::new(),
            blur: Default::default(),
            blurred: Vec::new(),
        }
    }

    /// Blur the background of `frame` into `self.blurred`: four planes at
    /// full resolution, weighted colour and then the weight.
    fn blur_background(&mut self, frame: &PlanarFrame, radius: f32) {
        let (w, h) = (frame.width(), frame.height());
        let pixels = w * h;
        let c = |k: usize| &self.coef_full[k * pixels..][..pixels];
        let [rp, gp, bp] = frame.planes();
        self.alpha.resize(pixels, 0.0);
        for (i, a) in self.alpha.iter_mut().enumerate() {
            let gray = (rp[i] + gp[i] + bp[i]) * (1.0 / 3.0);
            *a = (c(3)[i] * gray + c(7)[i]).clamp(0.0, 1.0);
        }
        let g = BlurGeometry::new(w, h, radius);
        let kernel = gaussian(&g);
        let [a, b] = &mut self.blur;
        a.resize(4 * g.width * g.height, 0.0);
        b.resize(4 * g.width * g.height, 0.0);
        blur_down(frame.planes(), &self.alpha, w, h, &g, a);
        blur_pass(a, b, &g, &kernel, true);
        blur_pass(b, a, &g, &kernel, false);
        self.blurred.resize(4 * pixels, 0.0);
        resample(a, g.width, g.height, &mut self.blurred, w, h, 4);
    }
}

impl Effect for ReplaceBackground {
    fn apply(&mut self, frame: &mut PlanarFrame) -> Result<(), String> {
        let (w, h) = (frame.width(), frame.height());
        if self.size != (w, h) {
            self.matte.reset();
            self.size = (w, h);
        }
        let (mw, mh) = self.matte.size();
        resample(&frame.data, w, h, &mut self.small, mw, mh, 3);
        self.small16.convert_from_f32_slice(&self.small);
        let (a, b) = self.matte.run(&self.small16)?;
        let n = 4 * mw * mh;
        a.convert_to_f32_slice(&mut self.coef[..n]);
        b.convert_to_f32_slice(&mut self.coef[n..]);

        // Upsample all 8 coefficient planes (A then b) to the frame size.
        let pixels = w * h;
        self.coef_full.resize(8 * pixels, 0.0);
        resample(&self.coef, mw, mh, &mut self.coef_full, w, h, 8);

        let color = match self.background {
            Background::Color(color) => Some(color),
            Background::Blur { radius } => {
                self.blur_background(frame, radius);
                None
            }
        };
        let [rp, gp, bp] = frame.planes_mut();
        let c = |k: usize| &self.coef_full[k * pixels..][..pixels];
        let blurred = |k: usize| &self.blurred[k * pixels..][..pixels];
        for i in 0..pixels {
            let x = [rp[i], gp[i], bp[i], (rp[i] + gp[i] + bp[i]) * (1.0 / 3.0)];
            let out = |k: usize| c(k)[i] * x[k] + c(4 + k)[i];
            let pha = out(3).clamp(0.0, 1.0);
            let fgr = [0, 1, 2].map(|k| (out(k) + x[k]).clamp(0.0, 1.0));
            let bg = color.unwrap_or_else(|| {
                let weight = blurred(3)[i];
                [0, 1, 2].map(|k| blurred(k)[i] / weight)
            });
            rp[i] = fgr[0] * pha + bg[0] * (1.0 - pha);
            gp[i] = fgr[1] * pha + bg[1] * (1.0 - pha);
            bp[i] = fgr[2] * pha + bg[2] * (1.0 - pha);
        }
        Ok(())
    }

    fn model_time(&self) -> Duration {
        self.matte.last_time()
    }

    fn reset(&mut self) {
        self.matte.reset();
    }

    fn matting_mut(&mut self) -> Option<&mut Matting> {
        Some(&mut self.matte)
    }

    fn set_background(&mut self, background: Background) {
        self.background = background;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame_from(width: usize, height: usize, rgb: &[[f32; 3]]) -> PlanarFrame {
        let mut frame = PlanarFrame::default();
        frame.resize(width, height);
        let [r, g, b] = frame.planes_mut();
        for (i, px) in rgb.iter().enumerate() {
            (r[i], g[i], b[i]) = (px[0], px[1], px[2]);
        }
        frame
    }

    fn pack(frame: &PlanarFrame) -> Vec<u8> {
        let mut out = vec![0; frame.width() * 2 * frame.height()];
        pack_yuy2_bt601(frame, &mut out, frame.width() * 2);
        out
    }

    #[test]
    fn yuy2_reference_colours() {
        let solid = |c: [f32; 3]| pack(&frame_from(2, 1, &[c, c]));
        assert_eq!(solid([1.0, 1.0, 1.0]), [235, 128, 235, 128]);
        assert_eq!(solid([0.0, 0.0, 0.0]), [16, 128, 16, 128]);
        assert_eq!(solid([1.0, 0.0, 0.0]), [81, 90, 81, 240]);
        assert_eq!(solid([0.0, 1.0, 0.0]), [145, 54, 145, 34]);
        assert_eq!(solid([0.0, 0.0, 1.0]), [41, 240, 41, 110]);
    }

    #[test]
    fn yuy2_round_trip() {
        let (w, h) = (8, 3);
        let mut src = Vec::new();
        for i in 0..w * h / 2 {
            let y0 = 60 + (i * 37 % 140) as u8;
            let y1 = 60 + (i * 53 % 140) as u8;
            let u = 108 + (i * 11 % 40) as u8;
            let v = 108 + (i * 7 % 40) as u8;
            src.extend_from_slice(&[y0, u, y1, v]);
        }
        let mut frame = PlanarFrame::default();
        frame.resize(w, h);
        assert!(unpack(InputFormat::Yuy2, &src, w * 2, &mut frame));
        for (a, b) in pack(&frame).iter().zip(&src) {
            assert!(a.abs_diff(*b) <= 1, "{a} vs {b}");
        }
    }

    #[test]
    fn packed_channel_order() {
        let px = [10u8, 20, 30, 255];
        let mut frame = PlanarFrame::default();
        frame.resize(1, 1);
        let rgbx = InputFormat::from_video_format(VideoFormat::RGBx).unwrap();
        assert!(unpack(rgbx, &px, 4, &mut frame));
        assert_eq!(frame.planes().map(|p| (p[0] * 255.0).round() as u8), [10, 20, 30]);
        let bgrx = InputFormat::from_video_format(VideoFormat::BGRx).unwrap();
        assert!(unpack(bgrx, &px, 4, &mut frame));
        assert_eq!(frame.planes().map(|p| (p[0] * 255.0).round() as u8), [30, 20, 10]);
    }

    #[test]
    fn unpack_honours_stride_and_rejects_short_input() {
        let rgbx = InputFormat::from_video_format(VideoFormat::RGBx).unwrap();
        // 1x2 frame with 4 bytes of padding per row.
        let src = [1, 2, 3, 0, 9, 9, 9, 9, 4, 5, 6, 0];
        let mut frame = PlanarFrame::default();
        frame.resize(1, 2);
        assert!(unpack(rgbx, &src, 8, &mut frame));
        assert_eq!((frame.planes()[2][1] * 255.0).round() as u8, 6);
        assert!(!unpack(rgbx, &src[..11], 8, &mut frame));
    }

    #[test]
    fn resample_matches_pytorch_half_pixel() {
        // Downsampling by exactly 4 averages the middle two of each four
        // samples: (i + 0.5) * 4 - 0.5 = 4i + 1.5.
        let src: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut dst = [0.0; 2];
        resample(&src, 8, 1, &mut dst, 2, 1, 1);
        assert_eq!(dst, [1.5, 5.5]);
        // Upsampling by 2 clamps at the edges and interpolates between.
        let mut up = [0.0; 4];
        resample(&[0.0, 1.0], 2, 1, &mut up, 4, 1, 1);
        assert_eq!(up, [0.0, 0.25, 0.75, 1.0]);
    }

    const GREEN: [f32; 3] = [0.0, 177.0 / 255.0, 64.0 / 255.0];

    #[test]
    fn colour_background_composites_by_alpha() {
        use crate::model::fake::FakeMatte;
        let (w, h) = (64, 36);
        let rows: Vec<[f32; 3]> = (0..w * h)
            .map(|i| [(i % w) as f32 / w as f32, (i / w) as f32 / h as f32, 0.5])
            .collect();
        let mut frame = frame_from(w, h, &rows);
        let matting = Matting::new(Box::<FakeMatte>::default(), false);
        let mut effect = ReplaceBackground::new(matting, Background::Color(GREEN));
        effect.apply(&mut frame).unwrap();
        // Every output pixel lies between the (clamped) foreground and green.
        for (k, plane) in frame.planes().iter().enumerate() {
            assert!(plane.iter().all(|v| (0.0..=1.0).contains(v)), "channel {k} out of range");
        }
        // The fake model's alpha is 2R - B + ...: low on the left, high on
        // the right, so the left edge is green and the right is not.
        let [r, g, b] = frame.planes();
        let left = (r[0], g[0], b[0]);
        assert!((left.0 - GREEN[0]).abs() < 0.05 && (left.1 - GREEN[1]).abs() < 0.05, "{left:?}");
        let right = w - 1;
        assert!((g[right] - GREEN[1]).abs() > 0.1, "right edge should keep its own colour");
    }

    #[test]
    fn blur_geometry_scales_with_frame_height() {
        let hd = BlurGeometry::new(1920, 1080, 2.0);
        assert_eq!((hd.factor, hd.width, hd.height), (4, 480, 270));
        assert!((hd.sigma - 5.4).abs() < 1e-4 && hd.taps == 17, "{hd:?}");
        // 720p blurs at a third of its size, by the same share of the height.
        let hd720 = BlurGeometry::new(1280, 720, 2.0);
        assert_eq!((hd720.factor, hd720.width, hd720.height), (3, 427, 240));
        assert!((hd720.sigma * 3.0 - 14.4).abs() < 1e-3);
        // Small frames aren't scaled at all.
        assert_eq!(BlurGeometry::new(70, 37, 2.0).factor, 1);
    }

    /// The blurred background of a frame whose left half is a red person
    /// (alpha 1) in front of a blue background (alpha 0).
    fn blur_red_on_blue(width: usize) -> Vec<[f32; 3]> {
        let (h, g) = (8, BlurGeometry { factor: 1, width, height: 8, sigma: 3.0, taps: 9 });
        let half = width / 2;
        let rows: Vec<[f32; 3]> =
            (0..width * h).map(|i| if i % width < half { [1.0, 0.0, 0.0] } else { [0.0, 0.0, 1.0] }).collect();
        let frame = frame_from(width, h, &rows);
        let alpha: Vec<f32> = (0..width * h).map(|i| if i % width < half { 1.0 } else { 0.0 }).collect();
        let n = width * h;
        let (mut a, mut b) = (vec![0.0; 4 * n], vec![0.0; 4 * n]);
        blur_down(frame.planes(), &alpha, width, h, &g, &mut a);
        let kernel = gaussian(&g);
        assert!((kernel.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        blur_pass(&a, &mut b, &g, &kernel, true);
        blur_pass(&b, &mut a, &g, &kernel, false);
        (0..width).map(|x| [0, 1, 2].map(|c| a[c * n + 3 * width + x] / a[3 * n + 3 * width + x])).collect()
    }

    #[test]
    fn blur_keeps_the_person_out_of_the_background() {
        let row = blur_red_on_blue(32);
        // Right next to the person, the background is still blue: the red
        // pixels carry almost no weight.
        let edge = row[16];
        assert!(edge[0] < 0.05 && edge[2] > 0.95, "{edge:?}");
        // Deep inside the person every weight is small, so there it is a
        // plain blur of the person; the person covers it anyway.
        let inside = row[0];
        assert!(inside[0] > 0.95, "{inside:?}");
    }

    #[test]
    fn blur_of_a_flat_background_is_flat() {
        let (w, h) = (20, 10);
        let g = BlurGeometry::new(w, h, 8.0);
        let frame = frame_from(w, h, &vec![[0.2, 0.4, 0.6]; w * h]);
        let alpha = vec![0.0; w * h];
        let n = g.width * g.height;
        let (mut a, mut b) = (vec![0.0; 4 * n], vec![0.0; 4 * n]);
        blur_down(frame.planes(), &alpha, w, h, &g, &mut a);
        let kernel = gaussian(&g);
        blur_pass(&a, &mut b, &g, &kernel, true);
        blur_pass(&b, &mut a, &g, &kernel, false);
        for (c, want) in [0.2, 0.4, 0.6].into_iter().enumerate() {
            assert!(a[c * n..][..n].iter().all(|v| (v - want).abs() < 1e-5));
        }
    }
}
