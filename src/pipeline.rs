//! Frame processing: camera pixels are unpacked into planar RGB, effects run
//! on that, and the result is packed as YUY2 for the virtual camera.

use pipewire::spa::param::video::VideoFormat;

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

/// Unpack a camera frame of `frame`'s size into `frame`. Returns false if
/// `src` is too short to hold it.
pub fn unpack(format: InputFormat, src: &[u8], stride: usize, frame: &mut PlanarFrame) -> bool {
    let (width, height) = (frame.width, frame.height);
    let row_bytes = format.row_bytes(width);
    if height == 0 || stride < row_bytes || src.len() < (height - 1) * stride + row_bytes {
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
    debug_assert!(width % 2 == 0, "YUY2 needs an even width");
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
    fn apply(&mut self, frame: &mut PlanarFrame);
}

/// Flips the frame upside down. A stand-in until real effects exist.
pub struct VerticalFlip;

impl Effect for VerticalFlip {
    fn apply(&mut self, frame: &mut PlanarFrame) {
        let (width, height) = (frame.width(), frame.height());
        for plane in frame.planes_mut() {
            for y in 0..height / 2 {
                let (top, bottom) = plane.split_at_mut((height - 1 - y) * width);
                top[y * width..][..width].swap_with_slice(&mut bottom[..width]);
            }
        }
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
    fn vertical_flip() {
        let rows: Vec<[f32; 3]> = (0..6).map(|i| [i as f32, 0.0, 0.0]).collect();
        let mut frame = frame_from(2, 3, &rows);
        VerticalFlip.apply(&mut frame);
        assert_eq!(frame.planes()[0], [4.0, 5.0, 2.0, 3.0, 0.0, 1.0]);
    }
}
