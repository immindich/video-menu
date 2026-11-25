// GPU version of src/pipeline.rs: unpack camera pixels into planar RGB,
// apply effects, pack as YUY2 (BT.601, limited range). Each entry point
// mirrors a CPU function there; keep the two in sync.
//
// Buffers hold raw bytes as little-endian u32 words. Rows of `src` and `dst`
// are tightly packed (the caller strips any stride padding).

struct Params {
    width: u32,
    height: u32,
    // u32 words per row of `src` and `dst`.
    src_words: u32,
    dst_words: u32,
    // Bit offsets of R, G and B within a packed-RGB pixel word.
    r_shift: u32,
    g_shift: u32,
    b_shift: u32,
    _pad0: u32,
    // Resolution the matting model runs at.
    small_w: u32,
    small_h: u32,
    _pad1: u32,
    _pad2: u32,
    // Background colour for composite_color.
    bg: vec4<f32>,
    // The blur's scaled-down copy (see BlurGeometry in pipeline.rs).
    blur_w: u32,
    blur_h: u32,
    blur_factor: u32,
    blur_taps: u32,
    // The blur kernel's weights for offsets 0..=blur_taps, four per vec4
    // (uniform arrays need 16-byte elements).
    blur_weights: array<vec4<f32>, 16>,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
// Three planes (R, G, B) of width*height samples in 0..1.
@group(0) @binding(2) var<storage, read_write> planar: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;
// The frame at model resolution: three planes of f16, two per word.
@group(0) @binding(4) var<storage, read_write> small: array<u32>;
// The model's coefficients A then b: eight planes of f16 at model resolution.
@group(0) @binding(5) var<storage, read> coef: array<u32>;
// The blur's two scaled-down buffers of blur_w*blur_h samples, each
// (weighted R, G, B, weight). Passes go a -> b -> a. (The CPU keeps these
// as planes; here one load fetches a whole sample.)
@group(0) @binding(6) var<storage, read_write> blur_a: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> blur_b: array<vec4<f32>>;
// The coefficients unpacked: A then b (as vec4) for each model pixel.
@group(0) @binding(8) var<storage, read_write> coef4: array<vec4<f32>>;
// Just alpha's coefficients (A, b) for each model pixel, for blur_down.
@group(0) @binding(9) var<storage, read_write> coef_alpha: array<vec2<f32>>;

fn byte_at(word: u32, shift: u32) -> f32 {
    return f32((word >> shift) & 0xffu);
}

fn store_rgb(i: u32, rgb: vec3<f32>) {
    let n = p.width * p.height;
    planar[i] = rgb.r;
    planar[n + i] = rgb.g;
    planar[2u * n + i] = rgb.b;
}

fn load_rgb(i: u32) -> vec3<f32> {
    let n = p.width * p.height;
    return vec3<f32>(planar[i], planar[n + i], planar[2u * n + i]);
}

// One invocation per pixel.
@compute @workgroup_size(16, 16)
fn unpack_packed(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.width || id.y >= p.height) {
        return;
    }
    let word = src[id.y * p.src_words + id.x];
    let rgb = vec3<f32>(byte_at(word, p.r_shift), byte_at(word, p.g_shift), byte_at(word, p.b_shift));
    store_rgb(id.y * p.width + id.x, rgb * (1.0 / 255.0));
}

// One invocation per pixel pair (one Y0 U Y1 V word).
@compute @workgroup_size(16, 16)
fn unpack_yuy2(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x * 2u >= p.width || id.y >= p.height) {
        return;
    }
    let word = src[id.y * p.src_words + id.x];
    let cb = (byte_at(word, 8u) - 128.0) * (1.0 / 224.0);
    let cr = (byte_at(word, 24u) - 128.0) * (1.0 / 224.0);
    let d = vec3<f32>(1.402 * cr, -0.344136 * cb - 0.714136 * cr, 1.772 * cb);
    let y0 = (byte_at(word, 0u) - 16.0) * (1.0 / 219.0);
    let y1 = (byte_at(word, 16u) - 16.0) * (1.0 / 219.0);
    let i = id.y * p.width + id.x * 2u;
    store_rgb(i, clamp(vec3<f32>(y0) + d, vec3<f32>(0.0), vec3<f32>(1.0)));
    store_rgb(i + 1u, clamp(vec3<f32>(y1) + d, vec3<f32>(0.0), vec3<f32>(1.0)));
}

// One invocation per pixel in the top half; swaps it with its mirror.
@compute @workgroup_size(16, 16)
fn vertical_flip(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.width || id.y >= p.height / 2u) {
        return;
    }
    let n = p.width * p.height;
    let a = id.y * p.width + id.x;
    let b = (p.height - 1u - id.y) * p.width + id.x;
    for (var c = 0u; c < 3u; c++) {
        let t = planar[c * n + a];
        planar[c * n + a] = planar[c * n + b];
        planar[c * n + b] = t;
    }
}

fn to_u8(v: f32) -> u32 {
    return u32(clamp(v, 0.0, 255.0) + 0.5);
}

fn luma(rgb: vec3<f32>) -> u32 {
    return to_u8(16.0 + dot(rgb, vec3<f32>(65.481, 128.553, 24.966)));
}

// One invocation per pixel pair; chroma from the pair's average colour.
@compute @workgroup_size(16, 16)
fn pack_yuy2(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x * 2u >= p.width || id.y >= p.height) {
        return;
    }
    let i = id.y * p.width + id.x * 2u;
    let a = load_rgb(i);
    let b = load_rgb(i + 1u);
    let m = (a + b) * 0.5;
    let u = to_u8(128.0 + dot(m, vec3<f32>(-37.797, -74.203, 112.0)));
    let v = to_u8(128.0 + dot(m, vec3<f32>(112.0, -93.786, -18.214)));
    dst[id.y * p.dst_words + id.x] = luma(a) | (u << 8u) | (luma(b) << 16u) | (v << 24u);
}

// ---- matting (see ReplaceBackground in pipeline.rs) ----

struct Tap {
    i0: u32,
    i1: u32,
    w: f32,
}

// Bilinear tap for output sample i when resampling n_in samples to n_out,
// as PyTorch does with align_corners=False. Mirrors taps() in pipeline.rs.
fn tap(i: u32, n_in: u32, n_out: u32) -> Tap {
    let scale = f32(n_in) / f32(n_out);
    let x = clamp((f32(i) + 0.5) * scale - 0.5, 0.0, f32(n_in - 1u));
    let i0 = u32(x);
    return Tap(i0, min(i0 + 1u, n_in - 1u), x - f32(i0));
}

fn planar_at(c: u32, x: u32, y: u32) -> f32 {
    return planar[c * p.width * p.height + y * p.width + x];
}

// Bilinear sample of full-resolution plane c (vertical, then horizontal,
// like resample() in pipeline.rs).
fn sample_planar(c: u32, tx: Tap, ty: Tap) -> f32 {
    let left = planar_at(c, tx.i0, ty.i0) * (1.0 - ty.w) + planar_at(c, tx.i0, ty.i1) * ty.w;
    let right = planar_at(c, tx.i1, ty.i0) * (1.0 - ty.w) + planar_at(c, tx.i1, ty.i1) * ty.w;
    return left * (1.0 - tx.w) + right * tx.w;
}

// One invocation per horizontal pair of model-resolution pixels.
@compute @workgroup_size(16, 16)
fn downsample(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x * 2u >= p.small_w || id.y >= p.small_h) {
        return;
    }
    let x = id.x * 2u;
    let ty = tap(id.y, p.height, p.small_h);
    let t0 = tap(x, p.width, p.small_w);
    let t1 = tap(x + 1u, p.width, p.small_w);
    let plane_words = p.small_w * p.small_h / 2u;
    let word = (id.y * p.small_w + x) / 2u;
    for (var c = 0u; c < 3u; c++) {
        let v = vec2<f32>(sample_planar(c, t0, ty), sample_planar(c, t1, ty));
        small[c * plane_words + word] = pack2x16float(v);
    }
}

fn coef_at(k: u32, i: u32) -> f32 {
    let j = k * p.small_w * p.small_h + i;
    return unpack2x16float(coef[j / 2u])[j % 2u];
}

// One invocation per model pixel: unpack its eight coefficients into coef4,
// and alpha's into coef_alpha.
@compute @workgroup_size(16, 16)
fn unpack_coef(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.small_w || id.y >= p.small_h) {
        return;
    }
    let i = id.y * p.small_w + id.x;
    coef4[2u * i] = vec4<f32>(coef_at(0u, i), coef_at(1u, i), coef_at(2u, i), coef_at(3u, i));
    coef4[2u * i + 1u] = vec4<f32>(coef_at(4u, i), coef_at(5u, i), coef_at(6u, i), coef_at(7u, i));
    coef_alpha[i] = vec2<f32>(coef_at(3u, i), coef_at(7u, i));
}

struct Coef {
    a: vec4<f32>,
    b: vec4<f32>,
}

// Bilinear sample of the coefficients (vertical, then horizontal, like
// resample() in pipeline.rs; per component the arithmetic is the same).
fn sample_coef(tx: Tap, ty: Tap) -> Coef {
    let r0 = ty.i0 * p.small_w;
    let r1 = ty.i1 * p.small_w;
    let wy = ty.w;
    let a_left = coef4[2u * (r0 + tx.i0)] * (1.0 - wy) + coef4[2u * (r1 + tx.i0)] * wy;
    let a_right = coef4[2u * (r0 + tx.i1)] * (1.0 - wy) + coef4[2u * (r1 + tx.i1)] * wy;
    let b_left = coef4[2u * (r0 + tx.i0) + 1u] * (1.0 - wy) + coef4[2u * (r1 + tx.i0) + 1u] * wy;
    let b_right = coef4[2u * (r0 + tx.i1) + 1u] * (1.0 - wy) + coef4[2u * (r1 + tx.i1) + 1u] * wy;
    return Coef(a_left * (1.0 - tx.w) + a_right * tx.w, b_left * (1.0 - tx.w) + b_right * tx.w);
}

// Just the alpha of matte_at, from coef_alpha: a quarter of the memory
// traffic, for blur_down, which needs it at every pixel.
fn alpha_at(x: u32, y: u32) -> f32 {
    let tx = tap(x, p.small_w, p.width);
    let ty = tap(y, p.small_h, p.height);
    let r0 = ty.i0 * p.small_w;
    let r1 = ty.i1 * p.small_w;
    let left = coef_alpha[r0 + tx.i0] * (1.0 - ty.w) + coef_alpha[r1 + tx.i0] * ty.w;
    let right = coef_alpha[r0 + tx.i1] * (1.0 - ty.w) + coef_alpha[r1 + tx.i1] * ty.w;
    let c = left * (1.0 - tx.w) + right * tx.w;
    let rgb = load_rgb(y * p.width + x);
    return clamp(c.x * (rgb.r + rgb.g + rgb.b) * (1.0 / 3.0) + c.y, 0.0, 1.0);
}

struct Matte {
    fgr: vec3<f32>,
    pha: f32,
}

// Upsample the coefficients to pixel (x, y) and apply them to its colour,
// giving its foreground colour and alpha.
fn matte_at(x: u32, y: u32) -> Matte {
    let c = sample_coef(tap(x, p.small_w, p.width), tap(y, p.small_h, p.height));
    let rgb = load_rgb(y * p.width + x);
    let out = c.a * vec4<f32>(rgb, (rgb.r + rgb.g + rgb.b) * (1.0 / 3.0)) + c.b;
    return Matte(clamp(out.rgb + rgb, vec3<f32>(0.0), vec3<f32>(1.0)), clamp(out.a, 0.0, 1.0));
}

// One invocation per pixel: composite the foreground over p.bg.
@compute @workgroup_size(16, 16)
fn composite_color(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.width || id.y >= p.height) {
        return;
    }
    let m = matte_at(id.x, id.y);
    store_rgb(id.y * p.width + id.x, m.fgr * m.pha + p.bg.rgb * (1.0 - m.pha));
}

// ---- background blur (see blur_down, blur_pass in pipeline.rs) ----

const BLUR_FOREGROUND_WEIGHT: f32 = 0.01;

// One invocation per blur sample: the block average of colour weighted by
// how much each pixel is background (by the previous frame's matte), and of
// the weights, into blur_a.
@compute @workgroup_size(16, 16)
fn blur_down(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.blur_w || id.y >= p.blur_h) {
        return;
    }
    let x0 = id.x * p.blur_factor;
    let y0 = id.y * p.blur_factor;
    let x1 = min(x0 + p.blur_factor, p.width);
    let y1 = min(y0 + p.blur_factor, p.height);
    var sum = vec4<f32>(0.0);
    for (var y = y0; y < y1; y++) {
        for (var x = x0; x < x1; x++) {
            let rgb = load_rgb(y * p.width + x);
            let w = 1.0 - alpha_at(x, y) * (1.0 - BLUR_FOREGROUND_WEIGHT);
            sum += vec4<f32>(rgb * w, w);
        }
    }
    let count = f32((x1 - x0) * (y1 - y0));
    blur_a[id.y * p.blur_w + id.x] = sum / count;
}

fn blur_weight(d: i32) -> f32 {
    let i = u32(abs(d));
    return p.blur_weights[i / 4u][i % 4u];
}

// One invocation per blur sample: blur_a -> blur_b along x.
@compute @workgroup_size(16, 16)
fn blur_h(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.blur_w || id.y >= p.blur_h) {
        return;
    }
    let t = i32(p.blur_taps);
    let row = id.y * p.blur_w;
    var sum = vec4<f32>(0.0);
    for (var d = -t; d <= t; d++) {
        sum += blur_weight(d) * blur_a[row + u32(clamp(i32(id.x) + d, 0, i32(p.blur_w) - 1))];
    }
    blur_b[row + id.x] = sum;
}

// One invocation per blur sample: blur_b -> blur_a along y.
@compute @workgroup_size(16, 16)
fn blur_v(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.blur_w || id.y >= p.blur_h) {
        return;
    }
    let t = i32(p.blur_taps);
    var sum = vec4<f32>(0.0);
    for (var d = -t; d <= t; d++) {
        sum += blur_weight(d) * blur_b[u32(clamp(i32(id.y) + d, 0, i32(p.blur_h) - 1)) * p.blur_w + id.x];
    }
    blur_a[id.y * p.blur_w + id.x] = sum;
}

fn sample_blur(tx: Tap, ty: Tap) -> vec4<f32> {
    let r0 = ty.i0 * p.blur_w;
    let r1 = ty.i1 * p.blur_w;
    let left = blur_a[r0 + tx.i0] * (1.0 - ty.w) + blur_a[r1 + tx.i0] * ty.w;
    let right = blur_a[r0 + tx.i1] * (1.0 - ty.w) + blur_a[r1 + tx.i1] * ty.w;
    return left * (1.0 - tx.w) + right * tx.w;
}

// One invocation per pixel: composite the foreground over the blurred
// background, sampled up from blur_a.
@compute @workgroup_size(16, 16)
fn composite_blur(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.width || id.y >= p.height) {
        return;
    }
    let m = matte_at(id.x, id.y);
    let tx = tap(id.x, p.blur_w, p.width);
    let ty = tap(id.y, p.blur_h, p.height);
    let blurred = sample_blur(tx, ty);
    let bg = blurred.rgb / blurred.w;
    store_rgb(id.y * p.width + id.x, m.fgr * m.pha + bg * (1.0 - m.pha));
}
