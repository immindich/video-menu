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
    _pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<u32>;
// Three planes (R, G, B) of width*height samples in 0..1.
@group(0) @binding(2) var<storage, read_write> planar: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<u32>;

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
