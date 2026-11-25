//! Opt-in timeline of each frame's GPU work, for finding where the time
//! between the CPU and the GPU goes. Set `VIDEO_MENU_GPU_PROFILE=FILE` and
//! the GPU backend writes one JSON line per frame to `FILE`:
//!
//! - `marks`: CPU events (`[name, ns]`), e.g. each submit and when the wait
//!   for it returned;
//! - `passes`: each compute pass (`[name, submission, begin_ns, end_ns]`),
//!   from GPU timestamps;
//! - `submits`: the submissions in order, so passes can be matched to them;
//! - `freq_mhz`: the GPU's actual clock at the start and end of the frame.
//!
//! CPU times count from when the profiler started; GPU times are the GPU's
//! own clock, whose offset from the CPU's is unknown (no calibrated
//! timestamps in wgpu). Durations on either clock are exact, and comparing a
//! submission's CPU wait with its GPU busy time gives the overhead around
//! it. `tools/gpu_profile_report.py` summarises a file.
//!
//! Recording costs a timestamp per pass and one small readback per frame.

use std::{
    cell::RefCell,
    fs::File,
    io::{BufWriter, Write},
    sync::mpsc,
    time::Instant,
};

/// Timestamp queries available per frame: two per pass.
const QUERIES: u32 = 64;

/// The GPU's actual clock, as the xe driver reports it.
const FREQ_PATH: &str = "/sys/class/drm/card0/device/tile0/gt0/freq0/act_freq";

#[derive(Default)]
struct Frame {
    marks: Vec<(&'static str, u64)>,
    /// Name and submission of each pass, in query order.
    passes: Vec<(&'static str, usize)>,
    submits: Vec<&'static str>,
    freq_start: Option<u32>,
    label: String,
}

pub struct GpuProfiler {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f64,
    origin: Instant,
    out: RefCell<BufWriter<File>>,
    frame: RefCell<Frame>,
}

impl GpuProfiler {
    /// A profiler writing to `VIDEO_MENU_GPU_PROFILE`, if it is set. `None`
    /// (with a message) if the device can't write timestamps.
    pub fn from_env(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        let path = std::env::var("VIDEO_MENU_GPU_PROFILE").ok().filter(|p| !p.is_empty())?;
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            eprintln!("profile: the GPU has no timestamp queries; not profiling");
            return None;
        }
        let file = match File::create(&path) {
            Ok(file) => file,
            Err(e) => {
                eprintln!("profile: cannot write {path}: {e}");
                return None;
            }
        };
        eprintln!("profile: writing a GPU timeline to {path}");
        let bytes = QUERIES as u64 * 8;
        let buffer = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes, usage, mapped_at_creation: false })
        };
        use wgpu::BufferUsages as U;
        Some(Self {
            query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("profile"),
                ty: wgpu::QueryType::Timestamp,
                count: QUERIES,
            }),
            resolve: buffer("profile resolve", U::QUERY_RESOLVE | U::COPY_SRC),
            readback: buffer("profile readback", U::MAP_READ | U::COPY_DST),
            period_ns: queue.get_timestamp_period() as f64,
            origin: Instant::now(),
            out: RefCell::new(BufWriter::new(file)),
            frame: RefCell::default(),
        })
    }

    /// Start a frame's record; `label` describes it (size, effect).
    pub fn begin(&self, label: String) {
        *self.frame.borrow_mut() = Frame { freq_start: read_freq(), label, ..Frame::default() };
        self.mark("start");
    }

    /// Note that the CPU reached `name` now.
    pub fn mark(&self, name: &'static str) {
        let ns = self.origin.elapsed().as_nanos() as u64;
        self.frame.borrow_mut().marks.push((name, ns));
    }

    /// Note a submission now: passes recorded after this belong to the next.
    pub fn submitted(&self, name: &'static str) {
        self.mark(name);
        self.frame.borrow_mut().submits.push(name);
    }

    /// Timestamp writes for a pass called `name`, if there is room.
    pub fn pass(&self, name: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let mut frame = self.frame.borrow_mut();
        let i = frame.passes.len() as u32;
        if 2 * i + 2 > QUERIES {
            return None;
        }
        let submission = frame.submits.len();
        frame.passes.push((name, submission));
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(2 * i),
            end_of_pass_write_index: Some(2 * i + 1),
        })
    }

    /// Copy the frame's timestamps out; record in the frame's last encoder.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let n = 2 * self.frame.borrow().passes.len() as u32;
        if n > 0 {
            encoder.resolve_query_set(&self.query_set, 0..n, &self.resolve, 0);
            encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, n as u64 * 8);
        }
    }

    /// Read the timestamps back (the frame's work must have finished) and
    /// write the frame's line.
    pub fn finish(&self, device: &wgpu::Device) {
        self.mark("end");
        let frame = std::mem::take(&mut *self.frame.borrow_mut());
        let n = 2 * frame.passes.len() as u64;
        let mut stamps = Vec::new();
        if n > 0 {
            let (tx, rx) = mpsc::channel();
            self.readback.slice(..n * 8).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            if let Ok(Ok(())) = rx.recv() {
                if let Ok(mapped) = self.readback.slice(..n * 8).get_mapped_range() {
                    stamps = bytemuck::cast_slice::<u8, u64>(&mapped).to_vec();
                }
                self.readback.unmap();
            }
        }
        let ns = |ticks: u64| (ticks as f64 * self.period_ns) as u64;
        let passes: Vec<String> = frame
            .passes
            .iter()
            .enumerate()
            .filter_map(|(i, (name, sub))| {
                let (b, e) = (stamps.get(2 * i)?, stamps.get(2 * i + 1)?);
                Some(format!("[\"{name}\",{sub},{},{}]", ns(*b), ns(*e)))
            })
            .collect();
        let marks: Vec<String> = frame.marks.iter().map(|(m, t)| format!("[\"{m}\",{t}]")).collect();
        let submits: Vec<String> = frame.submits.iter().map(|s| format!("\"{s}\"")).collect();
        let freq = |f: Option<u32>| f.map_or("null".into(), |f| f.to_string());
        let line = format!(
            "{{\"label\":\"{}\",\"freq_mhz\":[{},{}],\"submits\":[{}],\"marks\":[{}],\"passes\":[{}]}}",
            frame.label,
            freq(frame.freq_start),
            freq(read_freq()),
            submits.join(","),
            marks.join(","),
            passes.join(",")
        );
        let mut out = self.out.borrow_mut();
        let _ = writeln!(out, "{line}");
        let _ = out.flush();
    }
}

fn read_freq() -> Option<u32> {
    std::fs::read_to_string(FREQ_PATH).ok()?.trim().parse().ok()
}
