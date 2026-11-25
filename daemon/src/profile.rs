//! Opt-in timeline of each frame's GPU work, for finding where the time
//! between the CPU and the GPU goes. Set `VIDEO_MENU_GPU_PROFILE=FILE` and
//! the GPU backend writes one JSON line per finished frame to `FILE`:
//!
//! - `marks`: CPU events of that frame (`[name, ns]`), e.g. each submit,
//!   when the wait for it returned, when it went to the model;
//! - `passes`: its compute passes (`[name, submission, begin_ns, end_ns]`),
//!   from GPU timestamps;
//! - `submits`: its submissions in order, so passes can be matched to them;
//! - `notes`: numbers the worker adds (`[name, value]`), such as how busy
//!   its threads were since the previous frame;
//! - `freq_mhz`: the GPU's actual clock when the frame was ingested and
//!   finished.
//!
//! Frames overlap (see [`crate::worker`]), so a record belongs to the
//! frame's slot from ingest to finish. CPU times count from when the
//! profiler started; GPU times are the GPU's own clock, whose offset from
//! the CPU's is unknown (no calibrated timestamps in wgpu). Durations on
//! either clock are exact, and comparing a submission's CPU wait with its
//! GPU busy time gives the overhead around it. `tools/gpu_profile_report.py`
//! summarises a file.
//!
//! Recording costs a timestamp per pass and one small readback per frame.

use std::{
    cell::RefCell,
    fs::File,
    io::{BufWriter, Write},
    sync::mpsc,
    time::Instant,
};

use crate::gpu::SLOTS;

/// Timestamp queries per slot: two per pass.
const PER_SLOT: u32 = 32;

/// The GPU's actual clock, as the xe driver reports it.
const FREQ_PATH: &str = "/sys/class/drm/card0/device/tile0/gt0/freq0/act_freq";

#[derive(Default)]
struct Frame {
    marks: Vec<(&'static str, u64)>,
    /// Name and submission of each pass, in query order.
    passes: Vec<(&'static str, usize)>,
    submits: Vec<&'static str>,
    notes: Vec<(&'static str, u64)>,
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
    frames: RefCell<Vec<Frame>>,
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
        let queries = SLOTS as u32 * PER_SLOT;
        let bytes = queries as u64 * 8;
        let buffer = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes, usage, mapped_at_creation: false })
        };
        use wgpu::BufferUsages as U;
        Some(Self {
            query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("profile"),
                ty: wgpu::QueryType::Timestamp,
                count: queries,
            }),
            resolve: buffer("profile resolve", U::QUERY_RESOLVE | U::COPY_SRC),
            readback: buffer("profile readback", U::MAP_READ | U::COPY_DST),
            period_ns: queue.get_timestamp_period() as f64,
            origin: Instant::now(),
            out: RefCell::new(BufWriter::new(file)),
            frames: RefCell::new((0..SLOTS).map(|_| Frame::default()).collect()),
        })
    }

    /// Start the record of the frame ingested into `slot`; `label`
    /// describes it (size, effect).
    pub fn begin(&self, slot: usize, label: String) {
        self.frames.borrow_mut()[slot] = Frame { freq_start: read_freq(), label, ..Frame::default() };
        self.mark(slot, "start");
    }

    /// Note that the CPU reached `name` now, for the frame in `slot`.
    pub fn mark(&self, slot: usize, name: &'static str) {
        let ns = self.origin.elapsed().as_nanos() as u64;
        self.frames.borrow_mut()[slot].marks.push((name, ns));
    }

    /// Note a number about the frame in `slot`.
    pub fn note(&self, slot: usize, name: &'static str, value: u64) {
        self.frames.borrow_mut()[slot].notes.push((name, value));
    }

    /// Note a submission for the frame in `slot` now: its passes recorded
    /// after this belong to the next.
    pub fn submitted(&self, slot: usize, name: &'static str) {
        self.mark(slot, name);
        self.frames.borrow_mut()[slot].submits.push(name);
    }

    /// Timestamp writes for a pass called `name` of the frame in `slot`, if
    /// there is room.
    pub fn pass(&self, slot: usize, name: &'static str) -> Option<wgpu::ComputePassTimestampWrites<'_>> {
        let mut frames = self.frames.borrow_mut();
        let frame = &mut frames[slot];
        let i = frame.passes.len() as u32;
        if 2 * i + 2 > PER_SLOT {
            return None;
        }
        frame.passes.push((name, frame.submits.len()));
        let base = slot as u32 * PER_SLOT;
        Some(wgpu::ComputePassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(base + 2 * i),
            end_of_pass_write_index: Some(base + 2 * i + 1),
        })
    }

    /// Copy the timestamps of the frame in `slot` out; record in its last
    /// encoder.
    pub fn resolve(&self, slot: usize, encoder: &mut wgpu::CommandEncoder) {
        let n = 2 * self.frames.borrow()[slot].passes.len() as u32;
        if n > 0 {
            let base = slot as u32 * PER_SLOT;
            let offset = base as u64 * 8;
            encoder.resolve_query_set(&self.query_set, base..base + n, &self.resolve, offset);
            encoder.copy_buffer_to_buffer(&self.resolve, offset, &self.readback, offset, n as u64 * 8);
        }
    }

    /// Read the timestamps of the frame in `slot` back (its work must have
    /// finished) and write its line.
    pub fn finish(&self, slot: usize, device: &wgpu::Device) {
        self.mark(slot, "end");
        let frame = std::mem::take(&mut self.frames.borrow_mut()[slot]);
        let n = 2 * frame.passes.len() as u64;
        let offset = slot as u64 * PER_SLOT as u64 * 8;
        let mut stamps = Vec::new();
        if n > 0 {
            let range = offset..offset + n * 8;
            let (tx, rx) = mpsc::channel();
            self.readback.slice(range.clone()).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            if let Ok(Ok(())) = rx.recv() {
                if let Ok(mapped) = self.readback.slice(range).get_mapped_range() {
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
        let pairs = |items: &[(&str, u64)]| -> String {
            items.iter().map(|(m, t)| format!("[\"{m}\",{t}]")).collect::<Vec<_>>().join(",")
        };
        let submits: Vec<String> = frame.submits.iter().map(|s| format!("\"{s}\"")).collect();
        let freq = |f: Option<u32>| f.map_or("null".into(), |f| f.to_string());
        let line = format!(
            "{{\"label\":\"{}\",\"freq_mhz\":[{},{}],\"submits\":[{}],\"marks\":[{}],\"notes\":[{}],\"passes\":[{}]}}",
            frame.label,
            freq(frame.freq_start),
            freq(read_freq()),
            submits.join(","),
            pairs(&frame.marks),
            pairs(&frame.notes),
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
