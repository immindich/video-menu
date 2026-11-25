//! Processes camera frames on threads of their own, so the stages of
//! consecutive frames overlap: while the model works on one frame, the
//! backend ingests the next and finishes the previous (see
//! [`crate::scheduler`] for the rules).
//!
//! Two threads:
//! - the **pipeline** thread owns the backend (GPU or CPU), the JPEG
//!   decoder and the scheduler, and does everything but the model, in the
//!   order the scheduler says;
//! - the **model** thread owns the matting model and runs one inference at
//!   a time, in order, reporting back to the pipeline thread.
//!
//! The owner (the relay, on PipeWire's main loop) sends frames and settings
//! through a [`Worker`] and gets finished frames back through a callback.
//! Changes that invalidate frames in flight (new effects, a reset, a new
//! frame size) wait for the model to finish its current frame and drop it.

use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, Mutex, mpsc},
    thread::JoinHandle,
    time::{Duration, Instant},
};

use half::f16;

use crate::{
    model::Matte,
    pipeline::Background,
    processor::{Backend, Decoder, Effects, FrameFormat},
    scheduler::{Action, FrameId, Scheduler},
};

/// How many frames the backend holds at once (see [`Scheduler::take_slot`]).
const SLOTS: usize = 3;

/// A camera frame to process.
pub struct Job {
    /// Passed through to the [`Output`], so the owner can tell frames from
    /// before a change it made apart.
    pub generation: u64,
    pub format: FrameFormat,
    /// Raw pixels, rows `stride` apart, or a JPEG for MJPG.
    pub data: Vec<u8>,
    pub stride: usize,
    pub width: usize,
    pub height: usize,
    /// Row stride of the YUY2 output.
    pub out_stride: usize,
    pub pts: i64,
    pub seq: u64,
    /// When the frame came from the camera, for the latency.
    pub arrived: Instant,
}

/// A finished frame.
pub struct Output {
    pub generation: u64,
    /// YUY2, rows `out_stride` apart. Hand it back with
    /// [`Worker::recycle`] when done with it.
    pub frame: Vec<u8>,
    pub pts: i64,
    pub seq: u64,
    /// From the camera to here.
    pub latency: Duration,
    /// Time in each of [`crate::processor::STAGES`]. The model's is its
    /// inference time (for reused results, the latest inference's).
    pub stages: [Duration; 5],
    /// Frames dropped since the previous output: replaced by newer ones
    /// before the model got to them.
    pub dropped: u32,
}

enum Event {
    Frame(Job),
    Effects(Effects),
    Background(Background),
    Reuse(bool),
    Smoothing(f32),
    Reset,
    Flush(mpsc::Sender<()>),
    Recycle(Vec<u8>),
    ModelDone { id: FrameId, a: Vec<f16>, b: Vec<f16>, input: Vec<f16>, time: Duration },
    ModelFailed { id: FrameId, input: Vec<f16>, error: String },
    Stop,
}

enum ModelJob {
    Run { id: FrameId, input: Vec<f16> },
    Replace(Option<Box<dyn Matte>>),
    Reset,
    Smoothing(f32),
}

/// The owner's handle on the worker threads; dropping it stops them.
pub struct Worker {
    tx: mpsc::Sender<Event>,
    thread: Option<JoinHandle<()>>,
    /// Camera-frame buffers, recycled between the owner and the worker.
    buffers: Arc<Mutex<Vec<Vec<u8>>>>,
    description: String,
}

impl Worker {
    /// Start processing on `backend` with `effects`; `output` gets each
    /// finished frame, on the pipeline thread.
    pub fn start(backend: Backend, effects: Effects, output: impl FnMut(Output) + Send + 'static) -> Self {
        let (tx, rx) = mpsc::channel();
        let buffers = Arc::new(Mutex::new(Vec::new()));
        let description = backend.describe();
        let thread = std::thread::Builder::new()
            .name("pipeline".into())
            .spawn({
                let (tx, buffers) = (tx.clone(), buffers.clone());
                move || Pipeline::run(backend, effects, rx, tx, buffers, Box::new(output))
            })
            .expect("starting the pipeline thread");
        Self { tx, thread: Some(thread), buffers, description }
    }

    /// Where frames are processed, e.g. "GPU (Intel(R) Graphics)".
    pub fn describe(&self) -> &str {
        &self.description
    }

    /// An empty buffer to copy a camera frame into, reusing one the worker
    /// is done with if there is one.
    pub fn buffer(&self) -> Vec<u8> {
        let mut buffer = self.buffers.lock().unwrap().pop().unwrap_or_default();
        buffer.clear();
        buffer
    }

    pub fn send(&self, job: Job) {
        self.event(Event::Frame(job));
    }

    /// Hand back an [`Output::frame`] for reuse.
    pub fn recycle(&self, frame: Vec<u8>) {
        self.event(Event::Recycle(frame));
    }

    /// Replace the effects. Frames in flight are dropped.
    pub fn set_effects(&self, effects: Effects) {
        self.event(Event::Effects(effects));
    }

    /// Change what the background effect puts behind the person, from the
    /// next frame on.
    pub fn set_background(&self, background: Background) {
        self.event(Event::Background(background));
    }

    pub fn set_reuse(&self, reuse: bool) {
        self.event(Event::Reuse(reuse));
    }

    pub fn set_smoothing(&self, strength: f32) {
        self.event(Event::Smoothing(strength));
    }

    /// Forget temporal state (e.g. after switching cameras); frames in
    /// flight are dropped.
    pub fn reset(&self) {
        self.event(Event::Reset);
    }

    /// Wait until every frame sent so far has been finished or dropped and
    /// the model is idle.
    pub fn flush(&self) {
        let (tx, rx) = mpsc::channel();
        self.event(Event::Flush(tx));
        let _ = rx.recv();
    }

    fn event(&self, event: Event) {
        // The thread only stops when we are dropped.
        let _ = self.tx.send(event);
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.event(Event::Stop);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// A frame between ingest and finish.
struct InFlight {
    generation: u64,
    pts: i64,
    seq: u64,
    arrived: Instant,
    height: usize,
    out_stride: usize,
    stages: [Duration; 5],
}

/// The pipeline thread's state.
struct Pipeline {
    backend: Backend,
    decoder: Decoder,
    sched: Scheduler,
    rx: mpsc::Receiver<Event>,
    /// Events that came while waiting for the model in a barrier.
    deferred: VecDeque<Event>,
    model: mpsc::Sender<ModelJob>,
    model_thread: Option<JoinHandle<()>>,
    /// The model's resolution, if the effect runs one.
    model_size: Option<(usize, usize)>,
    background: Option<Background>,
    /// Size of the last frame; a new size resets temporal state.
    size: (usize, usize),
    next_id: FrameId,
    frames: HashMap<FrameId, InFlight>,
    /// Which frame each slot holds.
    slots: [Option<FrameId>; SLOTS],
    /// Model inputs of frames not yet given to the model.
    inputs: HashMap<FrameId, Vec<f16>>,
    input_pool: Vec<Vec<f16>>,
    /// The model's latest result, A and b.
    latest: (Vec<f16>, Vec<f16>),
    latest_time: Duration,
    out_pool: Vec<Vec<u8>>,
    buffers: Arc<Mutex<Vec<Vec<u8>>>>,
    output: Box<dyn FnMut(Output) + Send>,
    dropped: u32,
    flushes: Vec<mpsc::Sender<()>>,
}

impl Pipeline {
    fn run(
        backend: Backend,
        effects: Effects,
        rx: mpsc::Receiver<Event>,
        tx: mpsc::Sender<Event>,
        buffers: Arc<Mutex<Vec<Vec<u8>>>>,
        output: Box<dyn FnMut(Output) + Send>,
    ) {
        let (model, model_rx) = mpsc::channel();
        let model_thread = std::thread::Builder::new()
            .name("model".into())
            .spawn(move || run_model(model_rx, tx))
            .expect("starting the model thread");
        let mut pipeline = Self {
            backend,
            decoder: Decoder::default(),
            sched: Scheduler::new(SLOTS, false),
            rx,
            deferred: VecDeque::new(),
            model,
            model_thread: Some(model_thread),
            model_size: None,
            background: None,
            size: (0, 0),
            next_id: 1,
            frames: HashMap::new(),
            slots: [None; SLOTS],
            inputs: HashMap::new(),
            input_pool: Vec::new(),
            latest: (Vec::new(), Vec::new()),
            latest_time: Duration::ZERO,
            out_pool: Vec::new(),
            buffers,
            output,
            dropped: 0,
            flushes: Vec::new(),
        };
        pipeline.set_effects(effects);
        loop {
            let event = match pipeline.deferred.pop_front() {
                Some(event) => event,
                None => match pipeline.rx.recv() {
                    Ok(event) => event,
                    Err(_) => break,
                },
            };
            if !pipeline.handle(event) {
                break;
            }
            if pipeline.sched.idle() && !pipeline.flushes.is_empty() {
                for flush in pipeline.flushes.drain(..) {
                    let _ = flush.send(());
                }
            }
        }
        // Stop the model thread (it drops the model there) and wait for it.
        let Pipeline { model, model_thread, .. } = pipeline;
        drop(model);
        if let Some(thread) = model_thread {
            let _ = thread.join();
        }
    }

    /// Handle one event; false to stop.
    fn handle(&mut self, event: Event) -> bool {
        match event {
            Event::Frame(job) => {
                // Only the newest frame matters: if another is already
                // queued, skip this one.
                self.deferred.extend(self.rx.try_iter());
                if self.deferred.iter().any(|e| matches!(e, Event::Frame(_))) {
                    self.dropped += 1;
                    self.give_back(job.data);
                } else {
                    self.frame(job);
                }
            }
            Event::Effects(effects) => {
                self.barrier();
                self.set_effects(effects);
            }
            Event::Background(background) => {
                if self.background.is_some() {
                    self.background = Some(background);
                }
            }
            Event::Reuse(reuse) => self.sched.set_reuse(reuse),
            Event::Smoothing(strength) => {
                let _ = self.model.send(ModelJob::Smoothing(strength));
            }
            Event::Reset => {
                self.barrier();
                self.reset();
            }
            Event::Flush(reply) => self.flushes.push(reply),
            Event::Recycle(frame) => self.out_pool.push(frame),
            Event::ModelDone { id, a, b, input, time } => {
                self.input_pool.push(input);
                (self.latest, self.latest_time) = ((a, b), time);
                if let Some(frame) = self.frames.get_mut(&id) {
                    frame.stages[3] = time;
                }
                let actions = self.sched.model_done(id);
                self.act(actions);
            }
            Event::ModelFailed { id, input, error } => {
                eprintln!("pipeline: model failed: {error}");
                self.input_pool.push(input);
                let actions = self.sched.model_failed(id);
                self.act(actions);
            }
            Event::Stop => return false,
        }
        true
    }

    fn set_effects(&mut self, effects: Effects) {
        let (matte, reuse, background) = match effects {
            Effects::None => (None, false, None),
            Effects::Background { matte, reuse, background } => (Some(matte), reuse, Some(background)),
        };
        self.model_size = matte.as_ref().map(|m| m.size());
        if let Some(size) = self.model_size {
            self.backend.set_model_size(size);
        }
        self.background = background;
        self.sched.set_reuse(reuse);
        self.backend.reset_alpha();
        // The old model is dropped on the model thread, after its last job.
        let _ = self.model.send(ModelJob::Replace(matte));
    }

    fn reset(&mut self) {
        self.backend.reset_alpha();
        let _ = self.model.send(ModelJob::Reset);
    }

    /// Drop the frames in flight: the waiting one at once, the one in the
    /// model once the model is done with it (waiting for that here, and
    /// keeping other events for later).
    fn barrier(&mut self) {
        let (actions, busy) = self.sched.barrier();
        self.act(actions);
        if !busy {
            return;
        }
        // The model's result may already have been read (looking for newer
        // frames); otherwise wait for it, keeping other events for later.
        let is_result = |e: &Event| matches!(e, Event::ModelDone { .. } | Event::ModelFailed { .. });
        let mut early = self.deferred.iter().position(is_result).and_then(|i| self.deferred.remove(i));
        loop {
            let Some(event) = early.take().or_else(|| self.rx.recv().ok()) else { return };
            match event {
                Event::ModelDone { id, input, .. } => {
                    self.input_pool.push(input);
                    let actions = self.sched.model_done(id);
                    self.act(actions);
                    return;
                }
                Event::ModelFailed { id, input, .. } => {
                    self.input_pool.push(input);
                    let actions = self.sched.model_failed(id);
                    self.act(actions);
                    return;
                }
                other => self.deferred.push_back(other),
            }
        }
    }

    /// Decode and ingest a camera frame, then do what the scheduler says.
    fn frame(&mut self, job: Job) {
        let (width, height) = (job.width, job.height);
        if self.size != (width, height) {
            self.barrier();
            self.reset();
            self.size = (width, height);
        }
        let id = self.next_id;
        self.next_id += 1;
        let mut stages = [Duration::ZERO; 5];
        let slot = self.sched.take_slot();
        let wants = (|| {
            let t = Instant::now();
            let frame = self.decoder.decode(job.format, &job.data, job.stride, width, height, self.model_size)?;
            if let FrameFormat::Mjpg = job.format {
                stages[0] = t.elapsed();
            }
            let t = Instant::now();
            let background = self.background.filter(|_| self.model_size.is_some());
            let wants = self.backend.ingest(
                slot,
                frame.input,
                frame.src,
                frame.stride,
                width,
                height,
                background,
                frame.model_input,
            )?;
            stages[1] = t.elapsed();
            Ok::<_, String>(wants)
        })();
        let (generation, pts, seq, arrived, out_stride) = (job.generation, job.pts, job.seq, job.arrived, job.out_stride);
        self.give_back(job.data);
        let wants = match wants {
            Ok(wants) => wants,
            Err(e) => {
                eprintln!("pipeline: dropping frame: {e}");
                self.sched.release_slot(slot);
                self.dropped += 1;
                return;
            }
        };
        if wants {
            let mut input = self.input_pool.pop().unwrap_or_default();
            input.clear();
            input.extend_from_slice(self.backend.model_input());
            self.inputs.insert(id, input);
        }
        self.frames.insert(id, InFlight { generation, pts, seq, arrived, height, out_stride, stages });
        self.slots[slot] = Some(id);
        let actions = self.sched.ingested(id, slot, wants);
        self.act(actions);
    }

    /// Carry out the scheduler's actions, in order.
    fn act(&mut self, actions: Vec<Action>) {
        for action in actions {
            match action {
                Action::Dispatch(id) => {
                    let input = self.inputs.remove(&id).expect("a model input for each frame dispatched");
                    let _ = self.model.send(ModelJob::Run { id, input });
                }
                Action::Blur(slot) => {
                    let t = Instant::now();
                    self.backend.blur(slot);
                    if let Some(frame) = self.slots[slot].and_then(|id| self.frames.get_mut(&id)) {
                        frame.stages[2] += t.elapsed();
                    }
                }
                Action::Finish { id, slot, model } => self.finish(id, slot, model),
                Action::Drop { id, slot } => {
                    if let Some(input) = self.inputs.remove(&id) {
                        self.input_pool.push(input);
                    }
                    if let Some(slot) = slot {
                        self.slots[slot] = None;
                        self.frames.remove(&id);
                        self.dropped += 1;
                    }
                }
            }
        }
    }

    fn finish(&mut self, id: FrameId, slot: usize, model: bool) {
        self.slots[slot] = None;
        let Some(mut frame) = self.frames.remove(&id) else { return };
        let mut out = self.out_pool.pop().unwrap_or_default();
        out.resize(frame.out_stride * frame.height, 0);
        let t = Instant::now();
        let coefficients = model.then_some((self.latest.0.as_slice(), self.latest.1.as_slice()));
        if let Err(e) = self.backend.finish(slot, coefficients, &mut out, frame.out_stride) {
            eprintln!("pipeline: dropping frame: {e}");
            self.out_pool.push(out);
            self.dropped += 1;
            return;
        }
        frame.stages[4] = t.elapsed();
        if model && frame.stages[3].is_zero() {
            frame.stages[3] = self.latest_time;
        }
        (self.output)(Output {
            generation: frame.generation,
            frame: out,
            pts: frame.pts,
            seq: frame.seq,
            latency: frame.arrived.elapsed(),
            stages: frame.stages,
            dropped: std::mem::take(&mut self.dropped),
        });
    }

    fn give_back(&self, buffer: Vec<u8>) {
        self.buffers.lock().unwrap().push(buffer);
    }
}

/// The model thread: run each job in order, reporting results to the
/// pipeline thread, until the pipeline drops its sender.
fn run_model(jobs: mpsc::Receiver<ModelJob>, results: mpsc::Sender<Event>) {
    let mut matte: Option<Box<dyn Matte>> = None;
    for job in jobs {
        match job {
            ModelJob::Run { id, input } => {
                let t = Instant::now();
                let result = match &mut matte {
                    Some(m) => m.start(&input).and_then(|()| m.finish()).map(|()| m.coefficients()),
                    None => Err("no model".into()),
                };
                let event = match result {
                    Ok((a, b)) => Event::ModelDone { id, a: a.to_vec(), b: b.to_vec(), input, time: t.elapsed() },
                    Err(error) => Event::ModelFailed { id, input, error },
                };
                if results.send(event).is_err() {
                    break;
                }
            }
            ModelJob::Replace(new) => matte = new,
            ModelJob::Reset => {
                if let Some(m) = &mut matte {
                    m.reset();
                }
            }
            ModelJob::Smoothing(strength) => {
                if let Some(m) = &mut matte {
                    m.set_smoothing(strength);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use pipewire::spa::param::video::VideoFormat;

    use super::*;
    use crate::{
        gpu::GpuProcessor,
        model::fake::FakeMatte,
        pipeline::InputFormat,
        processor::Processor,
    };

    const GREEN: Background = Background::Color([0.0, 177.0 / 255.0, 64.0 / 255.0]);

    fn rgbx() -> InputFormat {
        InputFormat::from_video_format(VideoFormat::RGBx).unwrap()
    }

    /// A frame with gradients and fine detail that changes with `t`.
    fn pattern(w: usize, h: usize, t: usize) -> Vec<u8> {
        let mut px = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let v = (x * 7 + y * 13 + (x * y) % 31 + t * 17) % 256;
                px.extend([v as u8, (255 * x / w) as u8, (255 * y / h) as u8, 0]);
            }
        }
        px
    }

    fn jpeg(px: &[u8], w: usize, h: usize) -> Vec<u8> {
        let image = turbojpeg::Image { pixels: px, width: w, pitch: w * 4, height: h, format: turbojpeg::PixelFormat::RGBX };
        turbojpeg::compress(image, 90, turbojpeg::Subsamp::Sub2x1).unwrap().to_vec()
    }

    fn effects(bg: Option<Background>, reuse: bool) -> Effects {
        match bg {
            None => Effects::None,
            Some(background) => Effects::Background { matte: Box::new(FakeMatte::with_size(480, 270)), reuse, background },
        }
    }

    fn gpu() -> Option<Backend> {
        match GpuProcessor::new() {
            Ok(gpu) => Some(Backend::Gpu(Box::new(gpu))),
            Err(e) => {
                eprintln!("skipping: no GPU ({e})");
                None
            }
        }
    }

    fn start(backend: Backend, effects: Effects) -> (Worker, mpsc::Receiver<Output>) {
        let (tx, rx) = mpsc::channel();
        let worker = Worker::start(backend, effects, move |out| {
            let _ = tx.send(out);
        });
        (worker, rx)
    }

    fn job(worker: &Worker, format: FrameFormat, data: &[u8], (w, h): (usize, usize), seq: u64) -> Job {
        let mut buffer = worker.buffer();
        buffer.extend_from_slice(data);
        Job {
            generation: 0,
            format,
            data: buffer,
            stride: w * 4,
            width: w,
            height: h,
            out_stride: w * 2,
            pts: seq as i64,
            seq,
            arrived: Instant::now(),
        }
    }

    /// One frame in flight at a time, the threaded pipeline gives what the
    /// serial processor gives, for each effect, raw and MJPG input, with
    /// and without reuse.
    #[test]
    fn matches_the_serial_processor() {
        let (w, h) = (640, 360);
        let cases = [(None, false), (Some(GREEN), false), (Some(Background::Blur { radius: 2.0 }), false), (Some(GREEN), true), (Some(Background::Blur { radius: 3.0 }), true)];
        for (bg, reuse) in cases {
            for mjpg in [false, true] {
                let Some(backend) = gpu() else { return };
                let (worker, outputs) = start(backend, effects(bg, reuse));
                let mut serial = Processor::cpu(effects(bg, reuse));
                for t in 0..4 {
                    let px = pattern(w, h, t);
                    let (format, data) = match mjpg {
                        false => (FrameFormat::Raw(rgbx()), px),
                        true => (FrameFormat::Mjpg, jpeg(&px, w, h)),
                    };
                    worker.send(job(&worker, format, &data, (w, h), t as u64));
                    worker.flush();
                    let out = outputs.try_recv().expect("a frame out per frame in");
                    let mut want = vec![0u8; w * 2 * h];
                    serial.process(format, &data, w * 4, w, h, &mut want, w * 2).unwrap();
                    let worst = out.frame.iter().zip(&want).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
                    assert!(worst <= 2, "{bg:?} reuse {reuse} mjpg {mjpg} frame {t}: off by {worst}");
                    assert_eq!((out.seq, out.dropped), (t as u64, 0));
                    worker.recycle(out.frame);
                }
            }
        }
    }

    /// Frames sent much faster than they can be processed come out in
    /// order, every frame is either output or counted as dropped, and the
    /// worker doesn't hang.
    #[test]
    fn keeps_up_by_dropping_frames_in_order() {
        let (w, h) = (320, 180);
        for (bg, reuse) in [(None, false), (Some(GREEN), false), (Some(Background::Blur { radius: 2.0 }), true)] {
            let backend = Backend::Cpu(Box::default());
            let (worker, outputs) = start(backend, effects(bg, reuse));
            const FRAMES: u64 = 60;
            for seq in 0..FRAMES {
                worker.send(job(&worker, FrameFormat::Raw(rgbx()), &pattern(w, h, seq as usize), (w, h), seq));
            }
            worker.flush();
            let outs: Vec<Output> = outputs.try_iter().collect();
            let seqs: Vec<u64> = outs.iter().map(|o| o.seq).collect();
            assert!(seqs.windows(2).all(|s| s[0] < s[1]), "{bg:?}: out of order: {seqs:?}");
            let dropped: u64 = outs.iter().map(|o| o.dropped as u64).sum();
            // Drops after the last output aren't reported with a frame.
            assert!(seqs.len() as u64 + dropped <= FRAMES, "{bg:?}: {} out, {dropped} dropped", seqs.len());
            assert_eq!(seqs.last(), Some(&(FRAMES - 1)), "{bg:?}: the newest frame comes out");
        }
    }

    /// Settings change while frames are in flight: frames from before a
    /// change are dropped or finished, and processing goes on.
    #[test]
    fn changes_mid_stream() {
        let (w, h) = (320, 180);
        let backend = Backend::Cpu(Box::default());
        let (worker, outputs) = start(backend, effects(Some(GREEN), false));
        let mut seq = 0;
        let mut send = |worker: &Worker, n: usize, size: (usize, usize)| {
            for _ in 0..n {
                worker.send(job(worker, FrameFormat::Raw(rgbx()), &pattern(size.0, size.1, seq as usize), size, seq));
                seq += 1;
            }
        };
        send(&worker, 5, (w, h));
        worker.set_background(Background::Blur { radius: 2.0 });
        send(&worker, 5, (w, h));
        worker.set_effects(effects(Some(GREEN), true));
        send(&worker, 5, (w, h));
        worker.set_reuse(false);
        worker.set_smoothing(0.3);
        send(&worker, 5, (w, h));
        worker.reset();
        send(&worker, 5, (w, h));
        // A new frame size, then no effect at all.
        send(&worker, 5, (160, 90));
        worker.set_effects(Effects::None);
        send(&worker, 5, (160, 90));
        worker.flush();
        let outs: Vec<Output> = outputs.try_iter().collect();
        let seqs: Vec<u64> = outs.iter().map(|o| o.seq).collect();
        assert!(seqs.windows(2).all(|s| s[0] < s[1]), "out of order: {seqs:?}");
        assert_eq!(seqs.last(), Some(&(seq - 1)), "the last frame comes out");
        let last = outs.last().unwrap();
        assert_eq!(last.frame.len(), 160 * 2 * 90);
    }
}
