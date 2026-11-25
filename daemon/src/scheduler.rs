//! Decides what happens to frames in flight, so the GPU (or CPU) stages of
//! some frames overlap the model's work on another. A pure state machine:
//! the worker reports what happened (a frame was ingested, the model
//! finished) and carries out the [`Action`]s it gets back, in order. That
//! keeps the ordering rules here, testable without threads or a GPU.
//!
//! Frames go through three stages around the model: ingest (into a slot of
//! the backend), the model, and finish (out of the slot). The model takes
//! one frame at a time, in order, so a recurrent model's state follows the
//! frames. While it works, the next frame is ingested and waits; a newer one
//! replaces it (the camera's newest frame matters, not every frame). When
//! the model is done, the waiting frame goes to the model at once and the
//! finished frame is composited while the model works on the next.
//!
//! With `reuse`, frames don't wait for the model at all: each is finished
//! straight after ingest with the latest result, and the model works on the
//! newest frame in the background. The first frame after a reset still
//! waits for its own result, as there is nothing to reuse yet.
//!
//! The blur of a frame uses the alpha of the last frame finished before it
//! (see [`crate::pipeline::CpuProcessor`]), so it's started once the frame
//! before it is finished.

/// Frames are numbered in the order they arrive.
pub type FrameId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Send the frame's model input to the model.
    Dispatch(FrameId),
    /// Start blurring the background of the frame in this slot.
    Blur(usize),
    /// Finish the frame in `slot` and send it on, freeing the slot. With
    /// `model`, composite with the model's latest result (which, straight
    /// after the model finished this frame, is its own).
    Finish { id: FrameId, slot: usize, model: bool },
    /// Forget a frame that won't be finished: free its slot, if it holds
    /// one, and its model input.
    Drop { id: FrameId, slot: Option<usize> },
}

/// A frame the model is working on, or that waits for it. `slot` is set if
/// the frame waits in the backend to be finished with its own result.
#[derive(Clone, Copy, Debug)]
struct Pending {
    id: FrameId,
    slot: Option<usize>,
}

#[derive(Debug)]
pub struct Scheduler {
    reuse: bool,
    free: Vec<usize>,
    in_model: Option<Pending>,
    waiting: Option<Pending>,
    /// The model has a result since the last reset.
    latest: bool,
    /// The frame in the model is to be discarded (see [`barrier`]).
    ///
    /// [`barrier`]: Self::barrier
    discard: bool,
}

impl Scheduler {
    pub fn new(slots: usize, reuse: bool) -> Self {
        Self {
            reuse,
            free: (0..slots).rev().collect(),
            in_model: None,
            waiting: None,
            latest: false,
            discard: false,
        }
    }

    /// A free slot to ingest a frame into. There always is one: at most one
    /// frame is in the model and one waits, each holding a slot, so three
    /// slots leave one for the next frame.
    pub fn take_slot(&mut self) -> usize {
        self.free.pop().expect("a free slot (the worker ingests one frame at a time)")
    }

    /// A frame was ingested into `slot`; `wants_model` if its effect runs
    /// the model (its input is ready). Whether the effect runs the model
    /// only changes across a [`barrier`](Self::barrier).
    pub fn ingested(&mut self, id: FrameId, slot: usize, wants_model: bool) -> Vec<Action> {
        if !wants_model {
            debug_assert!(!self.ahead(), "the effect changed without a barrier");
            self.free.push(slot);
            return vec![Action::Finish { id, slot, model: false }];
        }
        // Finishing at once mustn't overtake an earlier frame still waiting
        // for its own result.
        if self.reuse && self.latest && !self.ahead() {
            self.free.push(slot);
            let mut actions = vec![Action::Finish { id, slot, model: true }];
            actions.extend(self.offer(Pending { id, slot: None }));
            return actions;
        }
        self.offer(Pending { id, slot: Some(slot) })
    }

    /// Give a frame to the model if it's idle; otherwise it waits, replacing
    /// any frame already waiting.
    fn offer(&mut self, frame: Pending) -> Vec<Action> {
        if self.in_model.is_none() {
            self.in_model = Some(frame);
            let mut actions = vec![Action::Dispatch(frame.id)];
            actions.extend(frame.slot.map(Action::Blur));
            return actions;
        }
        let mut actions = Vec::new();
        if let Some(old) = self.waiting.replace(frame) {
            actions.push(self.drop_frame(old));
        }
        actions
    }

    /// Whether an earlier frame waits in a slot to be finished.
    fn ahead(&self) -> bool {
        [self.in_model, self.waiting].iter().flatten().any(|f| f.slot.is_some())
    }

    fn drop_frame(&mut self, frame: Pending) -> Action {
        self.free.extend(frame.slot);
        Action::Drop { id: frame.id, slot: frame.slot }
    }

    /// The model finished frame `id` (its result is now the latest).
    pub fn model_done(&mut self, id: FrameId) -> Vec<Action> {
        let done = self.in_model.take().expect("the model finished a frame it was given");
        assert_eq!(done.id, id, "the model finished frames out of order");
        if std::mem::take(&mut self.discard) {
            return vec![self.drop_frame(done)];
        }
        self.latest = true;
        // The next frame to the model first, so it stays busy. With reuse,
        // a frame that waited for its own result (there was none to reuse
        // when it came) needn't any more: it's finished with this one.
        let next = self.waiting.take();
        let mut actions = Vec::new();
        if let Some(next) = next {
            let slot = if self.reuse { None } else { next.slot };
            self.in_model = Some(Pending { id: next.id, slot });
            actions.push(Action::Dispatch(next.id));
        }
        if let Some(slot) = done.slot {
            self.free.push(slot);
            actions.push(Action::Finish { id, slot, model: true });
        }
        if let Some(Pending { id: next, slot: Some(slot) }) = next {
            if self.reuse {
                self.free.push(slot);
                actions.push(Action::Finish { id: next, slot, model: true });
            } else {
                // Its blur needed the frame before it finished.
                actions.push(Action::Blur(slot));
            }
        }
        actions
    }

    /// Switch between waiting for each frame's own result and reusing the
    /// latest. Frames already waiting keep waiting for their own.
    pub fn set_reuse(&mut self, reuse: bool) {
        self.reuse = reuse;
    }

    /// Prepare for a change that invalidates frames in flight (a reset, a
    /// new model, a new frame size): drop the waiting frame, and mark the
    /// frame in the model, if any, to be dropped when the model returns it.
    /// Returns the actions, and whether the caller must wait for the model
    /// (and pass its result to [`model_done`](Self::model_done)) before
    /// making the change. Afterwards nothing is reused until the model has
    /// a new result.
    pub fn barrier(&mut self) -> (Vec<Action>, bool) {
        let mut actions = Vec::new();
        if let Some(waiting) = self.waiting.take() {
            actions.push(self.drop_frame(waiting));
        }
        let busy = self.in_model.is_some();
        self.discard = busy;
        self.latest = false;
        (actions, busy)
    }

    /// Whether no frame is in the model or waiting for it.
    pub fn idle(&self) -> bool {
        self.in_model.is_none() && self.waiting.is_none()
    }

    #[cfg(test)]
    fn free_slots(&self) -> usize {
        self.free.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{Action::*, *};

    #[test]
    fn effect_without_model_finishes_at_once() {
        let mut s = Scheduler::new(3, false);
        let slot = s.take_slot();
        assert_eq!(s.ingested(1, slot, false), [Finish { id: 1, slot, model: false }]);
        assert_eq!(s.free_slots(), 3);
    }

    #[test]
    fn one_frame_at_a_time() {
        let mut s = Scheduler::new(3, false);
        let slot = s.take_slot();
        assert_eq!(s.ingested(1, slot, true), [Dispatch(1), Blur(slot)]);
        assert_eq!(s.model_done(1), [Finish { id: 1, slot, model: true }]);
        assert!(s.idle());
        assert_eq!(s.free_slots(), 3);
    }

    #[test]
    fn next_frame_waits_then_goes_to_the_model_before_the_finish() {
        let mut s = Scheduler::new(3, false);
        let a = s.take_slot();
        s.ingested(1, a, true);
        let b = s.take_slot();
        assert_eq!(s.ingested(2, b, true), []);
        // Dispatch 2 first, so the model stays busy; then finish 1; then
        // blur 2, whose weights need 1 finished.
        assert_eq!(s.model_done(1), [Dispatch(2), Finish { id: 1, slot: a, model: true }, Blur(b)]);
        assert_eq!(s.model_done(2), [Finish { id: 2, slot: b, model: true }]);
        assert_eq!(s.free_slots(), 3);
    }

    #[test]
    fn newer_frame_replaces_the_waiting_one() {
        let mut s = Scheduler::new(3, false);
        let a = s.take_slot();
        s.ingested(1, a, true);
        let b = s.take_slot();
        s.ingested(2, b, true);
        let c = s.take_slot();
        assert_eq!(s.ingested(3, c, true), [Drop { id: 2, slot: Some(b) }]);
        assert_eq!(s.model_done(1), [Dispatch(3), Finish { id: 1, slot: a, model: true }, Blur(c)]);
        assert_eq!(s.model_done(3), [Finish { id: 3, slot: c, model: true }]);
        assert_eq!(s.free_slots(), 3);
    }

    #[test]
    fn reuse_finishes_with_the_latest_result_after_the_first() {
        let mut s = Scheduler::new(3, true);
        // Nothing to reuse yet: the first frame waits for its own result.
        let a = s.take_slot();
        assert_eq!(s.ingested(1, a, true), [Dispatch(1), Blur(a)]);
        assert_eq!(s.model_done(1), [Finish { id: 1, slot: a, model: true }]);
        // Then frames finish at once and the model follows behind.
        let b = s.take_slot();
        assert_eq!(s.ingested(2, b, true), [Finish { id: 2, slot: b, model: true }, Dispatch(2)]);
        let c = s.take_slot();
        assert_eq!(s.ingested(3, c, true), [Finish { id: 3, slot: c, model: true }]);
        let d = s.take_slot();
        // 3 was waiting for the model; 4 replaces it (no slot to free).
        assert_eq!(s.ingested(4, d, true), [Finish { id: 4, slot: d, model: true }, Drop { id: 3, slot: None }]);
        assert_eq!(s.model_done(2), [Dispatch(4)]);
        assert_eq!(s.model_done(4), []);
        assert!(s.idle());
        assert_eq!(s.free_slots(), 3);
    }

    #[test]
    fn reuse_never_finishes_a_frame_ahead_of_an_earlier_one() {
        let mut s = Scheduler::new(3, true);
        let a = s.take_slot();
        s.ingested(1, a, true); // nothing to reuse: waits for its own result
        let b = s.take_slot();
        assert_eq!(s.ingested(2, b, true), []); // waits too
        // 1's result arrives: 1 is finished with it, and so is 2, which
        // needn't wait any more; 2 goes to the model.
        assert_eq!(
            s.model_done(1),
            [Dispatch(2), Finish { id: 1, slot: a, model: true }, Finish { id: 2, slot: b, model: true }]
        );
        let c = s.take_slot();
        assert_eq!(s.ingested(3, c, true), [Finish { id: 3, slot: c, model: true }]);
        assert_eq!(s.free_slots(), 3);
    }

    #[test]
    fn turning_reuse_off_makes_the_next_frame_wait_for_its_own_result() {
        let mut s = Scheduler::new(3, true);
        let a = s.take_slot();
        s.ingested(1, a, true);
        s.model_done(1);
        let b = s.take_slot();
        s.ingested(2, b, true); // finished at once; the model works on 2
        s.set_reuse(false);
        let c = s.take_slot();
        assert_eq!(s.ingested(3, c, true), []);
        // 2's result only updates the latest; 3 goes to the model and is
        // finished with its own result.
        assert_eq!(s.model_done(2), [Dispatch(3), Blur(c)]);
        assert_eq!(s.model_done(3), [Finish { id: 3, slot: c, model: true }]);
        assert_eq!(s.free_slots(), 3);
    }

    #[test]
    fn barrier_drops_frames_in_flight() {
        let mut s = Scheduler::new(3, false);
        let a = s.take_slot();
        s.ingested(1, a, true);
        let b = s.take_slot();
        s.ingested(2, b, true);
        let (actions, busy) = s.barrier();
        assert_eq!(actions, [Drop { id: 2, slot: Some(b) }]);
        assert!(busy);
        assert_eq!(s.model_done(1), [Drop { id: 1, slot: Some(a) }]);
        assert!(s.idle());
        assert_eq!(s.free_slots(), 3);
        // Nothing to reuse after a barrier, even in reuse mode.
        let mut r = Scheduler::new(3, true);
        let a = r.take_slot();
        r.ingested(1, a, true);
        r.model_done(1);
        assert_eq!(r.barrier(), (vec![], false));
        let b = r.take_slot();
        assert_eq!(r.ingested(2, b, true), [Dispatch(2), Blur(b)]);
    }

    /// Whatever order frames and results come in, slots never run out, every
    /// frame is finished or dropped exactly once, finished frames come out
    /// in order, and the model gets frames in order.
    #[test]
    fn any_sequence_keeps_the_invariants() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rand = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for reuse in [false, true] {
            let mut s = Scheduler::new(3, reuse);
            let (mut next_id, mut in_model, mut ended) = (1u64, Vec::<FrameId>::new(), Vec::new());
            let (mut finished, mut dispatched) = (Vec::new(), Vec::new());
            // Whether the effect runs the model; it changes only at barriers.
            let mut wants_model = true;
            for _ in 0..2000 {
                let actions = match rand() % 5 {
                    4 => {
                        s.set_reuse(rand() % 2 == 0);
                        continue;
                    }
                    0 | 1 => {
                        let (id, slot) = (next_id, s.take_slot());
                        next_id += 1;
                        s.ingested(id, slot, wants_model)
                    }
                    2 if !in_model.is_empty() => s.model_done(in_model.remove(0)),
                    3 if rand() % 16 == 0 => {
                        let (mut actions, busy) = s.barrier();
                        if busy {
                            actions.extend(s.model_done(in_model.remove(0)));
                        }
                        wants_model = rand() % 4 != 0;
                        if rand() % 2 == 0 {
                            s.set_reuse(rand() % 2 == 0);
                        }
                        actions
                    }
                    _ => continue,
                };
                for action in actions {
                    match action {
                        Dispatch(id) => {
                            in_model.push(id);
                            dispatched.push(id);
                        }
                        Finish { id, .. } => {
                            finished.push(id);
                            ended.push(id);
                        }
                        Drop { id, slot } => {
                            // A frame finished early (reuse) may still be
                            // dropped from the model's queue.
                            if slot.is_some() || !finished.contains(&id) {
                                ended.push(id);
                            }
                        }
                        Blur(_) => {}
                    }
                }
                assert!(in_model.len() <= 1, "the model got two frames at once");
            }
            assert!(finished.windows(2).all(|w| w[0] < w[1]), "frames finished out of order");
            assert!(dispatched.windows(2).all(|w| w[0] < w[1]), "the model got frames out of order");
            let mut unique = ended.clone();
            unique.sort();
            unique.dedup();
            assert_eq!(unique.len(), ended.len(), "a frame ended twice");
        }
    }
}
