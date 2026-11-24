//! Settings and preview window for the video-menu daemon.
//!
//! Changes settings over D-Bus (see [`relay`]); the daemon must be running,
//! and the window shows when it isn't. `--preview` starts with the live
//! preview on.

mod preview;
mod relay;

use eframe::egui;
use video_menu_protocol::{EffectChoice, ModelChoice};

use preview::Preview;
use relay::{Command, RelayClient, RelayState};

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Video Menu")
            .with_app_id("video-menu")
            .with_inner_size([960.0, 560.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Video Menu",
        options,
        Box::new(|cc| {
            let ctx = cc.egui_ctx.clone();
            let relay = RelayClient::start(move || ctx.request_repaint());
            let preview = std::env::args().any(|a| a == "--preview").then(|| {
                let ctx = cc.egui_ctx.clone();
                Preview::start(move || ctx.request_repaint())
            });
            Ok(Box::new(App {
                relay,
                smoothing_edit: None,
                preview,
                texture: None,
                frames: FrameCounter::default(),
            }))
        }),
    )
}

struct App {
    relay: RelayClient,
    /// The smoothing slider's value while it is being dragged; sent to the
    /// daemon when the drag ends, rather than on every frame of it.
    smoothing_edit: Option<f64>,
    /// The live preview, while it is switched on.
    preview: Option<Preview>,
    /// The latest preview frame on the GPU.
    texture: Option<egui::TextureHandle>,
    frames: FrameCounter,
}

/// Counts preview frames to show the rate they arrive at.
#[derive(Default)]
struct FrameCounter {
    count: u32,
    since: Option<std::time::Instant>,
    fps: f64,
}

impl FrameCounter {
    fn tick(&mut self) {
        let since = *self.since.get_or_insert_with(std::time::Instant::now);
        self.count += 1;
        let elapsed = since.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            self.fps = self.count as f64 / elapsed;
            *self = Self { fps: self.fps, ..Self::default() };
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let state = self.relay.state.lock().unwrap().clone();
        egui::Panel::left("relay")
            .resizable(false)
            .exact_size(300.0)
            .show(ui, |ui| self.relay_panel(ui, &state));
        egui::CentralPanel::default().show(ui, |ui| self.preview_panel(ui, state.connected));
    }
}

impl App {
    fn relay_panel(&mut self, ui: &mut egui::Ui, state: &RelayState) {
        ui.heading("Relay");
        if !state.connected {
            ui.colored_label(ui.visuals().warn_fg_color, "The video-menu daemon isn't running.");
            return;
        }
        ui.add_space(8.0);

        egui::Grid::new("settings").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
            ui.label("Camera");
            let label = |name: &str| if name.is_empty() { "None".to_string() } else { name.to_string() };
            egui::ComboBox::from_id_salt("camera")
                .selected_text(label(&state.camera))
                .width(200.0)
                .show_ui(ui, |ui| {
                    let mut chosen = state.camera.clone();
                    ui.selectable_value(&mut chosen, String::new(), "None");
                    for (name, is_camera) in &state.sources {
                        let text = if *is_camera { name.clone() } else { format!("{name} (not a camera)") };
                        ui.selectable_value(&mut chosen, name.clone(), text);
                    }
                    if chosen != state.camera {
                        self.relay.send(Command::Camera(chosen));
                    }
                });
            ui.end_row();

            ui.label("Effect");
            ui.horizontal(|ui| {
                for (effect, text) in [(EffectChoice::GreenScreen, "Green screen"), (EffectChoice::Flip, "Flip")] {
                    if ui.radio(state.effect == Some(effect), text).clicked() {
                        self.relay.send(Command::Effect(effect));
                    }
                }
            });
            ui.end_row();

            let green = state.effect == Some(EffectChoice::GreenScreen);
            ui.label("Model");
            ui.add_enabled_ui(green, |ui| {
                ui.horizontal(|ui| {
                    for (model, text) in [(ModelChoice::Rvm, "RVM"), (ModelChoice::Selfie, "Selfie")] {
                        if ui.radio(state.model == Some(model), text).clicked() {
                            self.relay.send(Command::Model(model));
                        }
                    }
                });
            });
            ui.end_row();

            ui.label("");
            ui.add_enabled_ui(green, |ui| {
                let mut reuse = state.reuse_matte;
                if ui
                    .checkbox(&mut reuse, "Reuse previous matte")
                    .on_hover_text("Composite each frame with the previous frame's matte: lower latency, edges lag by a frame")
                    .changed()
                {
                    self.relay.send(Command::ReuseMatte(reuse));
                }
            });
            ui.end_row();

            ui.label("Smoothing");
            let selfie = green && state.model == Some(ModelChoice::Selfie);
            ui.add_enabled_ui(selfie, |ui| {
                let mut value = self.smoothing_edit.unwrap_or(state.selfie_smoothing);
                let response = ui
                    .add(egui::Slider::new(&mut value, 0.0..=0.95).fixed_decimals(2))
                    .on_hover_text("How strongly the selfie model's mask is smoothed over time where the image is still");
                if response.dragged() {
                    self.smoothing_edit = Some(value);
                } else if response.drag_stopped() || response.changed() {
                    self.smoothing_edit = None;
                    self.relay.send(Command::SelfieSmoothing(value));
                }
            });
            ui.end_row();
        });

        if let Some(error) = &state.error {
            ui.add_space(8.0);
            ui.colored_label(ui.visuals().error_fg_color, error);
        }

        ui.add_space(16.0);
        ui.heading("Performance");
        let stats = &state.stats;
        if stats.fps == 0.0 {
            ui.label("Not streaming: nothing is reading from the virtual camera.");
        } else {
            egui::Grid::new("stats").num_columns(2).spacing([12.0, 4.0]).show(ui, |ui| {
                ui.label("Frame rate");
                ui.label(format!("{:.1} fps", stats.fps));
                ui.end_row();
                ui.label("Per frame");
                ui.label(format!("{:.2} ms", stats.ms_per_frame));
                ui.end_row();
                for (stage, ms) in &stats.stages {
                    ui.label(format!("  {stage}"));
                    ui.label(format!("{ms:.2} ms"));
                    ui.end_row();
                }
            });
        }
        if !stats.backend.is_empty() {
            ui.add_space(4.0);
            ui.weak(format!("Processing on the {}", stats.backend));
        }
    }

    fn preview_panel(&mut self, ui: &mut egui::Ui, connected: bool) {
        ui.horizontal(|ui| {
            let mut on = self.preview.is_some();
            let toggle = ui
                .add_enabled(connected, egui::Checkbox::new(&mut on, "Show preview"))
                .on_hover_text("Reading the virtual camera keeps the real camera on, like any app using it");
            if toggle.changed() {
                if on {
                    let ctx = ui.ctx().clone();
                    self.preview = Some(Preview::start(move || ctx.request_repaint()));
                } else if let Some(preview) = self.preview.take() {
                    preview.stop();
                    self.texture = None;
                }
            }
            if self.preview.is_some() && self.texture.is_some() {
                ui.weak(format!("{:.0} fps", self.frames.fps));
            }
        });
        ui.separator();

        let Some(preview) = &self.preview else {
            ui.centered_and_justified(|ui| ui.weak("Preview off"));
            return;
        };
        let (frame, status) = {
            let mut shared = preview.shared.lock().unwrap();
            (shared.frame.take(), shared.status.clone())
        };
        if let Some(frame) = frame {
            self.frames.tick();
            match &mut self.texture {
                Some(texture) => texture.set(frame, egui::TextureOptions::LINEAR),
                None => {
                    self.texture = Some(ui.ctx().load_texture("preview", frame, egui::TextureOptions::LINEAR));
                }
            }
        }
        match &self.texture {
            Some(texture) => {
                ui.centered_and_justified(|ui| {
                    ui.add(egui::Image::new(texture).fit_to_exact_size(ui.available_size()).maintain_aspect_ratio(true));
                });
            }
            None => {
                ui.centered_and_justified(|ui| ui.weak(status));
            }
        }
    }
}
