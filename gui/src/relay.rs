//! Talking to the daemon over D-Bus, on a thread of its own so the UI never
//! waits for the bus.
//!
//! The thread polls the relay's properties twice a second and right after
//! every change it makes, which also copes with the daemon starting,
//! stopping or restarting: the next poll just finds it (or doesn't).

use std::{
    sync::{Arc, Mutex, mpsc},
    thread,
    time::Duration,
};

use video_menu_protocol::{EffectChoice, ModelChoice, RelayProxy, Stats};

const POLL_INTERVAL: Duration = Duration::from_millis(500);

/// What the GUI shows of the relay.
#[derive(Clone, Debug, Default)]
pub struct RelayState {
    /// Whether the daemon is reachable.
    pub connected: bool,
    pub camera: String,
    pub sources: Vec<(String, bool)>,
    pub model: Option<ModelChoice>,
    pub effect: Option<EffectChoice>,
    pub background_color: [u8; 3],
    pub blur_radius: f64,
    pub reuse_matte: bool,
    pub selfie_smoothing: f64,
    pub stats: Stats,
    /// The last failed change, to show the user.
    pub error: Option<String>,
}

/// A change the user asked for.
#[derive(Debug)]
pub enum Command {
    Camera(String),
    Model(ModelChoice),
    Effect(EffectChoice),
    BackgroundColor([u8; 3]),
    BlurRadius(f64),
    ReuseMatte(bool),
    SelfieSmoothing(f64),
}

/// The UI's handle on the D-Bus thread.
pub struct RelayClient {
    pub state: Arc<Mutex<RelayState>>,
    commands: mpsc::Sender<Command>,
}

impl RelayClient {
    /// Start the D-Bus thread. `repaint` is called whenever the state changes.
    pub fn start(repaint: impl Fn() + Send + 'static) -> Self {
        let state = Arc::new(Mutex::new(RelayState::default()));
        let (commands, commands_rx) = mpsc::channel();
        let shared = state.clone();
        thread::Builder::new()
            .name("dbus".into())
            .spawn(move || run(shared, commands_rx, repaint))
            .expect("starting the D-Bus thread");
        Self { state, commands }
    }

    pub fn send(&self, command: Command) {
        let _ = self.commands.send(command);
    }
}

fn run(state: Arc<Mutex<RelayState>>, commands: mpsc::Receiver<Command>, repaint: impl Fn()) {
    let mut proxy: Option<RelayProxy<'static>> = None;
    loop {
        // Wait for a command, or poll when none comes.
        let command = match commands.recv_timeout(POLL_INTERVAL) {
            Ok(command) => Some(command),
            Err(mpsc::RecvTimeoutError::Timeout) => None,
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        };
        if proxy.is_none() {
            proxy = zbus::block_on(connect()).ok();
        }
        let Some(relay) = &proxy else {
            update(&state, &repaint, |s| *s = RelayState::default());
            continue;
        };
        if let Some(command) = command {
            for command in coalesce(command, &commands) {
                let result = zbus::block_on(apply(relay, command));
                update(&state, &repaint, |s| s.error = result.err());
            }
        }
        match zbus::block_on(read(relay)) {
            Ok(mut fresh) => update(&state, &repaint, |s| {
                fresh.error = s.error.take();
                *s = fresh;
            }),
            Err(e) => {
                update(&state, &repaint, |s| *s = RelayState::default());
                // The proxy addresses the daemon by name, so it finds a
                // restarted daemon by itself; only a broken connection to the
                // bus needs a new one.
                if matches!(e, zbus::Error::InputOutput(_)) {
                    proxy = None;
                }
            }
        }
    }
}

/// `first` and any commands queued behind it, keeping only the last of each
/// kind: dragging a slider or colour picker sends a command per frame, and
/// only where it ends up matters.
fn coalesce(first: Command, queued: &mpsc::Receiver<Command>) -> Vec<Command> {
    let mut latest: Vec<Command> = Vec::new();
    for command in std::iter::once(first).chain(queued.try_iter()) {
        let kind = std::mem::discriminant(&command);
        latest.retain(|c| std::mem::discriminant(c) != kind);
        latest.push(command);
    }
    latest
}

/// Change the shared state with `f`, repainting if it changed.
fn update(state: &Mutex<RelayState>, repaint: &impl Fn(), f: impl FnOnce(&mut RelayState)) {
    let mut state = state.lock().unwrap();
    let before = format!("{state:?}");
    f(&mut state);
    if format!("{state:?}") != before {
        repaint();
    }
}

async fn connect() -> zbus::Result<RelayProxy<'static>> {
    let conn = zbus::Connection::session().await?;
    // Don't cache properties: the daemon can restart under us, and every
    // poll should see its current values.
    RelayProxy::builder(&conn)
        .cache_properties(zbus::proxy::CacheProperties::No)
        .build()
        .await
}

async fn read(relay: &RelayProxy<'_>) -> zbus::Result<RelayState> {
    Ok(RelayState {
        connected: true,
        camera: relay.camera().await?,
        sources: relay.sources().await?,
        model: relay.model().await?.parse().ok(),
        effect: relay.effect().await?.parse().ok(),
        background_color: relay.background_color().await?.into(),
        blur_radius: relay.blur_radius().await?,
        reuse_matte: relay.reuse_matte().await?,
        selfie_smoothing: relay.selfie_smoothing().await?,
        stats: relay.stats().await?,
        error: None,
    })
}

async fn apply(relay: &RelayProxy<'_>, command: Command) -> Result<(), String> {
    let result = match command {
        Command::Camera(name) => relay.set_camera(&name).await,
        Command::Model(model) => relay.set_model(model.as_str()).await,
        Command::Effect(effect) => relay.set_effect(effect.as_str()).await,
        Command::BackgroundColor(color) => relay.set_background_color(color.into()).await,
        Command::BlurRadius(radius) => relay.set_blur_radius(radius).await,
        Command::ReuseMatte(reuse) => relay.set_reuse_matte(reuse).await,
        Command::SelfieSmoothing(strength) => relay.set_selfie_smoothing(strength).await,
    };
    result.map_err(|e| match e {
        // Show the daemon's own message rather than zbus's wrapping of it.
        zbus::Error::MethodError(_, Some(message), _) => message,
        e => e.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesce_keeps_the_last_of_each_kind_in_order() {
        let (tx, rx) = mpsc::channel();
        for c in [Command::BlurRadius(2.0), Command::ReuseMatte(true), Command::BlurRadius(3.0)] {
            tx.send(c).unwrap();
        }
        let got = coalesce(Command::BlurRadius(1.0), &rx);
        assert!(
            matches!(got.as_slice(), [Command::ReuseMatte(true), Command::BlurRadius(r)] if *r == 3.0),
            "{got:?}"
        );
    }
}
