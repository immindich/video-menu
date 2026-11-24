//! The daemon's D-Bus interface (see [`video_menu_protocol`] for the names).
//!
//! zbus runs on a thread of its own ([`start`]), while the relay lives on
//! PipeWire's main loop and is single-threaded. They meet in two places:
//!
//! - Requests from D-Bus (switch camera, change a setting) are sent to the
//!   main loop over a [`pipewire::channel`] as [`Call`]s. The main loop
//!   handles them and replies over an async channel, so a slow request (a
//!   model's first compile takes seconds) never blocks the bus thread.
//! - Clients read properties from a [`Snapshot`] the main loop keeps
//!   current through [`Service::refresh`], which also tells the bus thread
//!   which properties changed so it can signal them.

use std::sync::{Arc, Mutex};

use pipewire as pw;
use video_menu_protocol::{BUS_NAME, EffectChoice, ModelChoice, OBJECT_PATH, Stats};
use zbus::{
    connection,
    fdo::{self, RequestNameFlags, RequestNameReply},
    object_server::InterfaceRef,
};

/// A setting a D-Bus client asked the relay to change.
#[derive(Debug)]
pub enum Request {
    /// Relay from the named source, or from none if empty.
    Camera(String),
    Model(ModelChoice),
    Effect(EffectChoice),
    ReuseMatte(bool),
    SelfieSmoothing(f32),
}

/// A [`Request`] and where to send its result.
pub struct Call {
    pub request: Request,
    reply: async_channel::Sender<Result<(), String>>,
}

impl Call {
    /// Report the outcome to the waiting client.
    pub fn reply(self, result: Result<(), String>) {
        let _ = self.reply.try_send(result);
    }
}

/// Everything clients can read, as of the last [`Service::refresh`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Snapshot {
    pub camera: String,
    pub sources: Vec<(String, bool)>,
    pub model: ModelChoice,
    pub effect: EffectChoice,
    pub reuse_matte: bool,
    pub selfie_smoothing: f64,
    pub stats: Stats,
}

/// Which properties changed, for the bus thread to signal.
#[derive(Clone, Copy, Debug)]
enum Change {
    Camera,
    Sources,
    Model,
    Effect,
    ReuseMatte,
    SelfieSmoothing,
    Stats,
}

/// The main loop's handle on the D-Bus service.
pub struct Service {
    snapshot: Arc<Mutex<Snapshot>>,
    changes: async_channel::Sender<Change>,
}

impl Service {
    /// Make `new` what clients see, signalling the properties that differ.
    pub fn refresh(&self, new: Snapshot) {
        let mut snapshot = self.snapshot.lock().unwrap();
        let old = std::mem::replace(&mut *snapshot, new);
        let new = &*snapshot;
        let changed = [
            (old.camera != new.camera, Change::Camera),
            (old.sources != new.sources, Change::Sources),
            (old.model != new.model, Change::Model),
            (old.effect != new.effect, Change::Effect),
            (old.reuse_matte != new.reuse_matte, Change::ReuseMatte),
            (old.selfie_smoothing != new.selfie_smoothing, Change::SelfieSmoothing),
            (old.stats != new.stats, Change::Stats),
        ];
        for (differs, change) in changed {
            if differs {
                let _ = self.changes.try_send(change);
            }
        }
    }
}

/// Connect to the session bus, serve the interface and claim [`BUS_NAME`] on
/// a thread of its own. Fails if the name is taken, i.e. another daemon is
/// running. `calls` delivers client requests to the main loop.
pub fn start(calls: pw::channel::Sender<Call>, initial: Snapshot) -> Result<Service, String> {
    let snapshot = Arc::new(Mutex::new(initial));
    let (changes, changes_rx) = async_channel::unbounded();
    let (ready, ready_rx) = std::sync::mpsc::channel();
    let interface = RelayInterface { calls, snapshot: snapshot.clone() };
    std::thread::Builder::new()
        .name("dbus".into())
        .spawn(move || zbus::block_on(serve(interface, changes_rx, ready)))
        .map_err(|e| format!("starting the D-Bus thread: {e}"))?;
    ready_rx
        .recv()
        .map_err(|_| "the D-Bus thread stopped".to_string())??;
    Ok(Service { snapshot, changes })
}

/// The bus thread: set up, report whether that worked, then signal changes
/// for as long as the daemon runs.
async fn serve(
    interface: RelayInterface,
    changes: async_channel::Receiver<Change>,
    ready: std::sync::mpsc::Sender<Result<(), String>>,
) {
    let setup = async {
        let conn = connection::Builder::session()
            .serve_at(OBJECT_PATH, interface)
            .build()
            .await?;
        // Request the name by hand: zbus's default flags include
        // REPLACE_EXISTING and ALLOW_REPLACEMENT, which would let a second
        // daemon take the name over. With only DO_NOT_QUEUE it fails instead.
        let reply = conn
            .request_name_with_flags(BUS_NAME, RequestNameFlags::DO_NOT_QUEUE)
            .await?;
        Ok::<_, zbus::Error>((conn, reply))
    };
    let conn = match setup.await {
        Ok((conn, RequestNameReply::PrimaryOwner | RequestNameReply::AlreadyOwner)) => conn,
        Ok(_) => {
            let _ = ready.send(Err(format!("video-menu is already running ({BUS_NAME} is taken)")));
            return;
        }
        Err(zbus::Error::NameTaken) => {
            let _ = ready.send(Err(format!("video-menu is already running ({BUS_NAME} is taken)")));
            return;
        }
        Err(e) => {
            let _ = ready.send(Err(format!("D-Bus: {e}")));
            return;
        }
    };
    let iface: InterfaceRef<RelayInterface> = match conn.object_server().interface(OBJECT_PATH).await {
        Ok(iface) => iface,
        Err(e) => {
            let _ = ready.send(Err(format!("D-Bus: {e}")));
            return;
        }
    };
    let _ = ready.send(Ok(()));

    while let Ok(change) = changes.recv().await {
        let relay = iface.get().await;
        let emitter = iface.signal_emitter();
        let result = match change {
            Change::Camera => relay.camera_changed(emitter).await,
            Change::Sources => relay.sources_changed(emitter).await,
            Change::Model => relay.model_changed(emitter).await,
            Change::Effect => relay.effect_changed(emitter).await,
            Change::ReuseMatte => relay.reuse_matte_changed(emitter).await,
            Change::SelfieSmoothing => relay.selfie_smoothing_changed(emitter).await,
            Change::Stats => relay.stats_changed(emitter).await,
        };
        if let Err(e) = result {
            eprintln!("dbus: signalling {change:?}: {e}");
        }
    }
}

struct RelayInterface {
    calls: pw::channel::Sender<Call>,
    snapshot: Arc<Mutex<Snapshot>>,
}

impl RelayInterface {
    /// Have the main loop carry out `request` and wait for the result.
    async fn call(&self, request: Request) -> fdo::Result<()> {
        let (reply, result) = async_channel::bounded(1);
        self.calls
            .send(Call { request, reply })
            .map_err(|_| fdo::Error::Failed("the relay has stopped".into()))?;
        result
            .recv()
            .await
            .map_err(|_| fdo::Error::Failed("the relay dropped the request".into()))?
            .map_err(fdo::Error::InvalidArgs)
    }

    fn read<T>(&self, f: impl FnOnce(&Snapshot) -> T) -> T {
        f(&self.snapshot.lock().unwrap())
    }
}

#[zbus::interface(name = "io.github.immindich.VideoMenu.Relay1")]
impl RelayInterface {
    /// Relay from the named source, or from none if `name` is empty.
    async fn set_camera(&self, name: String) -> fdo::Result<()> {
        self.call(Request::Camera(name)).await
    }

    /// PipeWire node name of the camera being relayed, or empty for none.
    #[zbus(property)]
    async fn camera(&self) -> String {
        self.read(|s| s.camera.clone())
    }

    /// Every video source the relay could use: (node name, has the Camera
    /// role).
    #[zbus(property)]
    async fn sources(&self) -> Vec<(String, bool)> {
        self.read(|s| s.sources.clone())
    }

    /// Matting model: "rvm" or "selfie".
    #[zbus(property)]
    async fn model(&self) -> String {
        self.read(|s| s.model.to_string())
    }

    #[zbus(property)]
    async fn set_model(&mut self, value: String) -> fdo::Result<()> {
        let model = value.parse().map_err(fdo::Error::InvalidArgs)?;
        self.call(Request::Model(model)).await
    }

    /// Effect: "green" or "flip".
    #[zbus(property)]
    async fn effect(&self) -> String {
        self.read(|s| s.effect.to_string())
    }

    #[zbus(property)]
    async fn set_effect(&mut self, value: String) -> fdo::Result<()> {
        let effect = value.parse().map_err(fdo::Error::InvalidArgs)?;
        self.call(Request::Effect(effect)).await
    }

    /// Composite each frame with the previous frame's matte.
    #[zbus(property)]
    async fn reuse_matte(&self) -> bool {
        self.read(|s| s.reuse_matte)
    }

    #[zbus(property)]
    async fn set_reuse_matte(&mut self, value: bool) -> fdo::Result<()> {
        self.call(Request::ReuseMatte(value)).await
    }

    /// Temporal smoothing of the selfie model's mask, 0..1.
    #[zbus(property)]
    async fn selfie_smoothing(&self) -> f64 {
        self.read(|s| s.selfie_smoothing)
    }

    #[zbus(property)]
    async fn set_selfie_smoothing(&mut self, value: f64) -> fdo::Result<()> {
        self.call(Request::SelfieSmoothing(value as f32)).await
    }

    /// Processing statistics over the last second or so.
    #[zbus(property)]
    async fn stats(&self) -> Stats {
        self.read(|s| s.stats.clone())
    }
}
