//! Interactive commands read from stdin, for switching cameras by hand.
//!
//! - `list`: show the video sources (`*` marks the one being relayed)
//! - `use <name>`: relay from the named source
//! - `none`: stop relaying from any camera

use std::{
    io::{BufRead, IsTerminal, Stdin},
    rc::Rc,
};

use pipewire::{
    loop_::IoSource,
    main_loop::MainLoopRc,
    spa::support::system::IoFlags,
};

use crate::relay::CameraRelay;

const USAGE: &str = "commands: list | use <name> | none";

/// Start reading commands from stdin, if it is a terminal. Commands are
/// handled for as long as the returned source is kept. EOF quits the loop.
pub fn attach(mainloop: &MainLoopRc, relay: Rc<CameraRelay>) -> Option<IoSource<'_, Stdin>> {
    let stdin = std::io::stdin();
    if !stdin.is_terminal() {
        return None;
    }
    eprintln!("{USAGE}");
    let quit = mainloop.clone();
    Some(mainloop.loop_().add_io(stdin, IoFlags::IN, move |stdin| {
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => quit.quit(),
            Ok(_) => run(&relay, line.trim()),
            Err(e) => eprintln!("stdin: {e}"),
        }
    }))
}

fn run(relay: &CameraRelay, line: &str) {
    let (cmd, arg) = line
        .split_once(char::is_whitespace)
        .map_or((line, ""), |(c, a)| (c, a.trim()));
    let result = match (cmd, arg) {
        ("", _) => Ok(()),
        ("list", "") => {
            let current = relay.camera();
            let sources = relay.sources().list();
            if sources.is_empty() {
                eprintln!("  (no video sources)");
            }
            for (name, is_camera) in sources {
                let mark = if current.as_deref() == Some(&name) { '*' } else { ' ' };
                let role = if is_camera { " (camera)" } else { "" };
                eprintln!("{mark} {name}{role}");
            }
            Ok(())
        }
        ("use", name) if !name.is_empty() => relay.set_camera(Some(name)),
        ("none", "") => relay.set_camera(None),
        _ => {
            eprintln!("{USAGE}");
            Ok(())
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
    }
}
