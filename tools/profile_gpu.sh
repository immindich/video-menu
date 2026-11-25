#!/usr/bin/env bash
# Profile the daemon's GPU pipeline on the live camera, once per effect.
#
# For each effect, runs a release build of the daemon with
# VIDEO_MENU_GPU_PROFILE set, streams FRAMES frames from the virtual camera
# (with the grab_frame example), records what else uses the GPU meanwhile
# (gpu_clients.py), and prints the report (gpu_profile_report.py). Files go
# to OUTDIR: EFFECT.jsonl (the timeline), EFFECT.log (the daemon's output),
# EFFECT.clients.txt and EFFECT.report.txt.
#
# usage: tools/profile_gpu.sh OUTDIR [EFFECT...]    (default: color blur)
#   FRAMES=600  frames to stream per effect (about 20 s at 30 fps)
#
# Needs the models (VIDEO_MENU_MODEL, VIDEO_MENU_SELFIE_MODEL: the devshell
# sets them) and no other video-menu daemon running. Nobody needs to be in
# front of the camera.
set -euo pipefail

out=${1:?usage: tools/profile_gpu.sh OUTDIR [EFFECT...]}
shift
effects=("$@")
[ ${#effects[@]} -gt 0 ] || effects=(color blur)
frames=${FRAMES:-600}
root=$(cd "$(dirname "$0")/.." && pwd)
tools=$root/tools

if busctl --user status io.github.immindich.VideoMenu >/dev/null 2>&1; then
    echo "a video-menu daemon is already running; stop it first" >&2
    exit 1
fi
: "${VIDEO_MENU_MODEL:?not set (run in the devshell)}"
mkdir -p "$out"
(cd "$root" && cargo build --release -q -p video-menu --bin video-menu --example grab_frame)
target=$(cd "$root" && cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')

for effect in "${effects[@]}"; do
    echo "== $effect: streaming $frames frames"
    VIDEO_MENU_EFFECT=$effect VIDEO_MENU_GPU_PROFILE=$out/$effect.jsonl \
        "$target/release/video-menu" >"$out/$effect.log" 2>&1 &
    daemon=$!
    for _ in $(seq 50); do
        grep -q "ready" "$out/$effect.log" && break
        kill -0 $daemon 2>/dev/null || { cat "$out/$effect.log" >&2; exit 1; }
        sleep 0.2
    done
    python3 "$tools/gpu_clients.py" >"$out/$effect.clients.txt" &
    clients=$!
    "$target/release/examples/grab_frame" "$out/$effect.yuv" video-menu.camera "$frames" >/dev/null 2>&1 || true
    kill -INT $clients
    wait $clients || true
    kill $daemon
    wait $daemon 2>/dev/null || true
    rm -f "$out/$effect.yuv"
    python3 "$tools/gpu_profile_report.py" "$out/$effect.jsonl" >"$out/$effect.report.txt"
    cat "$out/$effect.report.txt" "$out/$effect.clients.txt"
    grep "pipeline:" "$out/$effect.log" || true
done
