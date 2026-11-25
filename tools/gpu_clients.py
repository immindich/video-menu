"""Record which processes use the GPU, and how hard, while something runs.

Samples the xe driver's per-client counters (/proc/PID/fdinfo: busy cycles
per engine) and the GPU's actual clock every INTERVAL seconds until
interrupted (SIGINT/SIGTERM), then prints each process's average and peak
share of each engine, and the clock's range. Only processes of the current
user are visible, which covers desktop applications.

usage: python gpu_clients.py [INTERVAL]   (default 0.25)
"""
import os
import signal
import sys
import time

FREQ = "/sys/class/drm/card0/device/tile0/gt0/freq0/act_freq"
ENGINES = {"rcs": "render", "ccs": "compute", "bcs": "copy", "vcs": "video", "vecs": "video-enh"}


def sample():
    """{client_id: (pid, comm, {engine: (cycles, total_cycles)})} for xe clients."""
    clients = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            fds = os.listdir(f"/proc/{pid}/fdinfo")
        except OSError:
            continue
        comm = None
        for fd in fds:
            try:
                with open(f"/proc/{pid}/fdinfo/{fd}") as f:
                    text = f.read()
            except OSError:
                continue
            if "drm-driver:\txe" not in text:
                continue
            fields = dict(line.split(":\t", 1) for line in text.splitlines() if ":\t" in line)
            client = fields.get("drm-client-id")
            if client is None or client in clients:
                continue  # the same client through another fd
            if comm is None:
                try:
                    with open(f"/proc/{pid}/comm") as f:
                        comm = f.read().strip()
                except OSError:
                    comm = "?"
            engines = {}
            for key in ENGINES:
                if f"drm-cycles-{key}" in fields:
                    engines[key] = (int(fields[f"drm-cycles-{key}"]), int(fields[f"drm-total-cycles-{key}"]))
            clients[client] = (int(pid), comm, engines)
    return clients


def freq():
    try:
        with open(FREQ) as f:
            return int(f.read())
    except OSError:
        return None


def main():
    interval = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.append(1))
    busy = {}  # (pid, comm, engine) -> list of busy fractions
    freqs = []
    prev = sample()
    start = time.monotonic()
    while not stop:
        time.sleep(interval)
        cur = sample()
        f = freq()
        if f is not None:
            freqs.append(f)
        for client, (pid, comm, engines) in cur.items():
            if client not in prev:
                continue
            for key, (cycles, total) in engines.items():
                before = prev[client][2].get(key)
                if before is None or total <= before[1]:
                    continue
                share = (cycles - before[0]) / (total - before[1])
                busy.setdefault((pid, comm, key), []).append(share)
        prev = cur

    elapsed = time.monotonic() - start
    print(f"GPU clients over {elapsed:.1f} s ({len(freqs)} samples every {interval} s):")
    # Sum clients per process; show engines anyone used.
    rows = []
    for (pid, comm, key), shares in busy.items():
        mean = sum(shares) / len(shares)
        if max(shares) < 0.005:
            continue
        rows.append((mean, comm, pid, ENGINES[key], max(shares)))
    for mean, comm, pid, engine, peak in sorted(rows, reverse=True):
        print(f"  {comm:<20} {pid:>8}  {engine:<9} mean {100 * mean:5.1f}%  peak {100 * peak:5.1f}%")
    if not rows:
        print("  (no GPU use seen)")
    if freqs:
        freqs.sort()
        print(
            f"GPU clock: min {freqs[0]} MHz, median {freqs[len(freqs) // 2]} MHz, max {freqs[-1]} MHz"
        )


if __name__ == "__main__":
    main()
