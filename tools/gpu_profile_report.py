"""Summarise a GPU timeline recorded with VIDEO_MENU_GPU_PROFILE.

For each frame the daemon records CPU events (marks), the GPU's start and
end time for every compute pass, and the GPU clock. This prints, as min /
median / 95th percentile / mean over the frames (the first SKIP are left out
as warm-up):

- where the CPU's time goes, step by step through the frame;
- for each submission: how long the CPU waited for it, how long the GPU was
  actually busy with it, and the difference (the overhead that is neither
  shader time nor copies), split into the delay before the GPU started and
  the delay after it finished;
- time per pass on the GPU;
- the GPU clock (0 when the GPU is asleep), and whether slow frames follow
  a long idle gap or a low clock.

The GPU's clock and the CPU's have an unknown offset. Start delays are shown
relative to the quickest start seen in the file (so the true values are
higher by a constant) and end delays are correspondingly upper bounds; their
sum, the total overhead, is exact.

usage: python gpu_profile_report.py FILE.jsonl [SKIP]   (default SKIP 30)
"""
import json
import statistics
import sys


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(p / 100 * len(values)))]


def row(name, values, unit=1e6, width=34):
    """A line of min / p50 / p95 / mean, in ms by default."""
    if not values:
        return f"  {name:<{width}} (none)"
    v = [x / unit for x in values]
    return (
        f"  {name:<{width}} min {min(v):6.2f}  p50 {pct(v, 50):6.2f}  "
        f"p95 {pct(v, 95):6.2f}  mean {statistics.fmean(v):6.2f}"
    )


def main():
    path = sys.argv[1]
    skip = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    frames = [json.loads(line) for line in open(path) if line.strip()]
    frames = frames[skip:]
    if not frames:
        sys.exit(f"{path}: no frames after skipping {skip}")
    labels = sorted({f["label"] for f in frames})
    print(f"{path}: {len(frames)} frames ({', '.join(labels)})")

    # CPU steps: the time between consecutive marks.
    steps = {}
    totals = []
    for f in frames:
        marks = f["marks"]
        for (a, ta), (b, tb) in zip(marks, marks[1:]):
            steps.setdefault(f"{a} -> {b}", []).append(tb - ta)
        totals.append(marks[-1][1] - marks[0][1])
    print("\nCPU, step by step (ms):")
    order = []
    for f in frames:
        for (a, _), (b, _) in zip(f["marks"], f["marks"][1:]):
            if f"{a} -> {b}" not in order:
                order.append(f"{a} -> {b}")
    for step in order:
        print(row(step, steps[step], width=40))
    print(row("whole frame (GPU backend)", totals, width=40))

    # Submissions: CPU wait vs GPU busy.
    raw_start = {}  # submission -> [gpu_begin - cpu_submit]
    per_sub = {}
    for f in frames:
        marks = dict(f["marks"])
        for i, sub in enumerate(f["submits"]):
            passes = [p for p in f["passes"] if p[1] == i]
            if not passes:
                continue
            begin = min(p[2] for p in passes)
            end = max(p[3] for p in passes)
            busy = end - begin
            work = sum(p[3] - p[2] for p in passes)
            d = per_sub.setdefault(sub, {"busy": [], "work": [], "wait": [], "overhead": [], "start": [], "end": []})
            d["busy"].append(busy)
            d["work"].append(work)
            raw_start.setdefault(sub, []).append(begin - marks[sub])
            d["start"].append(begin - marks[sub])
            waited = marks.get(f"{sub} waited")
            if waited is not None:
                d["wait"].append(waited - marks[sub])
                d["overhead"].append(waited - marks[sub] - busy)
                d["end"].append(waited - end)
    offset = min(min(v) for v in raw_start.values())
    print("\nSubmissions (ms):")
    for sub, d in per_sub.items():
        print(f" {sub}:")
        if d["wait"]:
            print(row("CPU waited (submit -> wait returned)", d["wait"]))
        print(row("GPU busy (first pass start -> last end)", d["busy"]))
        print(row("  of which passes running", d["work"]))
        if d["overhead"]:
            print(row("overhead (waited - busy)", d["overhead"]))
        print(row("start delay (above quickest seen)", [s - offset for s in d["start"]]))
        if d["end"]:
            print(row("end delay (upper bound)", [e + offset for e in d["end"]]))

    # Gaps on the GPU between submissions of a frame.
    gaps = {}
    for f in frames:
        spans = []
        for i, sub in enumerate(f["submits"]):
            passes = [p for p in f["passes"] if p[1] == i]
            if passes:
                spans.append((sub, min(p[2] for p in passes), max(p[3] for p in passes)))
        for (a, _, end_a), (b, begin_b, _) in zip(spans, spans[1:]):
            gaps.setdefault(f"GPU idle {a} -> {b}", []).append(begin_b - end_a)
    if gaps:
        print("\nOn the GPU between submissions (ms):")
        for name, values in gaps.items():
            print(row(name, values, width=40))

    passes = {}
    for f in frames:
        for name, _, b, e in f["passes"]:
            passes.setdefault(name, []).append(e - b)
    print("\nGPU time per pass (ms):")
    for name, values in passes.items():
        print(row(name, values))

    # Clock, and what slow frames have in common.
    start_freq = [f["freq_mhz"][0] for f in frames if f["freq_mhz"][0] is not None]
    end_freq = [f["freq_mhz"][1] for f in frames if f["freq_mhz"][1] is not None]
    print("\nGPU clock (MHz; 0 means the GPU was asleep, power-gated):")
    print(row("at frame start", start_freq, unit=1))
    print(row("at frame end", end_freq, unit=1))
    asleep = sum(1 for fr in start_freq if fr == 0)
    print(f"  asleep at {asleep} of {len(start_freq)} frame starts")

    idle = []  # CPU time from the previous frame's end to this one's start
    for prev, f in zip(frames, frames[1:]):
        idle.append(f["marks"][0][1] - prev["marks"][-1][1])
    print(row("\n  idle between frames (ms)", idle, width=33))

    overhead = []
    for f in frames:
        marks = dict(f["marks"])
        total = 0
        for i, sub in enumerate(f["submits"]):
            passes = [p for p in f["passes"] if p[1] == i]
            if passes and f"{sub} waited" in marks:
                total += marks[f"{sub} waited"] - marks[sub] - (max(p[3] for p in passes) - min(p[2] for p in passes))
        overhead.append(total)
    if len(frames) > 10:
        pairs = list(zip(idle, overhead[1:]))
        print("\nDoes the overhead follow long idle gaps or a low clock? (median overhead, ms)")
        cut = statistics.median(i for i, _ in pairs)
        short = [o for i, o in pairs if i <= cut]
        long_ = [o for i, o in pairs if i > cut]
        print(f"  after idle <= {cut / 1e6:.1f} ms: {statistics.median(short) / 1e6:.2f}   "
              f"after longer idle: {statistics.median(long_) / 1e6:.2f}")
        with_freq = [(f["freq_mhz"][0], o) for f, o in zip(frames, overhead) if f["freq_mhz"][0] is not None]
        if with_freq:
            fcut = statistics.median(fr for fr, _ in with_freq)
            low = [o for fr, o in with_freq if fr < fcut]
            high = [o for fr, o in with_freq if fr >= fcut]
            if low and high:
                print(f"  clock < {fcut} MHz at start: {statistics.median(low) / 1e6:.2f}   "
                      f">= {fcut} MHz: {statistics.median(high) / 1e6:.2f}")
            else:
                print(f"  clock was {fcut} MHz at every frame start")


if __name__ == "__main__":
    main()
