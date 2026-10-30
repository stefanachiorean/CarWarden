"""Summary of one or more capture files: where are the attacks, are there timing gaps?

  python src/capture_info.py data/Fuzzy_dataset.csv data/gear_dataset.csv
"""
import sys

import numpy as np

from data import load_frames


def capture_summary(frames, burst_gap=1.0, n_gaps=3):
    """Facts about one capture. Times are seconds from the first frame."""
    ordered = frames.sort_values("timestamp", kind="stable")
    t = ordered["timestamp"].to_numpy()
    t0 = float(t[0])
    summary = {
        "rows": int(len(ordered)),
        "duration_s": float(t[-1] - t0),
        "injected_frames": int(ordered["label"].sum()),
        "first_attack_s": None,
        "last_attack_s": None,
        "n_bursts": 0,
    }
    attack_t = t[ordered["label"].to_numpy() == 1]
    if len(attack_t) > 0:
        summary["first_attack_s"] = float(attack_t[0] - t0)
        summary["last_attack_s"] = float(attack_t[-1] - t0)
        summary["n_bursts"] = 1 + int((np.diff(attack_t) > burst_gap).sum())

    gaps = np.diff(t)
    largest = np.argsort(-gaps)[:n_gaps]
    summary["largest_gaps"] = [(float(t[i] - t0), float(gaps[i])) for i in largest]
    return summary


def print_summary(path, summary):
    print(path)
    print(f"  rows: {summary['rows']}, duration: {summary['duration_s']:.1f} s")
    print(f"  injected frames: {summary['injected_frames']}")
    if summary["first_attack_s"] is None:
        print("  no injected frames")
    else:
        print(f"  attacks from {summary['first_attack_s']:.1f} s to {summary['last_attack_s']:.1f} s "
              f"in {summary['n_bursts']} bursts")
        tail = summary["duration_s"] - summary["last_attack_s"]
        print(f"  time after the last attack: {tail:.1f} s")
    print("  largest gaps between consecutive frames (starts at, length):")
    for start, length in summary["largest_gaps"]:
        print(f"    {start:.1f} s, {length:.3f} s")


if __name__ == "__main__":
    for file_path in sys.argv[1:]:
        print_summary(file_path, capture_summary(load_frames(file_path)))
