"""Synthetic CAN traffic in the same text format as the HCRL Car-Hacking files.

This is ONLY for testing the code and for the demo. Results on synthetic data
say nothing about performance on real vehicles. The attacks are deliberately
simple (they mimic the four HCRL attack types), so detection is much easier
here than on the real files.
"""
import os

import numpy as np

# (can_id, period in seconds, dlc, has_counter_in_byte0)
NORMAL_IDS = [
    (0x0316, 0.010, 8, True),
    (0x018F, 0.010, 8, False),
    (0x0260, 0.010, 8, True),
    (0x02A0, 0.010, 8, False),
    (0x0329, 0.010, 8, True),
    (0x0545, 0.010, 8, False),
    (0x0130, 0.010, 8, True),
    (0x0131, 0.010, 8, False),
    (0x043F, 0.010, 8, False),
    (0x0140, 0.020, 8, True),
    (0x0153, 0.020, 8, False),
    (0x02B0, 0.020, 5, False),   # short frame (DLC 5), like in the real data
    (0x02C0, 0.100, 8, False),
    (0x0350, 0.100, 8, True),
    (0x0370, 0.100, 8, False),
]

ATTACK_TYPES = ["DoS", "Fuzzy", "Gear spoofing", "RPM spoofing"]

FILE_NAMES = {
    "DoS": "DoS_dataset.csv",
    "Fuzzy": "Fuzzy_dataset.csv",
    "Gear spoofing": "gear_dataset.csv",
    "RPM spoofing": "RPM_dataset.csv",
}


def _normal_frames(duration, rng):
    """Periodic traffic. Returns a list of (t, can_id, dlc, data, flag)."""
    rows = []
    for can_id, period, dlc, has_counter in NORMAL_IDS:
        base = rng.integers(0, 256, size=8)
        times = np.arange(0.0, duration, period) + rng.normal(0, period * 0.02, size=int(np.ceil(duration / period)))
        times = np.sort(np.clip(times, 0, duration))
        data = base.copy()
        for k, t in enumerate(times):
            if has_counter:
                data[0] = k % 256
            elif rng.random() < 0.1:
                data[1] = rng.integers(0, 256)
            rows.append((t, can_id, dlc, [int(b) for b in data[:dlc]], "R"))
    return rows


def _attack_frames(attack_type, duration, rng, burst_seconds, gap_seconds, offset):
    """Injected frames, sent in bursts like in the real captures."""
    rows = []
    start = offset
    spoof_payload = [int(b) for b in rng.integers(0, 256, size=8)]
    while start < duration:
        end = min(start + burst_seconds, duration)
        if attack_type == "DoS":
            times = np.arange(start, end, 0.0003)
            for t in times:
                rows.append((t, 0x0000, 8, [0] * 8, "T"))
        elif attack_type == "Fuzzy":
            times = np.arange(start, end, 0.0005)
            for t in times:
                rows.append((t, int(rng.integers(0, 0x800)), 8,
                             [int(b) for b in rng.integers(0, 256, size=8)], "T"))
        elif attack_type == "Gear spoofing":
            for t in np.arange(start, end, 0.001):
                rows.append((t, 0x043F, 8, spoof_payload, "T"))
        elif attack_type == "RPM spoofing":
            for t in np.arange(start, end, 0.001):
                rows.append((t, 0x0316, 8, spoof_payload, "T"))
        else:
            raise ValueError(f"Unknown attack type {attack_type!r}")
        start += burst_seconds + gap_seconds
    return rows


def generate_rows(attack_type, duration=100.0, seed=0, burst_seconds=4.0,
                  gap_seconds=16.0, offset=10.0, t_start=1478190000.0):
    rng = np.random.default_rng(seed)
    rows = _normal_frames(duration, rng)
    if attack_type is not None:
        rows += _attack_frames(attack_type, duration, rng, burst_seconds, gap_seconds, offset)
    rows.sort(key=lambda r: r[0])
    return [(r[0] + t_start,) + r[1:] for r in rows]


def write_csv(rows, path):
    """Write rows in the exact format of the HCRL files (no header, flag last)."""
    lines = []
    for t, can_id, dlc, data, flag in rows:
        data_text = ",".join(f"{b:02x}" for b in data)
        lines.append(f"{t:.6f},{can_id:04x},{dlc},{data_text},{flag}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def generate_dataset(out_dir, duration=100.0, seed=0, burst_seconds=4.0,
                     gap_seconds=16.0, offset=10.0):
    """Write four synthetic capture files. Returns {attack name: path}."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for i, attack_type in enumerate(ATTACK_TYPES):
        rows = generate_rows(attack_type, duration, seed + i, burst_seconds, gap_seconds, offset)
        path = os.path.join(out_dir, FILE_NAMES[attack_type])
        write_csv(rows, path)
        paths[attack_type] = path
    return paths
