"""Phase 2: attack our own detector.

Evasion attack used here: the attacker lowers the injection rate ("low and slow")
so that the injected traffic hides inside normal traffic. We delete a random part
of the injected frames from the test capture and measure how recall drops.

This is a stealth-vs-effectiveness trade-off: fewer injected frames are harder to
detect, but they also do less damage to the vehicle (a DoS that only sends a few
frames no longer blocks the bus). The table shows where the detector breaks.
"""
import numpy as np
import pandas as pd

from detector import ALERT_RISK
from features import make_windows


def thin_attack_frames(frames, keep_fraction, seed=0):
    """Keep all normal frames and a random keep_fraction of the injected frames."""
    rng = np.random.default_rng(seed)
    is_attack = frames["label"].to_numpy() == 1
    keep = (~is_attack) | (rng.random(len(frames)) < keep_fraction)
    return frames[keep].reset_index(drop=True)


def evasion_sweep(detector, frames, window_seconds, attack_type, t0,
                  keep_fractions=(1.0, 0.5, 0.25, 0.1, 0.05, 0.02), seed=0):
    """Recall of the detector on the original attack windows as the attack gets thinner."""
    known_ids = detector.known_ids_
    original = make_windows(frames, window_seconds, known_ids, attack_type, t0=t0)
    attack_ids = set(original.loc[original["is_attack"], "window"])
    attack_seconds = len(attack_ids) * window_seconds
    n_attack_frames = int((frames["label"] == 1).sum())

    rows = []
    for keep in keep_fractions:
        thinned = thin_attack_frames(frames, keep, seed)
        windows = make_windows(thinned, window_seconds, known_ids, attack_type, t0=t0)
        selected = windows["window"].isin(attack_ids).to_numpy()
        risk = detector.risk(windows)
        flagged = risk >= ALERT_RISK
        rows.append({
            "attack_type": attack_type,
            "keep_fraction": keep,
            "injected_frames_per_second": (n_attack_frames * keep / attack_seconds) if attack_seconds else 0.0,
            "recall": float(flagged[selected].mean()) if selected.any() else float("nan"),
            "mean_risk_on_attack_windows": float(risk[selected].mean()) if selected.any() else float("nan"),
        })
    return pd.DataFrame(rows)
