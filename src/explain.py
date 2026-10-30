"""Two-layer explanation, as in SentryNet.

Layer 1 (numeric): which window features are far outside normal traffic?
Layer 2 (categorical): which CAN IDs are new or suddenly over-represented?
"""
import numpy as np

from features import FEATURES

FEATURE_TEXT = {
    "n_msgs": "messages per window",
    "n_unique_ids": "distinct CAN IDs",
    "id_entropy": "ID entropy (bits)",
    "top_id_share": "share of the most frequent ID",
    "iat_mean": "mean gap between frames (s)",
    "iat_std": "spread of gaps between frames (s)",
    "iat_min": "smallest gap between frames (s)",
    "mean_dlc": "mean payload length",
    "payload_change_rate": "payload change rate for repeated IDs",
    "frac_unknown_id": "share of frames with an ID never seen in normal traffic",
}


def explain_numeric(detector, row, top_n=3, z_limit=3.0):
    """Features whose robust z-score is beyond z_limit, strongest first."""
    x = np.array([row[name] for name in FEATURES], dtype=float)
    z = (x - detector.median_) / detector.scale_
    order = np.argsort(-np.abs(z))
    reasons = []
    for i in order:
        if abs(z[i]) < z_limit or len(reasons) >= top_n:
            break
        name = FEATURES[i]
        direction = "higher" if z[i] > 0 else "lower"
        reasons.append(f"{FEATURE_TEXT[name]}: {x[i]:.4g} "
                       f"({direction} than normal ~{detector.median_[i]:.4g}, {z[i]:+.1f} robust sigma)")
    return reasons


def explain_ids(detector, frames_in_window, top_n=2, min_share=0.05, over_factor=3.0):
    """Unknown IDs and IDs that are far more frequent than in normal traffic."""
    if frames_in_window is None or len(frames_in_window) == 0:
        return []
    counts = frames_in_window["can_id"].value_counts()
    total = len(frames_in_window)
    unknown, over = [], []
    for can_id, count in counts.items():
        share = count / total
        if share < min_share:
            continue
        can_id = int(can_id)
        if can_id not in detector.known_ids_:
            unknown.append(f"CAN ID 0x{can_id:04x} never appears in normal traffic "
                           f"({share:.0%} of frames)")
        else:
            normal_share = detector.id_share_.get(can_id, 0.0)
            if normal_share > 0 and share / normal_share >= over_factor:
                over.append(f"CAN ID 0x{can_id:04x} makes up {share:.0%} of frames "
                            f"(normally {normal_share:.1%})")
    return (unknown + over)[:top_n]


def explain_window(detector, row, frames_in_window=None):
    """All reasons for one window: numeric layer first, then the ID layer."""
    return explain_numeric(detector, row) + explain_ids(detector, frames_in_window)
