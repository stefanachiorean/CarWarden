import numpy as np
import pandas as pd

from data import BYTE_COLUMNS

FEATURES = [
    "n_msgs",               # frames in the window
    "n_unique_ids",         # distinct CAN IDs
    "id_entropy",           # Shannon entropy of the ID distribution (bits)
    "top_id_share",         # share of the most frequent ID
    "iat_mean",             # mean gap between consecutive frames (s)
    "iat_std",              # std of the gaps
    "iat_min",              # smallest gap
    "mean_dlc",             # mean payload length
    "payload_change_rate",  # how often a repeated ID changes its payload
    "frac_unknown_id",      # share of frames whose ID never appeared in normal training traffic
]

# The classifier (attack TYPE) also gets the ID that dominates the window. The anomaly detector
# does not use it, so detection is unchanged. Spoofing attacks inject one specific ID, so this is
# what tells Gear spoofing from RPM spoofing. It is specific to the vehicle/dataset.
CLASSIFIER_FEATURES = FEATURES + ["top_id"]


def payload_keys(frames):
    """Pack the 8 data bytes of every frame into one uint64 number."""
    key = np.zeros(len(frames), dtype=np.uint64)
    for name in BYTE_COLUMNS:
        key = (key << np.uint64(8)) | frames[name].to_numpy().astype(np.uint64)
    return key


def payload_changes(frames):
    """For each frame: did the payload differ from the previous frame with the same ID?

    Returns two boolean arrays: has_previous and changed.
    (Uses exact integer comparison, so no float rounding problems.)
    """
    ids = frames["can_id"].to_numpy()
    keys = payload_keys(frames)
    n = len(frames)

    order = np.argsort(ids, kind="stable")   # groups the same ID, keeps time order
    ids_sorted = ids[order]
    keys_sorted = keys[order]

    same_id = ids_sorted[1:] == ids_sorted[:-1]
    different_payload = keys_sorted[1:] != keys_sorted[:-1]

    has_previous = np.zeros(n, dtype=bool)
    changed = np.zeros(n, dtype=bool)
    has_previous[order[1:]] = same_id
    changed[order[1:]] = same_id & different_payload
    return has_previous, changed


def learn_id_stats(frames):
    """From normal frames: the set of known IDs and the share of each ID."""
    counts = frames["can_id"].value_counts()
    known_ids = set(int(i) for i in counts.index)
    shares = {int(i): float(c) / len(frames) for i, c in counts.items()}
    return known_ids, shares


def make_windows(frames, window_seconds=0.1, known_ids=None, attack_type="attack", t0=None):
    """Cut a capture into fixed time windows and compute features per window.

    frames:      DataFrame from load_data / load_frames (sorted by time)
    known_ids:   IDs seen in normal training traffic (for frac_unknown_id); None -> 0
    attack_type: name given to windows that contain at least one injected (T) frame
    t0:          start time of window 0 (default: first timestamp)
    """
    df = frames.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if t0 is None:
        t0 = float(df["timestamp"].iloc[0])

    df["window"] = np.floor((df["timestamp"].to_numpy() - t0) / window_seconds).astype(np.int64)

    # gaps between consecutive frames, restarted at the first frame of each window
    df["iat"] = df["timestamp"].diff()
    first_in_window = df["window"] != df["window"].shift(1)
    df.loc[first_in_window, "iat"] = np.nan

    has_previous, changed = payload_changes(df)
    df["has_previous"] = has_previous
    df["changed"] = changed

    if known_ids is None:
        df["unknown"] = 0.0
    else:
        df["unknown"] = (~df["can_id"].isin(list(known_ids))).astype(float)

    grouped = df.groupby("window")
    out = pd.DataFrame({
        "n_msgs": grouped.size(),
        "iat_mean": grouped["iat"].mean(),
        "iat_std": grouped["iat"].std(),
        "iat_min": grouped["iat"].min(),
        "mean_dlc": grouped["dlc"].mean(),
        "frac_unknown_id": grouped["unknown"].mean(),
        "attack_frac": grouped["label"].mean(),
    })

    # ID distribution per window
    counts = df.groupby(["window", "can_id"]).size()
    totals = counts.groupby(level=0).transform("sum")
    p = counts / totals
    out["n_unique_ids"] = counts.groupby(level=0).size()
    out["id_entropy"] = (-(p * np.log2(p))).groupby(level=0).sum()
    out["top_id_share"] = p.groupby(level=0).max()
    by_count = counts.reset_index(name="n").sort_values(["window", "n", "can_id"],
                                                        ascending=[True, False, True])
    out["top_id"] = by_count.drop_duplicates("window").set_index("window")["can_id"].astype(float)

    # payload change rate = changes / pairs that could be compared
    pairs = grouped["has_previous"].sum()
    changes = grouped["changed"].sum()
    out["payload_change_rate"] = (changes / pairs.where(pairs > 0)).fillna(0.0)

    # windows with 1 frame have no gaps: use the window length / 0 spread
    out["iat_mean"] = out["iat_mean"].fillna(window_seconds)
    out["iat_min"] = out["iat_min"].fillna(window_seconds)
    out["iat_std"] = out["iat_std"].fillna(0.0)

    out = out.reset_index()
    out["start_time"] = t0 + out["window"] * window_seconds
    out["is_attack"] = out["attack_frac"] > 0
    out["attack_type"] = np.where(out["is_attack"], attack_type, "normal")

    columns = ["window", "start_time"] + CLASSIFIER_FEATURES + ["attack_frac", "is_attack", "attack_type"]
    out = out[columns].copy()
    out["id_entropy"] = out["id_entropy"].abs()   # avoid -0.0
    return out