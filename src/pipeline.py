"""End to end: load captures -> windows -> train -> evaluate -> evasion -> save."""
import json
import os

import joblib
import numpy as np
import pandas as pd

from classifier import AttackClassifier
from data import load_frames
from detector import Detector, ALERT_RISK, risk_level
from evaluate import confusion_table, detection_table
from evasion import evasion_sweep
from explain import explain_window, explain_numeric
from features import FEATURES, learn_id_stats, make_windows

DEFAULT_FILES = {
    "DoS": "DoS_dataset.csv",
    "Fuzzy": "Fuzzy_dataset.csv",
    "Gear spoofing": "gear_dataset.csv",
    "RPM spoofing": "RPM_dataset.csv",
}


def choose_cut(frames, train_frac):
    """Time that separates the training part from the test part of one capture.

    The cut is placed so that train_frac of the INJECTED frames come before it.
    (A cut at 60% of the wall-clock duration is wrong when the attacks do not
    spread over the whole file: all attacks can end up in the training part.)
    A capture without injected frames is cut at train_frac of its duration.
    """
    timestamps = frames["timestamp"].to_numpy()
    attack_times = timestamps[frames["label"].to_numpy() == 1]
    if len(attack_times) == 0:
        return float(timestamps[0] + train_frac * (timestamps[-1] - timestamps[0]))
    return float(np.quantile(attack_times, train_frac))


def trim_after_gap(frames, min_gap):
    """Drop everything after the first gap longer than min_gap seconds between two frames.

    The HCRL attack captures contain one huge timestamp jump (22 s to over 2 hours). What comes
    after it is a different, attack-free segment (see the ROAD dataset paper). frames must be
    sorted by timestamp.
    """
    timestamps = frames["timestamp"].to_numpy()
    big_gaps = np.where(np.diff(timestamps) > min_gap)[0]
    if len(big_gaps) == 0:
        return frames
    return frames.iloc[: big_gaps[0] + 1].reset_index(drop=True)


def frames_in_window(frames, timestamps, start_time, window_seconds):
    """The frames of one window (frames sorted by time, timestamps = their time column)."""
    lo = np.searchsorted(timestamps, start_time, side="left")
    hi = np.searchsorted(timestamps, start_time + window_seconds, side="left")
    return frames.iloc[lo:hi]


def add_predictions(detector, classifier, windows, frames_by_capture, window_seconds,
                    max_explained=2000):
    """Risk, level, attack type, confidence and reasons for every window."""
    windows = windows.copy().reset_index(drop=True)
    windows["risk"] = detector.risk(windows)
    windows["level"] = [risk_level(r) for r in windows["risk"]]
    windows["flagged"] = windows["risk"] >= ALERT_RISK

    labels, confidence = classifier.predict(windows)
    predicted = np.where(labels == "normal", "unknown anomaly", labels)
    windows["predicted_type"] = np.where(windows["flagged"], predicted, "normal")
    windows["confidence"] = np.where(windows["flagged"], confidence, np.nan)

    reasons = [""] * len(windows)
    for capture, part in windows[windows["flagged"]].groupby("capture"):
        frames = frames_by_capture[capture]
        timestamps = frames["timestamp"].to_numpy()
        top = part.sort_values("risk", ascending=False).head(max_explained)
        for i, row in top.iterrows():
            in_window = frames_in_window(frames, timestamps, row["start_time"], window_seconds)
            reasons[i] = " | ".join(explain_window(detector, row, in_window))
        for i in part.index.difference(top.index):
            reasons[i] = " | ".join(explain_numeric(detector, windows.loc[i]))
    windows["reasons"] = reasons
    return windows


def run_pipeline(files, out_dir="outputs", model_dir="models", window_seconds=0.1,
                 train_frac=0.6, max_rows=None, seed=42, run_evasion=True,
                 n_estimators=200, trim_gap=None, log=print):
    """files: {attack name: csv path}. Time split (see choose_cut): everything before the cut
    trains, everything after tests. The detector sees NORMAL windows of the training part only."""
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 1. load captures and split them in time
    captures = {}
    for name, path in files.items():
        log(f"loading {name}: {path}")
        frames = load_frames(path, max_rows=max_rows).sort_values("timestamp").reset_index(drop=True)
        if trim_gap is not None:
            n_before = len(frames)
            frames = trim_after_gap(frames, trim_gap)
            if len(frames) < n_before:
                log(f"  trimmed {n_before - len(frames)} frames after a gap longer than {trim_gap} s")
        t0 = float(frames["timestamp"].iloc[0])
        cut = choose_cut(frames, train_frac)
        captures[name] = {"frames": frames, "t0": t0, "cut": cut}
        log(f"  {len(frames)} frames, {(frames['label'] == 1).mean():.1%} injected")

    # 2. which IDs appear in normal training traffic
    normal_train = pd.concat([
        c["frames"][(c["frames"]["timestamp"] < c["cut"]) & (c["frames"]["label"] == 0)]
        for c in captures.values()
    ])
    known_ids, id_share = learn_id_stats(normal_train)

    # 3. windows with features, split into train / test by time
    train_parts, test_parts = [], []
    frames_test = {}
    for name, c in captures.items():
        windows = make_windows(c["frames"], window_seconds, known_ids, attack_type=name, t0=c["t0"])
        windows["capture"] = name
        windows["t_rel"] = windows["start_time"] - c["t0"]
        train_part = windows[windows["start_time"] < c["cut"]]
        test_part = windows[windows["start_time"] >= c["cut"]]
        n_train_attack = int(train_part["is_attack"].sum())
        n_test_attack = int(test_part["is_attack"].sum())
        log(f"  {name}: cut at {c['cut'] - c['t0']:.0f} s; attack windows: "
            f"train {n_train_attack}, test {n_test_attack}")
        if n_train_attack == 0 or n_test_attack == 0:
            raise ValueError(
                f"{name}: train has {n_train_attack} and test has {n_test_attack} attack windows, "
                f"but both need some. Inspect the file with: python src/capture_info.py <file>")
        train_parts.append(train_part)
        test_parts.append(test_part)
        frames_test[name] = c["frames"][c["frames"]["timestamp"] >= c["cut"]].reset_index(drop=True)
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    log(f"windows: train {len(train)}, test {len(test)}")

    # 4. detector: fit on normal windows (first 80% of each capture), calibrate on the last 20%
    fit_parts, calib_parts = [], []
    for part in train_parts:
        normal = part[~part["is_attack"]]
        n_fit = int(len(normal) * 0.8)
        fit_parts.append(normal.iloc[:n_fit])
        calib_parts.append(normal.iloc[n_fit:])
    fit_windows = pd.concat(fit_parts, ignore_index=True)
    calib_windows = pd.concat(calib_parts, ignore_index=True)
    detector = Detector(n_estimators=n_estimators, random_state=seed)
    detector.fit(fit_windows, calib_windows, known_ids, id_share)
    log(f"detector trained on {len(fit_windows)} normal windows, calibrated on {len(calib_windows)}")

    # 5. attack-type classifier on all training windows
    classifier = AttackClassifier(random_state=seed).fit(train)

    # 6. evaluate on the test part
    scored = add_predictions(detector, classifier, test, frames_test, window_seconds)
    table = detection_table(scored, scored["risk"].to_numpy(), scored["flagged"].to_numpy())
    labels = ["normal"] + list(files.keys())
    raw_labels, _ = classifier.predict(scored)
    confusion = confusion_table(scored["attack_type"], raw_labels, labels)
    accuracy = float((raw_labels == scored["attack_type"].to_numpy()).mean())

    # 7. evasion test (Phase 2)
    evasion = pd.DataFrame()
    if run_evasion:
        parts = []
        for name, c in captures.items():
            log(f"evasion sweep: {name}")
            parts.append(evasion_sweep(detector, frames_test[name], window_seconds, name, c["t0"]))
        evasion = pd.concat(parts, ignore_index=True)

    # 8. save everything the CLI and the dashboard need
    scored.to_csv(os.path.join(out_dir, "scored_windows.csv"), index=False)
    evasion.to_csv(os.path.join(out_dir, "evasion.csv"), index=False)
    metrics = {
        "window_seconds": window_seconds,
        "trim_gap": trim_gap,
        "train_frac": train_frac,
        "n_train_windows": int(len(train)),
        "n_test_windows": int(len(test)),
        "n_fit_normal_windows": int(len(fit_windows)),
        "alert_threshold_risk": ALERT_RISK,
        "classifier_accuracy": accuracy,
        "detection": table.to_dict(orient="records"),
        "confusion_matrix": json.loads(confusion.to_json(orient="split")),
        "feature_importance": classifier.feature_importance(),
        "features": FEATURES,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump({"detector": detector, "classifier": classifier,
                 "window_seconds": window_seconds},
                os.path.join(model_dir, "carwarden.joblib"))
    log(f"saved results to {out_dir}/ and model to {model_dir}/")
    cuts = {name: c["cut"] - c["t0"] for name, c in captures.items()}
    return {"metrics": metrics, "detection": table, "confusion": confusion,
            "evasion": evasion, "scored": scored, "cuts": cuts}