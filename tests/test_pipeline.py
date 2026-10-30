import json
import os

import numpy as np
import pandas as pd
import pytest

from data import load_frames
from evasion import thin_attack_frames
from pipeline import run_pipeline
from synth import generate_dataset


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    base = tmp_path_factory.mktemp("run")
    paths = generate_dataset(str(base / "data"), duration=60.0, burst_seconds=3.0,
                             gap_seconds=9.0, offset=6.0)
    out = run_pipeline(paths, out_dir=str(base / "outputs"), model_dir=str(base / "models"),
                       n_estimators=60, log=lambda *a: None)
    out["base"] = base
    out["paths"] = paths
    return out


def test_output_files_exist(result):
    base = result["base"]
    for name in ["outputs/scored_windows.csv", "outputs/metrics.json",
                 "outputs/evasion.csv", "models/carwarden.joblib"]:
        assert os.path.exists(base / name)


def test_overall_detection_is_good_on_synthetic_data(result):
    row = result["detection"].set_index("capture").loc["ALL"]
    assert row["auc"] > 0.95
    assert row["recall"] > 0.8
    assert row["false_positive_rate"] < 0.1


def test_dos_is_detected_almost_perfectly(result):
    row = result["detection"].set_index("capture").loc["DoS"]
    assert row["recall"] > 0.95


def test_flagged_windows_have_type_and_reasons(result):
    scored = result["scored"]
    flagged = scored[scored["flagged"]]
    assert len(flagged) > 0
    assert (flagged["predicted_type"] != "normal").all()
    assert (flagged["reasons"].str.len() > 0).mean() > 0.8
    assert flagged["confidence"].between(0, 1).all()
    assert scored[~scored["flagged"]]["predicted_type"].eq("normal").all()


def test_training_never_sees_attack_windows(result):
    metrics = result["metrics"]
    assert metrics["n_fit_normal_windows"] < metrics["n_train_windows"]


def test_train_and_test_are_separated_in_time(result):
    for capture, part in result["scored"].groupby("capture"):
        assert part["t_rel"].min() >= result["cuts"][capture] - 1e-6


def test_every_capture_has_attack_windows_in_the_test_part(result):
    assert (result["scored"].groupby("capture")["is_attack"].sum() > 0).all()


def test_evasion_recall_drops_when_attack_gets_thinner(result):
    evasion = result["evasion"]
    dos = evasion[evasion["attack_type"] == "DoS"].set_index("keep_fraction")
    assert dos.loc[1.0, "recall"] >= dos.loc[0.02, "recall"]
    assert dos.loc[1.0, "recall"] > 0.9


def test_metrics_json_is_valid(result):
    with open(result["base"] / "outputs" / "metrics.json") as f:
        metrics = json.load(f)
    assert "detection" in metrics and "confusion_matrix" in metrics


def test_thin_attack_frames_keeps_normal_and_reduces_attacks(result):
    frames = load_frames(result["paths"]["DoS"])
    thinned = thin_attack_frames(frames, 0.25, seed=1)
    assert (thinned["label"] == 0).sum() == (frames["label"] == 0).sum()
    ratio = (thinned["label"] == 1).sum() / (frames["label"] == 1).sum()
    assert 0.2 < ratio < 0.3
    assert thin_attack_frames(frames, 1.0)["label"].sum() == frames["label"].sum()
    assert thin_attack_frames(frames, 0.0)["label"].sum() == 0
