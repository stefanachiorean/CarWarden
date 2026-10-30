import numpy as np
import pandas as pd
import pytest

from capture_info import capture_summary
from pipeline import choose_cut, run_pipeline, trim_after_gap
from synth import generate_dataset, generate_rows, write_csv


def timestamps_and_labels(times, attack_mask):
    return pd.DataFrame({"timestamp": times, "label": attack_mask.astype(int)})


def test_choose_cut_follows_the_attacks_not_the_wall_clock():
    times = np.arange(0, 100, 0.01)
    frames = timestamps_and_labels(times, times < 20)      # attacks only in the first 20 s
    cut = choose_cut(frames, 0.6)
    assert cut == pytest.approx(12.0, abs=0.1)              # 60% of the attack frames, not 60 s


def test_choose_cut_without_attacks_uses_duration():
    times = np.arange(0, 100, 0.01)
    frames = timestamps_and_labels(times, np.zeros(len(times), dtype=bool))
    assert choose_cut(frames, 0.6) == pytest.approx(60.0, abs=0.1)


def test_capture_summary_finds_bursts_gaps_and_tail():
    # frames every 0.1 s; a 5 s hole at t=30; attacks in two bursts (10-12 s and 20-22 s)
    times = np.concatenate([np.arange(0, 30, 0.1), np.arange(35, 60, 0.1)])
    attack = ((times >= 10) & (times < 12)) | ((times >= 20) & (times < 22))
    summary = capture_summary(timestamps_and_labels(times, attack))
    assert summary["first_attack_s"] == pytest.approx(10.0, abs=0.01)
    assert summary["last_attack_s"] == pytest.approx(21.9, abs=0.01)
    assert summary["n_bursts"] == 2
    assert summary["duration_s"] == pytest.approx(59.9, abs=0.01)
    start, length = summary["largest_gaps"][0]
    assert start == pytest.approx(29.9, abs=0.01)
    assert length == pytest.approx(5.1, abs=0.01)


def test_capture_summary_without_attacks():
    times = np.arange(0, 10, 0.1)
    summary = capture_summary(timestamps_and_labels(times, np.zeros(len(times), dtype=bool)))
    assert summary["first_attack_s"] is None
    assert summary["n_bursts"] == 0


def test_pipeline_handles_attacks_that_stop_early(tmp_path):
    # 100 s captures where the attacks stop after 30 s: the old 60%-of-duration cut
    # would have put every attack into the training part.
    paths = generate_dataset(str(tmp_path / "early"), duration=100.0, burst_seconds=3.0,
                             gap_seconds=5.0, offset=5.0)
    for name, path in paths.items():
        rows = [r for r in generate_rows(name, duration=100.0, seed=3, burst_seconds=3.0,
                                         gap_seconds=5.0, offset=5.0)
                if not (r[4] == "T" and r[0] - 1478190000.0 > 30.0)]
        write_csv(rows, path)
    result = run_pipeline(paths, out_dir=str(tmp_path / "out"), model_dir=str(tmp_path / "models"),
                          n_estimators=40, run_evasion=False, log=lambda *a: None)
    attack_windows = result["scored"].groupby("capture")["is_attack"].sum()
    assert (attack_windows > 0).all()
    for capture, part in result["scored"].groupby("capture"):
        assert part["t_rel"].min() >= result["cuts"][capture] - 1e-6
        assert result["cuts"][capture] < 30.0


def test_pipeline_refuses_a_capture_without_attacks(tmp_path):
    normal_path = tmp_path / "normal_only.csv"
    write_csv(generate_rows(None, duration=30.0, seed=1), str(normal_path))
    with pytest.raises(ValueError, match="attack windows"):
        run_pipeline({"DoS": str(normal_path)}, out_dir=str(tmp_path / "out"),
                     model_dir=str(tmp_path / "models"), n_estimators=20,
                     run_evasion=False, log=lambda *a: None)


def test_trim_after_gap_cuts_at_the_first_big_gap():
    times = np.concatenate([np.arange(0, 30, 0.1), np.arange(1030, 1060, 0.1)])
    frames = timestamps_and_labels(times, np.zeros(len(times), dtype=bool))
    trimmed = trim_after_gap(frames, 10.0)
    assert trimmed["timestamp"].max() < 30.0
    assert len(trimmed) == 300


def test_trim_after_gap_keeps_everything_without_a_big_gap():
    times = np.arange(0, 30, 0.1)
    frames = timestamps_and_labels(times, np.zeros(len(times), dtype=bool))
    assert len(trim_after_gap(frames, 10.0)) == len(frames)


def test_pipeline_with_trim_gap_ignores_the_segment_after_the_gap(tmp_path):
    # 100 s captures with a 1000 s hole at 70 s; everything after the hole must be ignored
    paths = generate_dataset(str(tmp_path / "gap"), duration=100.0, burst_seconds=3.0,
                             gap_seconds=5.0, offset=5.0)
    for name, path in paths.items():
        rows = generate_rows(name, duration=100.0, seed=3, burst_seconds=3.0,
                             gap_seconds=5.0, offset=5.0)
        shifted = [((r[0] + 1000.0,) + r[1:]) if r[0] - 1478190000.0 > 70.0 else r for r in rows]
        write_csv(shifted, path)
    result = run_pipeline(paths, out_dir=str(tmp_path / "out"), model_dir=str(tmp_path / "models"),
                          n_estimators=40, run_evasion=False, trim_gap=10.0, log=lambda *a: None)
    assert result["scored"]["t_rel"].max() < 71.0
    assert result["metrics"]["trim_gap"] == 10.0


def test_classifier_separates_gear_from_rpm_spoofing(tmp_path):
    paths = generate_dataset(str(tmp_path / "gr"), duration=100.0)
    result = run_pipeline(paths, out_dir=str(tmp_path / "out"), model_dir=str(tmp_path / "models"),
                          n_estimators=40, run_evasion=False, log=lambda *a: None)
    confusion = result["confusion"]
    gear_row = confusion.loc["true Gear spoofing"]
    rpm_row = confusion.loc["true RPM spoofing"]
    assert gear_row["pred Gear spoofing"] > 5 * gear_row["pred RPM spoofing"]
    assert rpm_row["pred RPM spoofing"] > 5 * rpm_row["pred Gear spoofing"]