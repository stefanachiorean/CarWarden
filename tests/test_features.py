import numpy as np
import pandas as pd
import pytest

from data import COLUMNS
from features import FEATURES, make_windows, learn_id_stats, payload_changes


def make_frames(rows):
    """rows: (timestamp, can_id, first_byte, label). Other bytes are 0, dlc is 8."""
    data = []
    for t, can_id, first_byte, label in rows:
        data.append([t, can_id, 8, first_byte, 0, 0, 0, 0, 0, 0, 0, label])
    return pd.DataFrame(data, columns=COLUMNS)


def test_single_window_known_values():
    # one window of 1 second, IDs [1, 1, 2, 3]
    frames = make_frames([
        (0.0, 1, 10, 0),
        (0.1, 1, 11, 0),   # same ID, payload changed
        (0.2, 2, 5, 0),
        (0.3, 3, 7, 0),
    ])
    w = make_windows(frames, window_seconds=1.0)
    assert len(w) == 1
    row = w.iloc[0]
    assert row["n_msgs"] == 4
    assert row["n_unique_ids"] == 3
    assert row["id_entropy"] == pytest.approx(1.5)        # p = .5, .25, .25
    assert row["top_id_share"] == pytest.approx(0.5)
    assert row["iat_mean"] == pytest.approx(0.1)
    assert row["iat_std"] == pytest.approx(0.0, abs=1e-9)
    assert row["iat_min"] == pytest.approx(0.1)
    assert row["mean_dlc"] == 8
    assert row["payload_change_rate"] == pytest.approx(1.0)   # 1 pair, 1 change
    assert row["frac_unknown_id"] == 0.0
    assert not row["is_attack"]
    assert row["attack_type"] == "normal"


def test_payload_change_rate_zero_when_payload_constant():
    frames = make_frames([(0.0, 1, 10, 0), (0.1, 1, 10, 0), (0.2, 1, 10, 0)])
    w = make_windows(frames, window_seconds=1.0)
    assert w.iloc[0]["payload_change_rate"] == 0.0


def test_payload_changes_uses_exact_comparison_on_low_bits():
    # two payloads that differ only in the last byte must count as different
    frames = pd.DataFrame(
        [[0.0, 5, 8, 1, 2, 3, 4, 5, 6, 7, 8, 0],
         [0.1, 5, 8, 1, 2, 3, 4, 5, 6, 7, 9, 0]], columns=COLUMNS)
    has_prev, changed = payload_changes(frames)
    assert list(has_prev) == [False, True]
    assert list(changed) == [False, True]


def test_two_windows_and_attack_label():
    frames = make_frames([
        (0.0, 1, 0, 0), (0.5, 2, 0, 0),
        (1.2, 1, 0, 0), (1.4, 0, 0, 1),   # attack frame in window 1
    ])
    w = make_windows(frames, window_seconds=1.0, attack_type="DoS")
    assert list(w["window"]) == [0, 1]
    assert list(w["is_attack"]) == [False, True]
    assert list(w["attack_type"]) == ["normal", "DoS"]
    assert w.iloc[1]["attack_frac"] == pytest.approx(0.5)


def test_frac_unknown_id():
    frames = make_frames([(0.0, 1, 0, 0), (0.1, 2, 0, 0), (0.2, 99, 0, 1), (0.3, 99, 0, 1)])
    w = make_windows(frames, window_seconds=1.0, known_ids={1, 2})
    assert w.iloc[0]["frac_unknown_id"] == pytest.approx(0.5)


def test_single_frame_window_has_no_nan():
    frames = make_frames([(0.0, 1, 0, 0), (2.5, 1, 0, 0)])
    w = make_windows(frames, window_seconds=1.0)
    assert not w[FEATURES].isna().any().any()


def test_learn_id_stats():
    frames = make_frames([(0.0, 1, 0, 0), (0.1, 1, 0, 0), (0.2, 2, 0, 0), (0.3, 2, 0, 0)])
    known, shares = learn_id_stats(frames)
    assert known == {1, 2}
    assert shares[1] == pytest.approx(0.5)


def test_window_time_alignment_with_fixed_t0():
    frames = make_frames([(10.05, 1, 0, 0), (10.15, 1, 0, 0)])
    w = make_windows(frames, window_seconds=0.1, t0=10.0)
    assert list(w["window"]) == [0, 1]
    assert w.iloc[1]["start_time"] == pytest.approx(10.1)


def test_top_id_is_the_most_frequent_id_of_the_window():
    # window 0: ID 7 twice, ID 3 once -> 7. window 1: ID 5 and ID 2 tie once each -> smaller ID (2).
    frames = make_frames([(0.0, 7, 0, 0), (0.1, 3, 0, 0), (0.2, 7, 0, 0),
                          (1.1, 5, 0, 0), (1.2, 2, 0, 0)])
    w = make_windows(frames, window_seconds=1.0)
    assert list(w["top_id"]) == [7.0, 2.0]