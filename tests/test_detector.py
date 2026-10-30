import numpy as np
import pandas as pd
import pytest

from classifier import AttackClassifier
from detector import Detector, risk_level, ALERT_RISK
from explain import explain_numeric, explain_ids, explain_window
from features import CLASSIFIER_FEATURES, make_windows, learn_id_stats
from synth import generate_rows
from data import COLUMNS


def synthetic_windows(attack_type, seed):
    rows = generate_rows(attack_type, duration=60.0, seed=seed, burst_seconds=3.0, gap_seconds=9.0, offset=6.0)
    frames = pd.DataFrame([[t, i, d] + list(data) + [0] * (8 - len(data)) + [1 if f == "T" else 0]
                           for t, i, d, data, f in rows], columns=COLUMNS)
    return frames


@pytest.fixture(scope="module")
def trained():
    normal_frames = synthetic_windows(None, seed=1)
    known, share = learn_id_stats(normal_frames)
    normal = make_windows(normal_frames, 0.1, known)
    n_fit = int(len(normal) * 0.8)
    det = Detector(n_estimators=100).fit(normal.iloc[:n_fit], normal.iloc[n_fit:], known, share)
    return det, known, share


def test_risk_range_and_levels(trained):
    det, known, _ = trained
    attack = make_windows(synthetic_windows("DoS", seed=2), 0.1, known, "DoS")
    risk = det.risk(attack)
    assert risk.min() >= 0 and risk.max() <= 100
    assert risk_level(10) == "NORMAL"
    assert risk_level(60) == "MEDIUM"
    assert risk_level(90) == "HIGH"


def test_strong_attack_reaches_high_risk(trained):
    det, known, _ = trained
    attack = make_windows(synthetic_windows("DoS", seed=2), 0.1, known, "DoS")
    risk = det.risk(attack)
    assert np.median(risk[attack["is_attack"].to_numpy()]) >= 75


def test_normal_windows_are_mostly_not_flagged(trained):
    det, known, _ = trained
    normal = make_windows(synthetic_windows(None, seed=3), 0.1, known)
    assert det.predict(normal).mean() < 0.10


def test_dos_windows_are_flagged(trained):
    det, known, _ = trained
    attack = make_windows(synthetic_windows("DoS", seed=2), 0.1, known, "DoS")
    flagged = det.predict(attack)
    assert flagged[attack["is_attack"].to_numpy()].mean() > 0.9


def test_attack_risk_higher_than_normal_risk(trained):
    det, known, _ = trained
    attack = make_windows(synthetic_windows("Fuzzy", seed=4), 0.1, known, "Fuzzy")
    risk = det.risk(attack)
    is_attack = attack["is_attack"].to_numpy()
    assert risk[is_attack].mean() > risk[~is_attack].mean() + 20


def test_explain_numeric_for_dos_mentions_message_rate(trained):
    det, known, _ = trained
    frames = synthetic_windows("DoS", seed=2)
    windows = make_windows(frames, 0.1, known, "DoS")
    row = windows[windows["is_attack"]].iloc[2]
    reasons = explain_numeric(det, row)
    assert len(reasons) > 0
    assert any("messages per window" in r or "unknown" in r for r in reasons)


def test_explain_ids_finds_unknown_id(trained):
    det, known, _ = trained
    frames = synthetic_windows("DoS", seed=2)
    attack_frames = frames[frames["label"] == 1].iloc[:300]
    reasons = explain_ids(det, attack_frames)
    assert any("0x0000" in r and "never appears" in r for r in reasons)


def test_explain_ids_empty_input(trained):
    det, _, _ = trained
    assert explain_ids(det, None) == []


def test_classifier_names_attack_types():
    parts = []
    known = learn_id_stats(synthetic_windows(None, seed=1))[0]
    for name in ["DoS", "Fuzzy", "Gear spoofing", "RPM spoofing"]:
        parts.append(make_windows(synthetic_windows(name, seed=7), 0.1, known, name))
    parts.append(make_windows(synthetic_windows(None, seed=8), 0.1, known, "normal"))
    train = pd.concat(parts, ignore_index=True)
    clf = AttackClassifier(n_estimators=50).fit(train)

    test = make_windows(synthetic_windows("DoS", seed=9), 0.1, known, "DoS")
    labels, conf = clf.predict(test)
    dos = test["is_attack"].to_numpy()
    assert (labels[dos] == "DoS").mean() > 0.9
    assert conf.min() >= 0 and conf.max() <= 1
    assert set(clf.feature_importance()) == set(CLASSIFIER_FEATURES)