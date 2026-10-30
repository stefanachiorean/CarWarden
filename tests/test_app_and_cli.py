import io
import os

import pandas as pd
import pytest
from rich.console import Console
from streamlit.testing.v1 import AppTest

from carwarden import main
from report import print_incident

APP = os.path.join(os.path.dirname(__file__), "..", "src", "app.py")


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory):
    base = tmp_path_factory.mktemp("cli")
    console = Console(file=io.StringIO(), width=140)
    code = main(["demo", "--duration", "60", "--synthetic-dir", str(base / "syn"),
                 "--out", str(base / "out"), "--model-dir", str(base / "models")], console=console)
    assert code == 0
    return base


def capture_output(argv):
    buffer = io.StringIO()
    code = main(argv, console=Console(file=buffer, width=140))
    return code, buffer.getvalue()


def test_report_prints_tables_and_incidents(demo_dir):
    code, text = capture_output(["report", "--out", str(demo_dir / "out"), "--top", "2"])
    assert code == 0
    assert "Detection on the test part" in text
    assert "Evasion test" in text
    assert "RISK" in text and "Why:" in text


def test_scan_flags_attack_file(demo_dir):
    dos_file = demo_dir / "syn" / "DoS_dataset.csv"
    code, text = capture_output(["scan", str(dos_file), "--model", str(demo_dir / "models" / "carwarden.joblib"),
                                 "--top", "1"])
    assert code == 0
    assert "flagged" in text
    assert "DoS" in text


def test_train_with_missing_files_fails_cleanly(tmp_path):
    code, text = capture_output(["train", "--data-dir", str(tmp_path / "nothing")])
    assert code == 1
    assert "Missing files" in text


def test_print_incident_handles_missing_confidence():
    buffer = io.StringIO()
    row = pd.Series({"level": "MEDIUM", "predicted_type": "unknown anomaly", "risk": 55.0,
                     "confidence": float("nan"), "t_rel": 3.2, "n_msgs": 120, "reasons": ""})
    print_incident(Console(file=buffer, width=100), row)
    assert "MEDIUM RISK" in buffer.getvalue()
    assert "confidence n/a" in buffer.getvalue()


def test_dashboard_renders_without_errors(demo_dir, monkeypatch):
    monkeypatch.setenv("CARWARDEN_OUT", str(demo_dir / "out"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "CarWarden"
    assert len(at.tabs) == 3
    assert len(at.metric) >= 3


def test_dashboard_shows_selected_incident(demo_dir, monkeypatch):
    monkeypatch.setenv("CARWARDEN_OUT", str(demo_dir / "out"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.selectbox[0].select("DoS").run()
    assert not at.exception
    banners = [e.value for e in at.error] + [e.value for e in at.warning]
    assert any("RISK" in b and "DoS" in b for b in banners)


def test_dashboard_without_results_shows_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CARWARDEN_OUT", str(tmp_path / "empty"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert len(at.error) == 1
    assert "carwarden.py train" in at.error[0].value
    
    
def test_report_prints_confusion_matrix_and_evasion_caveat(demo_dir):
    code, text = capture_output(["report", "--out", str(demo_dir / "out"), "--top", "0"])
    assert code == 0
    assert "confusion matrix" in text
    assert "Caveat" in text
