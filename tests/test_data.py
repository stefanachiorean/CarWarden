import pandas as pd
import pytest
from data import parse_line, load_data, load_frames


def test_parse_line_full_frame():
    row = parse_line("1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R")
    assert row == [1478198376.389427, 0x0316, 8,
                   0x05, 0x21, 0x68, 0x09, 0x21, 0x21, 0x00, 0x6f, 0]


def test_parse_line_short_frame_is_padded():
    row = parse_line("1478195721.905736,02b0,5,ff,7f,00,05,49,R")
    assert row[2] == 5
    assert row[3:11] == [0xff, 0x7f, 0x00, 0x05, 0x49, 0, 0, 0]


def test_parse_line_attack_flag_is_label_1():
    row = parse_line("1478198380.000000,0000,8,00,00,00,00,00,00,00,00,T")
    assert row[-1] == 1


def test_parse_line_wrong_field_count_raises():
    with pytest.raises(ValueError):
        parse_line("1478195721.905736,02b0,5,ff,7f,R")


def test_load_data_reads_rows_and_respects_max_rows(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text(
        "1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R\n"
        "1478195721.905736,02b0,5,ff,7f,00,05,49,R\n"
        "\n"
        "1478198380.000000,0000,8,00,00,00,00,00,00,00,00,T\n"
    )
    df = load_data(str(path))
    assert len(df) == 3
    assert list(df["label"]) == [0, 0, 1]
    assert len(load_data(str(path), max_rows=2)) == 2


SAMPLE = (
    "1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R\n"
    "1478195721.905736,02b0,5,ff,7f,00,05,49,R\n"
    "1478195721.908437,0002,8,00,00,00,00,00,01,07,15,R\n"
    "1478198380.000000,0000,8,00,00,00,00,00,00,00,00,T\n"
    "1478198380.000300,0000,0,T\n"
)


def test_load_frames_matches_load_data(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text(SAMPLE)
    slow = load_data(str(path))
    fast = load_frames(str(path), chunk_rows=2)   # tiny chunks to test chunking
    pd.testing.assert_frame_equal(slow, fast, check_dtype=False)


def test_load_frames_respects_max_rows(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text(SAMPLE)
    assert len(load_frames(str(path), max_rows=3, chunk_rows=2)) == 3


def test_load_frames_bad_flag_reports_line(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R\n"
                    "1478198376.389500,0316,8,05,21,68,09,21,21,00,6f,X\n")
    with pytest.raises(ValueError, match="line 2"):
        load_frames(str(path))
