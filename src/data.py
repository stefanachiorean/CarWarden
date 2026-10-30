import numpy as np
import pandas as pd

COLUMNS = ["timestamp", "can_id", "dlc",
           "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "label"]
BYTE_COLUMNS = ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]


def parse_line(line):
    """Turn one CSV line into [timestamp, can_id, dlc, b0..b7, label]."""
    parts = line.strip().split(",")
    dlc = int(parts[2])
    if dlc < 0 or dlc > 8:
        raise ValueError(f"Invalid DLC {dlc}")
    if len(parts) != 4 + dlc:
        raise ValueError(f"Expected {4 + dlc} fields, got {len(parts)}")

    timestamp = float(parts[0])
    can_id = int(parts[1], 16)
    data = [int(b, 16) for b in parts[3:3 + dlc]]
    data = data + [0] * (8 - dlc)  # pad short frames; dlc column keeps the real length

    flag = parts[-1]
    if flag == "R":
        label = 0
    elif flag == "T":
        label = 1
    else:
        raise ValueError(f"Unknown flag {flag!r}")

    return [timestamp, can_id, dlc] + data + [label]


def load_data(filepath, max_rows=None):
    """Load a Car-Hacking CSV into a DataFrame. Stops early if max_rows is set."""
    rows = []
    with open(filepath, "r") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip() == "":
                continue
            try:
                rows.append(parse_line(line))
            except ValueError as e:
                raise ValueError(f"{filepath}, line {line_number}: {e}")
            if max_rows is not None and len(rows) >= max_rows:
                break
    return pd.DataFrame(rows, columns=COLUMNS)


# ---------------------------------------------------------------------------
# Fast loader for the real files (3-4 million rows each).
# It gives the same result as load_data(), but works on whole chunks with
# pandas instead of one Python call per line.
# ---------------------------------------------------------------------------

HEX_BYTE = {f"{i:02x}": i for i in range(256)}
HEX_BYTE.update({f"{i:02X}": i for i in range(256)})


def _convert_chunk(raw, first_line_number, filepath):
    """Convert one chunk of raw string columns (0..11) into the final columns."""
    n = len(raw)

    dlc = raw[2].astype(int).to_numpy()
    if (dlc < 0).any() or (dlc > 8).any():
        bad = int(np.argmax((dlc < 0) | (dlc > 8)))
        raise ValueError(f"{filepath}, line {first_line_number + bad}: invalid DLC {dlc[bad]}")

    # The flag (R or T) sits right after the data bytes, so its column is 3 + dlc.
    values = raw.to_numpy()
    flags = values[np.arange(n), 3 + dlc]
    is_r = flags == "R"
    is_t = flags == "T"
    if not (is_r | is_t).all():
        bad = int(np.argmax(~(is_r | is_t)))
        raise ValueError(f"{filepath}, line {first_line_number + bad}: unknown flag {flags[bad]!r}")

    out = pd.DataFrame()
    out["timestamp"] = raw[0].astype(float).to_numpy()

    id_text = raw[1]
    id_lookup = {text: int(text, 16) for text in id_text.unique()}
    out["can_id"] = id_text.map(id_lookup).to_numpy().astype(np.int32)
    out["dlc"] = dlc.astype(np.int8)

    for k, name in enumerate(BYTE_COLUMNS):
        mapped = raw[3 + k].map(HEX_BYTE)
        missing = mapped.isna().to_numpy()
        # Cells at or after position dlc are padding (the flag or nothing) -> 0.
        bad_cells = missing & (k < dlc)
        if bad_cells.any():
            bad = int(np.argmax(bad_cells))
            raise ValueError(f"{filepath}, line {first_line_number + bad}: bad data byte")
        out[name] = mapped.fillna(0).to_numpy().astype(np.uint8)

    out["label"] = is_t.astype(np.int8)
    return out[COLUMNS]


def load_frames(filepath, max_rows=None, chunk_rows=500_000):
    """Fast version of load_data(). Same columns, much smaller memory/time."""
    pieces = []
    rows_read = 0
    reader = pd.read_csv(filepath, header=None, names=list(range(12)), dtype=str,
                         chunksize=chunk_rows, skip_blank_lines=True)
    for raw in reader:
        if max_rows is not None and rows_read + len(raw) > max_rows:
            raw = raw.iloc[: max_rows - rows_read]
        pieces.append(_convert_chunk(raw, rows_read + 1, filepath))
        rows_read += len(raw)
        if max_rows is not None and rows_read >= max_rows:
            break
    if not pieces:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(pieces, ignore_index=True)
