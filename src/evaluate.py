import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def detection_table(test_windows, risk, flagged):
    """Precision / recall / F1 / false-positive rate per capture and overall.

    test_windows needs the columns: capture, is_attack
    Each capture contains one attack type plus normal traffic.
    """
    rows = []
    groups = list(test_windows.groupby("capture", sort=False)) + [("ALL", test_windows)]
    for name, part in groups:
        idx = part.index.to_numpy()
        y_true = part["is_attack"].to_numpy().astype(int)
        y_pred = np.asarray(flagged)[idx].astype(int)
        scores = np.asarray(risk)[idx]
        normal = y_true == 0
        auc = roc_auc_score(y_true, scores) if len(set(y_true)) == 2 else float("nan")
        rows.append({
            "capture": name,
            "windows": len(part),
            "attack_windows": int(y_true.sum()),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "false_positive_rate": float(y_pred[normal].mean()) if normal.any() else float("nan"),
            "auc": auc,
        })
    return pd.DataFrame(rows)


def confusion_table(y_true, y_pred, labels):
    """Confusion matrix as a DataFrame (rows = true, columns = predicted)."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=[f"true {l}" for l in labels],
                        columns=[f"pred {l}" for l in labels])
