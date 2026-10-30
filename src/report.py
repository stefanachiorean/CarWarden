"""Rich terminal output (same idea as the SentryNet report)."""
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

LEVEL_STYLE = {"HIGH": "bold red", "MEDIUM": "bold yellow", "NORMAL": "green"}


def make_console():
    return Console()


def print_incident(console, row):
    """One flagged window as a panel: level, attack type, confidence, reasons."""
    level = row["level"]
    style = LEVEL_STYLE.get(level, "white")
    confidence = row["confidence"]
    confidence_text = f"{confidence:.0%}" if pd.notna(confidence) else "n/a"

    title = f"{level} RISK - {row['predicted_type']} - risk {row['risk']:.0f}/100 - confidence {confidence_text}"
    lines = [f"t = {row['t_rel']:.1f} s into the capture, window of {int(row['n_msgs'])} frames"]
    reasons = [r for r in str(row["reasons"]).split(" | ") if r and r != "nan"]
    if reasons:
        lines.append("Why:")
        lines += [f"  - {r}" for r in reasons]
    console.print(Panel("\n".join(lines), title=title, border_style=style, title_align="left"))


def print_detection_table(console, detection):
    table = Table(title="Detection on the test part (held out in time)")
    for column in ["capture", "windows", "attack windows", "precision", "recall", "F1", "FPR", "AUC"]:
        table.add_column(column, justify="left" if column == "capture" else "right")
    for row in detection:
        table.add_row(str(row["capture"]), str(row["windows"]), str(row["attack_windows"]),
                      f"{row['precision']:.3f}", f"{row['recall']:.3f}", f"{row['f1']:.3f}",
                      f"{row['false_positive_rate']:.3f}", f"{row['auc']:.3f}")
    console.print(table)


def print_confusion_table(console, confusion):
    """confusion: the 'split' dict saved in metrics.json (index, columns, data)."""
    table = Table(title="Attack-type classifier: confusion matrix (rows = true, columns = predicted)")
    table.add_column("")
    for column in confusion["columns"]:
        table.add_column(column.replace("pred ", ""), justify="right")
    for name, values in zip(confusion["index"], confusion["data"]):
        table.add_row(name.replace("true ", ""), *[str(v) for v in values])
    console.print(table)


def print_evasion_table(console, evasion):
    if len(evasion) == 0:
        console.print("[dim]no evasion results[/dim]")
        return
    table = Table(title="Evasion test: recall as the attacker injects fewer frames")
    for column in ["attack", "kept injected frames", "injected frames/s", "recall", "mean risk"]:
        table.add_column(column, justify="left" if column == "attack" else "right")
    for _, r in evasion.iterrows():
        table.add_row(str(r["attack_type"]), f"{r['keep_fraction']:.0%}",
                      f"{r['injected_frames_per_second']:.0f}", f"{r['recall']:.3f}",
                      f"{r['mean_risk_on_attack_windows']:.1f}")
    console.print(table)
    console.print("[dim]Caveat: DoS and fuzzy attacks also disturb the legitimate traffic, and removing "
                  "injected frames from a capture does not undo that. Recall stays high for them for that "
                  "reason, so read the spoofing rows as the meaningful ones.[/dim]")