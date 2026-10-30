"""CarWarden command line.

  python src/carwarden.py demo                      synthetic data, whole pipeline (no download needed)
  python src/carwarden.py train --data-dir data     train + evaluate on the real HCRL files
  python src/carwarden.py scan data/DoS_dataset.csv score one capture with the trained model
  python src/carwarden.py report                    print the saved results again
"""
import argparse
import json
import os
import sys

import joblib
import pandas as pd

from data import load_frames
from features import make_windows
from pipeline import DEFAULT_FILES, add_predictions, run_pipeline
from report import (make_console, print_confusion_table, print_detection_table,
                    print_evasion_table, print_incident)
from synth import generate_dataset


def print_results(console, out_dir, top=0):
    with open(os.path.join(out_dir, "metrics.json")) as f:
        metrics = json.load(f)
    print_detection_table(console, metrics["detection"])
    console.print(f"Attack-type classifier accuracy on test windows: {metrics['classifier_accuracy']:.3f}")
    print_confusion_table(console, metrics["confusion_matrix"])
    evasion_path = os.path.join(out_dir, "evasion.csv")
    if os.path.exists(evasion_path):
        try:
            print_evasion_table(console, pd.read_csv(evasion_path))
        except pd.errors.EmptyDataError:
            pass
    if top > 0:
        scored = pd.read_csv(os.path.join(out_dir, "scored_windows.csv")).fillna({"reasons": ""})
        for _, row in scored[scored["flagged"]].sort_values("risk", ascending=False).head(top).iterrows():
            print_incident(console, row)


def command_demo(args, console):
    console.print("Generating synthetic CAN captures (for testing only, not real vehicle data)...")
    paths = generate_dataset(args.synthetic_dir, duration=args.duration)
    run_pipeline(paths, out_dir=args.out, model_dir=args.model_dir, window_seconds=args.window,
                 log=console.print)
    print_results(console, args.out, top=args.top)


def command_train(args, console):
    files = {name: os.path.join(args.data_dir, filename) for name, filename in DEFAULT_FILES.items()}
    missing = [p for p in files.values() if not os.path.exists(p)]
    if missing:
        console.print("[red]Missing files:[/red] " + ", ".join(missing))
        return 1
    run_pipeline(files, out_dir=args.out, model_dir=args.model_dir, window_seconds=args.window,
                 max_rows=args.max_rows, run_evasion=not args.no_evasion,
                 trim_gap=args.trim_gap, log=console.print)
    print_results(console, args.out, top=args.top)
    return 0


def command_scan(args, console):
    model = joblib.load(args.model)
    detector, classifier, window_seconds = model["detector"], model["classifier"], model["window_seconds"]
    frames = load_frames(args.file, max_rows=args.max_rows).sort_values("timestamp").reset_index(drop=True)
    t0 = float(frames["timestamp"].iloc[0])
    windows = make_windows(frames, window_seconds, detector.known_ids_, "attack", t0=t0)
    windows["capture"] = "scan"
    windows["t_rel"] = windows["start_time"] - t0
    scored = add_predictions(detector, classifier, windows, {"scan": frames}, window_seconds)

    flagged = scored[scored["flagged"]]
    console.print(f"{args.file}: {len(scored)} windows, {len(flagged)} flagged")
    if scored["is_attack"].any():
        caught = int((scored["flagged"] & scored["is_attack"]).sum())
        console.print(f"capture contains injected frames in {int(scored['is_attack'].sum())} windows; "
                      f"flagged {caught} of them, {len(flagged) - caught} flagged windows had none")
    for _, row in flagged.sort_values("risk", ascending=False).head(args.top).iterrows():
        print_incident(console, row)
    return 0


def command_report(args, console):
    print_results(console, args.out, top=args.top)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="carwarden", description="AI security analyst for in-vehicle CAN networks")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--out", default="outputs", help="folder for results")
        p.add_argument("--model-dir", default="models")
        p.add_argument("--window", type=float, default=0.1, help="window length in seconds")
        p.add_argument("--top", type=int, default=3, help="how many incidents to print")

    demo = sub.add_parser("demo", help="run everything on synthetic data")
    add_common(demo)
    demo.set_defaults(out="outputs_demo", model_dir="models_demo")
    demo.add_argument("--synthetic-dir", default="data_synthetic")
    demo.add_argument("--duration", type=float, default=100.0)

    train = sub.add_parser("train", help="train and evaluate on the real HCRL files")
    add_common(train)
    train.add_argument("--data-dir", default="data")
    train.add_argument("--max-rows", type=int, default=None, help="only read this many rows per file (quick test)")
    train.add_argument("--no-evasion", action="store_true")
    train.add_argument("--trim-gap", type=float, default=None, metavar="SECONDS",
                       help="drop everything after the first gap between frames longer than this "
                            "(use 10 for the HCRL files)")

    scan = sub.add_parser("scan", help="score one capture file with the trained model")
    scan.add_argument("file")
    scan.add_argument("--model", default="models/carwarden.joblib")
    scan.add_argument("--max-rows", type=int, default=None)
    scan.add_argument("--top", type=int, default=5)

    report = sub.add_parser("report", help="print saved results")
    report.add_argument("--out", default="outputs")
    report.add_argument("--top", type=int, default=3)
    return parser


def main(argv=None, console=None):
    console = console or make_console()
    args = build_parser().parse_args(argv)
    handlers = {"demo": command_demo, "train": command_train, "scan": command_scan, "report": command_report}
    result = handlers[args.command](args, console)
    return result or 0


if __name__ == "__main__":
    sys.exit(main())