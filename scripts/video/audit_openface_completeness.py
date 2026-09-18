"""
Audits every cached OpenFace output file for silent truncation: counts the
actual rows and compares against the expected frame count for that
participant and task's duration. Mirrors the completeness check built into
the main video extraction script, so older or externally produced outputs
can be re-checked on demand.

Usage:
    python scripts/video/audit_openface_completeness.py
    python scripts/video/audit_openface_completeness.py --min-ratio 0.95
"""

import argparse
import glob
import os

import pandas as pd

NOMINAL_FPS = 60


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--openface-dir", default=os.path.join("data", "processed", "video", "openface_raw"))
    parser.add_argument("--windows-dir", default=os.path.join("data", "processed", "synced"))
    parser.add_argument("--min-ratio", type=float, default=0.95)
    args = parser.parse_args()

    task_dirs = sorted(d for d in glob.glob(os.path.join(args.openface_dir, "*")) if os.path.isdir(d))
    if not task_dirs:
        print(f"No task directories found under {args.openface_dir}")
        return

    suspects = []
    checked = 0
    for d in task_dirs:
        name = os.path.basename(d)
        if "_" not in name:
            continue
        participant, task = name.split("_", 1)
        windows_path = os.path.join(args.windows_dir, f"{participant}_task_windows.csv")
        if not os.path.exists(windows_path):
            print(f"SKIP {name}: no task_windows.csv for {participant}")
            continue
        windows = pd.read_csv(windows_path)
        row = windows[windows["task"] == task]
        if row.empty:
            print(f"SKIP {name}: task '{task}' not found in {windows_path}")
            continue
        duration_s = float(row["duration_s"].iloc[0])
        expected = duration_s * NOMINAL_FPS

        csvs = glob.glob(os.path.join(d, "*.csv"))
        if not csvs:
            print(f"SKIP {name}: no csv output present")
            continue

        n_rows = len(pd.read_csv(csvs[0]))
        checked += 1
        ratio = n_rows / expected if expected else 0
        if ratio < args.min_ratio:
            suspects.append((name, n_rows, expected, ratio))
            print(f"SUSPECT: {participant} {task}: {n_rows} rows vs expected ~{expected:.0f} ({ratio*100:.1f}%)")

    print(f"\nChecked {checked} OpenFace output(s); {len(suspects)} suspect(s) found "
          f"(below {args.min_ratio*100:.0f}% completeness).")
    if not suspects:
        print("All checked outputs pass the completeness threshold.")


if __name__ == "__main__":
    main()
