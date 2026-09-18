"""
Extract per-task stress-rating target labels for one participant from the
raw PsychoPy data.

Rating1 is the target for Stroop, Rating3 for Arithmetic, Rating4 for Face
Recall. Rating2, after the face-memory encoding phase, is not used, since
that phase has no active response or time pressure.

Each rating is given on a continuous 0 to 9 scale. Values are validated to
fall in that range; a missing or out-of-range value is reported rather
than silently dropped or clamped.

Usage:
    python 01_extract_stress_labels.py p20
"""

import argparse
import csv
import glob
import os

TASK_RATING_COLUMN = {
    "stroop": "key_resp_11.keys",
    "arithmetic": "key_resp_13.keys",
    "face_recall": "key_resp_14.keys",
}
VALID_RANGE = (0.0, 9.0)


def find_psychopy_csv(psychopy_dir, participant):
    pattern = os.path.join(psychopy_dir, f"{participant}_*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No PsychoPy csv found for participant '{participant}' in {psychopy_dir} "
                                 f"(pattern: {pattern})")
    if len(matches) > 1:
        print(f"Warning: multiple PsychoPy csv files match '{participant}', using the first: {matches[0]}")
    return matches[0]


def load_rows(csv_path):
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def extract_label(rows, rating_col, task):
    nonempty = [(i, row[rating_col]) for i, row in enumerate(rows)
                if row.get(rating_col, "") not in ("", None)]

    if not nonempty:
        print(f"  WARNING [{task}]: no response found in column '{rating_col}' -- label missing.")
        return None

    if len(nonempty) > 1:
        print(f"  WARNING [{task}]: {len(nonempty)} responses found in column '{rating_col}' "
              f"(expected exactly 1) -- using the first at row {nonempty[0][0] + 2}.")

    row_idx, raw_value = nonempty[0]
    try:
        value = float(raw_value)
    except ValueError:
        print(f"  WARNING [{task}]: response '{raw_value}' in column '{rating_col}' (row {row_idx + 2}) "
              f"is not numeric -- label invalid.")
        return None

    lo, hi = VALID_RANGE
    if not (lo <= value <= hi):
        print(f"  WARNING [{task}]: value {value} in column '{rating_col}' (row {row_idx + 2}) "
              f"is outside the expected [{lo}, {hi}] range.")

    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("participant", help="Participant id, e.g. p20")
    parser.add_argument("--psychopy-dir", default=os.path.join("data", "raw", "psychopy"))
    parser.add_argument("--out-dir", default=os.path.join("data", "processed", "labels"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    csv_path = find_psychopy_csv(args.psychopy_dir, args.participant)
    print(f"Loading PsychoPy data from {csv_path} ...")
    rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f"No data rows in {csv_path}")

    print(f"Extracting stress labels for {args.participant} "
          f"(Rating1->stroop, Rating3->arithmetic, Rating4->face_recall; Rating2 excluded per methodology_notes.md)")
    label_rows = []
    for task, rating_col in TASK_RATING_COLUMN.items():
        value = extract_label(rows, rating_col, task)
        label_rows.append({"participant": args.participant, "task": task, "stress_label": value})

    out_path = os.path.join(args.out_dir, f"{args.participant}_labels.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["participant", "task", "stress_label"])
        writer.writeheader()
        writer.writerows(label_rows)

    print(f"\nSaved labels: {out_path}")
    print("\n=== Stress labels ===")
    for row in label_rows:
        print(f"  {row['task']:12s} stress_label = {row['stress_label']}")


if __name__ == "__main__":
    main()
