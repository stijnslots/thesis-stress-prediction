"""
Extract task-window boundaries (Stroop, Arithmetic, Face Recall) from a
PsychoPy (CSV).

Excluded on purpose: trials_stroop_pr, trials_arithmetic_pr (practice blocks)
    trials_faceMemory + Rating2 (encoding phase has no active response / time
    pressure, so it is not a comparable cognitive-load window. Also no stress score))
"""

import argparse
import csv
import glob
import os
import re
import sys
from datetime import datetime

# task -> (loop column prefix, begin marker column, end marker column, rating index)
TASKS = {
    "stroop": {
        "loop": "trials_stroop",
        "begin_marker": "stroop_begin.started",
        "end_marker": "blank.stopped",
        "rating_col": "key_resp_11.keys",
        "rating_rt_col": "key_resp_11.rt",
    },
    "arithmetic": {
        "loop": "trials_arithmetic",
        "begin_marker": "arithmetic_begin.started",
        "end_marker": "arithmetic_2.stopped",
        "rating_col": "key_resp_13.keys",
        "rating_rt_col": "key_resp_13.rt",
    },
    "face_recall": {
        "loop": "trials_faceRecall",
        "begin_marker": "instruct_faceRecall.started",
        "end_marker": "face_recall_blank_2.stopped",
        "rating_col": "key_resp_14.keys",
        "rating_rt_col": "key_resp_14.rt",
    },
}

def find_psychopy_csv(psychopy_dir, participant):
    pattern = os.path.join(psychopy_dir, f"{participant}_*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No PsychoPy csv found for participant '{participant}' in {psychopy_dir} (pattern: {pattern})")
    if len(matches) > 1:
        print(f"Warning: multiple PsychoPy csv files found for '{participant}', using the first: {matches[0]}", file=sys.stderr)
    return matches[0]


def parse_exp_start(exp_start_str):
    # e.g. "2026-03-18 13h57.42.262463 +0100"
    return datetime.strptime(exp_start_str.strip(), "%Y-%m-%d %Hh%M.%S.%f %z")


def load_rows(csv_path):
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def nonempty(row, col):
    val = row.get(col, "")
    return val not in ("", None)


def extract_task_window(rows, task_name, spec):
    loop_col = f"{spec['loop']}.thisN"
    idxs = [i for i, row in enumerate(rows) if nonempty(row, loop_col)]
    if not idxs:
        raise ValueError(f"No rows found for loop column '{loop_col}' (task '{task_name}')")

    first_idx, last_idx = idxs[0], idxs[-1]
    first_row, last_row = rows[first_idx], rows[last_idx]

    # Preferred start: instruction/begin-routine marker just before the loop.
    # PsychoPy writes standalone (non-loop) routines to their own row, so the
    # marker sits on a row before the loop's first row, not on first_row itself.
    # Fall back to the first trial's own onset if the marker can't be found.
    begin_col = spec["begin_marker"]
    begin_idxs = [i for i in range(first_idx) if nonempty(rows[i], begin_col)]
    if begin_idxs:
        start_s = float(rows[begin_idxs[-1]][begin_col])
        start_source = begin_col
    else:
        # fall back: earliest non-empty ".started" timestamp on the first trial row
        started_cols = [c for c in first_row if c.endswith(".started") and nonempty(first_row, c)]
        if not started_cols:
            raise ValueError(f"No usable start marker for task '{task_name}' (row {first_idx + 2})")
        start_s = min(float(first_row[c]) for c in started_cols)
        start_source = f"fallback:min(.started) on first trial row"

    end_col = spec["end_marker"]
    if not nonempty(last_row, end_col):
        raise ValueError(f"End marker column '{end_col}' empty on last trial row ({last_idx + 2}) for task '{task_name}'")
    end_s = float(last_row[end_col])

    n_trials = idxs[-1] - idxs[0] + 1
    if n_trials != len(idxs):
        print(f"Warning: task '{task_name}' loop rows are not contiguous ({len(idxs)} matched rows spanning {n_trials} csv rows)", file=sys.stderr)

    rating_col = spec["rating_col"]
    rating_rt_col = spec["rating_rt_col"]
    # Rating routine is the row immediately after the last trial row.
    rating_row_idx = last_idx + 1
    rating_value, rating_rt = None, None
    if rating_row_idx < len(rows):
        rating_row = rows[rating_row_idx]
        if nonempty(rating_row, rating_col):
            rating_value = rating_row[rating_col]
            rating_rt = rating_row.get(rating_rt_col) or None

    return {
        "task": task_name,
        "n_trials": len(idxs),
        "first_row": first_idx + 2,  # +2: 1-indexed + header row
        "last_row": last_idx + 2,
        "start_s": start_s,
        "start_source": start_source,
        "end_s": end_s,
        "end_source": end_col,
        "duration_s": end_s - start_s,
        "rating_col": rating_col,
        "rating_value": rating_value,
        "rating_rt_s": rating_rt,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("participant", help="Participant id, e.g. p20")
    parser.add_argument("--psychopy-dir", default=os.path.join("data", "raw", "psychopy"))
    parser.add_argument("--out", default=os.path.join("data", "processed", "synced"))
    args = parser.parse_args()

    csv_path = find_psychopy_csv(args.psychopy_dir, args.participant)
    fields, rows = load_rows(csv_path)
    if not rows:
        raise ValueError(f"No data rows in {csv_path}")

    meta_row = rows[0]
    exp_start_str = meta_row.get("expStart", "")
    exp_start_dt = parse_exp_start(exp_start_str) if exp_start_str else None

    results = []
    for task_name, spec in TASKS.items():
        window = extract_task_window(rows, task_name, spec)
        window["participant"] = args.participant
        window["exp_start"] = exp_start_str
        if exp_start_dt is not None:
            for prefix, s_key in (("start", "start_s"), ("end", "end_s")):
                abs_dt = exp_start_dt.timestamp() + window[s_key]
                window[f"{prefix}_abs_unix"] = abs_dt
                window[f"{prefix}_abs_iso"] = datetime.fromtimestamp(abs_dt, tz=exp_start_dt.tzinfo).isoformat()
        results.append(window)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.participant}_task_windows.csv")
    out_fields = [
        "participant", "task", "n_trials", "first_row", "last_row",
        "start_s", "start_source", "end_s", "end_source", "duration_s",
        "rating_col", "rating_value", "rating_rt_s",
        "exp_start", "start_abs_unix", "start_abs_iso", "end_abs_unix", "end_abs_iso",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in out_fields})

    print(f"Wrote {len(results)} task windows to {out_path}")
    for row in results:
        print(f"  {row['task']:12s} start={row['start_s']:.3f}s  end={row['end_s']:.3f}s  "
              f"duration={row['duration_s']:.1f}s  rating={row['rating_value']}  n_trials={row['n_trials']}")


if __name__ == "__main__":
    main()
