"""
Merge PPG, video and label features for one participant into a single
master feature table, one row per task.

Either PPG or video can be missing for a given participant, which is a
real, permanent situation for some (missing data), not just not-yet-
processed. Both cases are handled the same way: the missing source's
columns are filled with NaN.

Usage:
    python 01_merge_features.py p20
    python 01_merge_features.py p35 --allow-missing-video
    python 01_merge_features.py p1 --allow-missing-ppg
"""

import argparse
import os

import pandas as pd

# Task name from a normalized variant.
TASK_NAME_MAP = {
    "stroop": "stroop",
    "arithmetic": "arithmetic",
    "facerecall": "face_recall",
    "face_recall": "face_recall",
}


def normalize_task_name(name):
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    if key not in TASK_NAME_MAP:
        raise ValueError(f"Unrecognized task name '{name}' -- add it to TASK_NAME_MAP in this script "
                          "if it's a legitimate naming variant.")
    return TASK_NAME_MAP[key]


def load_and_normalize(path, source_label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{source_label} features not found: {path}")
    df = pd.read_csv(path)
    if "task" not in df.columns:
        raise ValueError(f"{source_label} file {path} has no 'task' column.")
    original_tasks = df["task"].tolist()
    df["task"] = df["task"].apply(normalize_task_name)
    changed = [(o, n) for o, n in zip(original_tasks, df["task"]) if o != n]
    if changed:
        print(f"  {source_label}: normalized task names: {changed}")
    return df


def load_or_placeholder(path, source_label, allow_missing, template_participant, template_dir, task_frame):
    """Load `path` normally, or -- if it's missing and allowed -- build a
    NaN-filled placeholder with the same columns as
    {template_dir}/{template_participant}_{basename of path}."""
    pending = not os.path.exists(path)
    if pending and not allow_missing:
        raise FileNotFoundError(f"{source_label} features not found: {path} "
                                 f"(pass --allow-missing-{source_label.lower()} to build a partial table instead)")

    if not pending:
        df = load_and_normalize(path, source_label)
        cols = [c for c in df.columns if c not in ("participant", "task")]
        return df, cols, False

    template_name = os.path.basename(path).split("_", 1)[1]  # e.g. "ppg_features.csv"
    template_path = os.path.join(template_dir, f"{template_participant}_{template_name}")
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"{source_label} template {template_path} not found -- needed to know the "
                                 "column layout for the NaN placeholder columns.")
    template = pd.read_csv(template_path)
    cols = [c for c in template.columns if c not in ("participant", "task")]
    print(f"  {source_label}: {path} not found -- building NaN placeholder columns from "
          f"{template_participant}'s layout ({len(cols)} columns), {source_label.lower()}_pending=True")
    df = task_frame[["participant", "task"]].copy()
    for c in cols:
        df[c] = float("nan")
    return df, cols, True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("participant", help="Participant id, e.g. p20")
    parser.add_argument("--ppg-path", default=None, help="Default: data/processed/ppg/{participant}_ppg_features.csv")
    parser.add_argument("--video-path", default=None, help="Default: data/processed/video/{participant}_video_features.csv")
    parser.add_argument("--labels-path", default=None, help="Default: data/processed/labels/{participant}_labels.csv")
    parser.add_argument("--out-dir", default=os.path.join("data", "processed", "features"))
    parser.add_argument("--allow-missing-video", action="store_true",
                         help="If set, a missing video_features.csv is not an error -- video columns are "
                              "filled with NaN and video_pending=True is set, instead of raising.")
    parser.add_argument("--allow-missing-ppg", action="store_true",
                         help="If set, a missing ppg_features.csv is not an error -- PPG columns are "
                              "filled with NaN and ppg_pending=True is set, instead of raising. Needed for "
                              "participants whose raw PPG is permanently unusable (e.g. p1) but who DO have "
                              "video, not just for 'not processed yet'.")
    parser.add_argument("--video-template-participant", default="p20",
                         help="Whose video_features.csv column layout to copy when this participant's "
                              "own video is missing (default: p20).")
    parser.add_argument("--ppg-template-participant", default="p20",
                         help="Whose ppg_features.csv column layout to copy when this participant's "
                              "own PPG is missing (default: p20).")
    args = parser.parse_args()

    ppg_path = args.ppg_path or os.path.join("data", "processed", "ppg", f"{args.participant}_ppg_features.csv")
    video_path = args.video_path or os.path.join("data", "processed", "video", f"{args.participant}_video_features.csv")
    labels_path = args.labels_path or os.path.join("data", "processed", "labels", f"{args.participant}_labels.csv")

    print("Loading source feature tables ...")
    labels = load_and_normalize(labels_path, "Labels")

    ppg, ppg_cols, ppg_pending = load_or_placeholder(
        ppg_path, "PPG", args.allow_missing_ppg, args.ppg_template_participant,
        os.path.join("data", "processed", "ppg"), labels)
    video, video_cols, video_pending = load_or_placeholder(
        video_path, "Video", args.allow_missing_video, args.video_template_participant,
        os.path.join("data", "processed", "video"), labels)

    ppg_tasks, video_tasks, label_tasks = set(ppg["task"]), set(video["task"]), set(labels["task"])
    if not (ppg_tasks == video_tasks == label_tasks):
        print(f"  WARNING: task sets differ across sources -- PPG={ppg_tasks}, Video={video_tasks}, "
              f"Labels={label_tasks}. Rows for non-overlapping tasks will be dropped by the inner merge below.")

    overlap = set(ppg.columns) & set(video.columns) - {"participant", "task"}
    if overlap:
        print(f"  WARNING: PPG and video feature tables share non-key column name(s) {overlap} -- "
              "pandas will suffix them (_x/_y); check the output columns.")

    merged = labels.merge(ppg, on=["participant", "task"], how="inner", validate="one_to_one")
    merged = merged.merge(video, on=["participant", "task"], how="inner", validate="one_to_one")
    merged["ppg_pending"] = ppg_pending
    merged["video_pending"] = video_pending

    if len(merged) != 3:
        print(f"  WARNING: expected 3 rows (Stroop/Arithmetic/Face Recall), got {len(merged)}. "
              "Check the task-set warning above.")

    other_cols = [c for c in merged.columns
                  if c not in ("participant", "task", "stress_label", "ppg_pending", "video_pending")]
    ordered_cols = ["participant", "task", "stress_label", "ppg_pending", "video_pending"] + ppg_cols + video_cols
    missing = set(other_cols) - set(ordered_cols)
    if missing:
        # shouldn't happen given the merges above, but don't silently drop columns if it does
        ordered_cols += sorted(missing)
    merged = merged[ordered_cols]

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.participant}_master_features.csv")
    merged.to_csv(out_path, index=False)

    print(f"\nSaved master feature table: {out_path}")
    print(f"  {merged.shape[0]} rows x {merged.shape[1]} columns "
          f"(1 participant + 1 task + 1 label + ppg_pending + video_pending + "
          f"{len(ppg_cols)} PPG + {len(video_cols)} video)")
    print(f"  ppg_pending: {ppg_pending}, video_pending: {video_pending}")
    print("\n=== Preview ===")
    print(merged[["participant", "task", "stress_label"]].to_string(index=False))


if __name__ == "__main__":
    main()
