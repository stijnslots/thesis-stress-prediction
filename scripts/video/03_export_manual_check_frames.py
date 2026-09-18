"""
Manual-check script that exports selected frames from the raw video as
images for manual quality checks of the OpenFace-based video feature
extraction.

For each task, exports the first and last frame of the task window, the
3 highest-AU04 and 3 highest-AU07 frames, and one representative frame per
low-confidence episode.

Usage:
    python 03_export_manual_check_frames.py p20
"""

import argparse
import os
import re

import imageio_ffmpeg
import numpy as np
import pandas as pd
import subprocess

TASK_DISPLAY_NAMES = {"stroop": "stroop", "arithmetic": "arithmetic", "face_recall": "facerecall"}


def find_openface_csv(openface_raw_dir, participant, task):
    task_dir = os.path.join(openface_raw_dir, f"{participant}_{task}")
    candidates = [f for f in os.listdir(task_dir) if f.endswith(".csv")]
    if not candidates:
        raise FileNotFoundError(f"No OpenFace csv found in {task_dir}")
    return os.path.join(task_dir, candidates[0])


def compute_task_video_offset(task_windows_path, obs_log_path, task):
    from datetime import datetime
    windows = pd.read_csv(task_windows_path)
    row = windows[windows["task"] == task].iloc[0]
    task_start_abs = datetime.fromisoformat(row["start_abs_iso"])
    with open(obs_log_path, encoding="utf-8") as f:
        content = f.read()
    start_line = next(l for l in content.splitlines() if l.startswith("EVENT:START RECORDING"))
    ts_str = start_line.split("@", 1)[1].strip()
    obs_start_naive = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    obs_start_abs = obs_start_naive.replace(tzinfo=task_start_abs.tzinfo)
    return (task_start_abs - obs_start_abs).total_seconds()


def pick_distinct_peaks(df, col, n, min_gap_s):
    """Top-n rows by `col`, enforcing a minimum time gap between picks so
    they represent distinct moments rather than adjacent frames of one peak."""
    sorted_df = df.sort_values(col, ascending=False)
    picked = []
    for _, row in sorted_df.iterrows():
        if all(abs(row["timestamp"] - p["timestamp"]) >= min_gap_s for p in picked):
            picked.append(row)
        if len(picked) == n:
            break
    return picked


def find_lowconf_episodes(df, threshold=0.95):
    low = (df["confidence"] < threshold).to_numpy()
    if not low.any():
        return []
    diff = np.diff(low.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if low[0]:
        starts = np.insert(starts, 0, 0)
    if low[-1]:
        ends = np.append(ends, len(low))
    episodes = []
    for s, e in zip(starts, ends):
        seg = df.iloc[s:e]
        min_row = seg.loc[seg["confidence"].idxmin()]
        episodes.append({
            "start_time_s": seg["timestamp"].iloc[0],
            "end_time_s": seg["timestamp"].iloc[-1],
            "duration_s": seg["timestamp"].iloc[-1] - seg["timestamp"].iloc[0],
            "min_confidence": min_row["confidence"],
            "min_confidence_row": min_row,
        })
    return episodes


def extract_frame(ffmpeg_exe, video_path, abs_time_s, out_path):
    coarse = max(abs_time_s - 3.0, 0.0)
    fine = abs_time_s - coarse
    cmd = [ffmpeg_exe, "-y", "-ss", f"{coarse:.3f}", "-i", video_path,
           "-ss", f"{fine:.3f}", "-vframes", "1", "-q:v", "2", out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"Frame extraction failed for {out_path}: {result.stderr[-500:]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("participant")
    parser.add_argument("--tasks", nargs="+", default=["stroop", "arithmetic", "face_recall"])
    parser.add_argument("--video-dir", default=os.path.join("data", "raw", "video"))
    parser.add_argument("--windows-dir", default=os.path.join("data", "processed", "synced"))
    parser.add_argument("--obs-log-dir", default=os.path.join("data", "raw", "obs_logs"))
    parser.add_argument("--openface-raw-dir", default=os.path.join("data", "processed", "video", "openface_raw"))
    parser.add_argument("--out-dir", default=os.path.join("results", "video_exploration", "manual_check"))
    parser.add_argument("--min-peak-gap-s", type=float, default=3.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.95)
    parser.add_argument("--max-lowconf-frames", type=int, default=20,
                         help="Cap on exported low-confidence images per task; full episode list always goes in the manifest")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # Prefix match.
    id_pattern = re.compile(rf"^{re.escape(args.participant)}(?!\d)", re.IGNORECASE)
    video_matches = [f for f in os.listdir(args.video_dir) if id_pattern.match(f)]
    video_path = os.path.join(args.video_dir, sorted(video_matches)[0])
    windows_path = os.path.join(args.windows_dir, f"{args.participant}_task_windows.csv")
    obs_matches = [f for f in os.listdir(args.obs_log_dir) if id_pattern.match(f)]
    obs_log_path = os.path.join(args.obs_log_dir, sorted(obs_matches)[0])

    manifest_rows = []
    all_lowconf_episodes = []

    for task in args.tasks:
        print(f"\n=== {task} ===")
        csv_path = find_openface_csv(args.openface_raw_dir, args.participant, task)
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        offset_s = compute_task_video_offset(windows_path, obs_log_path, task)
        disp = TASK_DISPLAY_NAMES.get(task, task)

        selections = []  # (filename, category, row, value_col, value)

        first_row = df.iloc[0]
        selections.append((f"{disp}_start.png", "task_start", first_row, "timestamp", first_row["timestamp"]))
        last_row = df.iloc[-1]
        selections.append((f"{disp}_end.png", "task_end", last_row, "timestamp", last_row["timestamp"]))

        for au, label in [("AU04_r", "au04_peak"), ("AU07_r", "au07_peak")]:
            peaks = pick_distinct_peaks(df, au, 3, args.min_peak_gap_s)
            for i, row in enumerate(peaks, 1):
                selections.append((f"{disp}_{label}{i}.png", label, row, au, row[au]))
            if len(peaks) < 3:
                print(f"  Note: only found {len(peaks)}/3 distinct {au} peaks (>= {args.min_peak_gap_s}s apart)")

        episodes = find_lowconf_episodes(df, args.confidence_threshold)
        for ep in episodes:
            ep["task"] = task
        all_lowconf_episodes.extend(episodes)
        print(f"  {len(episodes)} low-confidence (<{args.confidence_threshold}) episodes found")

        episodes_to_export = sorted(episodes, key=lambda e: e["min_confidence"])[:args.max_lowconf_frames]
        if len(episodes) > args.max_lowconf_frames:
            print(f"  Exporting images for the {args.max_lowconf_frames} most severe episodes only "
                  f"(all {len(episodes)} are in the manifest CSV)")
        # Rounded timestamps, append a letter suffix so no export overwrites
        # another.
        seconds_seen = {}
        for ep in episodes_to_export:
            row = ep["min_confidence_row"]
            t_int = int(round(row["timestamp"]))
            seconds_seen[t_int] = seconds_seen.get(t_int, 0) + 1
            suffix = "" if seconds_seen[t_int] == 1 else f"_{chr(ord('a') + seconds_seen[t_int] - 2)}"
            selections.append((f"{disp}_lowconf_{t_int}s{suffix}.png", "low_confidence", row, "confidence", row["confidence"]))

        seen_filenames = set()
        for filename, category, row, value_col, value in selections:
            if filename in seen_filenames:
                raise RuntimeError(f"Duplicate export filename within task '{task}': {filename} "
                                    "would silently overwrite a previous export -- fix the naming logic.")
            seen_filenames.add(filename)
            abs_time_s = offset_s + row["timestamp"]
            out_path = os.path.join(args.out_dir, filename)
            extract_frame(ffmpeg_exe, video_path, abs_time_s, out_path)
            manifest_rows.append({
                "filename": filename,
                "task": task,
                "category": category,
                "frame_number": int(row["frame"]),
                "timestamp_in_task_s": round(float(row["timestamp"]), 3),
                "abs_video_time_s": round(abs_time_s, 3),
                "value_column": value_col,
                "value": round(float(value), 4) if value_col != "timestamp" else "",
            })
            print(f"  exported {filename}  (t={row['timestamp']:.3f}s, {value_col}={value if value_col=='timestamp' else round(float(value),4)})")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    episodes_df = pd.DataFrame([{k: v for k, v in ep.items() if k != "min_confidence_row"} for ep in all_lowconf_episodes])
    episodes_path = os.path.join(args.out_dir, "lowconf_episodes_all.csv")
    episodes_df.to_csv(episodes_path, index=False)

    print(f"\nExported {len(manifest_rows)} frames to {args.out_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Full low-confidence episode list ({len(all_lowconf_episodes)} total): {episodes_path}")
    print("\n=== Manifest ===")
    print(manifest_df.to_string(index=False))


if __name__ == "__main__":
    main()
