"""
Extracts per-task facial and video features (blink rate, gaze, head pose,
AU04/AU07) for one participant, using OpenFace 2.0.

Key design choices:

- OpenFace runs once per participant on the full, uncut session video, not
  on three separately-cut task clips. Testing showed OpenFace's AU/blink
  output depends on how much surrounding video context it sees: per-clip
  versus session-wide processing gave meaningfully different AU values and
  a very different blink count on identical frames. Since blink detection
  drives the exclusion filter below, that's not cosmetic. The 3 task
  windows are sliced out of the session-wide output afterward by
  timestamp.

- Camera intrinsics are passed explicitly (same webcam and zoom for every
  participant) instead of letting OpenFace estimate them. This measurably
  corrects absolute head pose and vertical gaze, but doesn't affect blink
  contamination: those turned out to be separate problems.

- Blink filtering is adaptive per participant: a fixed duration threshold
  tuned on one participant didn't generalize to others, so the threshold
  is now derived per participant from their own blink-duration
  distribution (a standard IQR outlier fence). Frames inside a detected
  blink, plus a margin either side, are excluded before averaging AU,
  gaze and pose values. A separate low-confidence filter catches frames
  the blink filter misses.

- Head pitch and one gaze column are dropped from the final feature table:
  both stayed flagged as contaminated even after the two fixes above, so
  rather than publish unreliable values they're left out of the summary
  (the raw per-frame data still has them).

Output: one row per task in data/processed/video/{participant}_video_features.csv,
plus the per-participant blink threshold and quality diagnostics.

Usage:
    python 02_extract_video_features.py p20 --openface-exe "C:/path/to/FeatureExtraction.exe"
"""

import argparse
import glob
import os
import re
import subprocess
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

from blink_utils import (au45_blink_run_durations_ms, derive_adaptive_blink_thresholds,
                          compute_blink_exclusion_mask, EXCLUDED_CONTAMINATED_FEATURES)

DEFAULT_TASKS = ["stroop", "arithmetic", "face_recall"]

# Margin (frames) excluded on both sides of every real blink episode
# before averaging AU, gaze and pose values.
BLINK_FILTER_MARGIN_FRAMES = 15

# Frames below this OpenFace tracking confidence are excluded from
# aggregation, separately from the blink filter but added to the
# excluded pool.
CONFIDENCE_EXCLUSION_THRESHOLD = 0.7

# A pooled raw AU04 median above this is flagged as a likely glasses or
# occlusion bias rather than brow activity. No real literature
# validation (CHECK this).
GLASSES_AU04_MEDIAN_THRESHOLD = 0.3

# Estimate for progress printing.
OPENFACE_SECONDS_PER_FRAME_ESTIMATE = 238.0 / 2700.0

# Estimate focal length from image size. Calibration measurably corrects
# absolute head pose and gaze, but has no effect on blink-related
# contamination.
CAMERA_FX = 1637
CAMERA_FY = 1637
CAMERA_CX = 960
CAMERA_CY = 540

# Imported from blink_utils.


def find_file(directory, participant, must_contain=None):
    """Prefix match (participant id not followed by another digit), NOT a
    plain substring check -- 'p2' is a substring of 'p20.mp4', 'p21', ...,
    'p29', so p1-p7 could silently grab a sibling participant's file from
    their own decade. The (?!\\d) lookahead rejects those false matches."""
    pattern = re.compile(rf"^{re.escape(participant)}(?!\d)", re.IGNORECASE)
    candidates = [
        f for f in os.listdir(directory)
        if pattern.match(f) and (must_contain is None or must_contain.lower() in f.lower())
    ]
    if not candidates:
        raise FileNotFoundError(f"No file matching participant '{participant}' "
                                 f"(must_contain={must_contain!r}) in {directory}")
    if len(candidates) > 1:
        print(f"Warning: multiple files match in {directory}, using the first: {sorted(candidates)[0]}")
    return os.path.join(directory, sorted(candidates)[0])


def get_video_duration_s(video_path):
    """True video duration from container metadata (fps * frame count), used
    as the completeness-check denominator for the session-wide OpenFace run.
    Independent of OpenFace's own output -- checking a truncated OpenFace csv
    against a duration derived from that same (possibly truncated) csv would
    be circular and could never detect the truncation."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if not fps or not n_frames:
        raise RuntimeError(f"Could not read video metadata (fps={fps}, frames={n_frames}) from {video_path}")
    return n_frames / fps


def compute_task_video_offset(task_windows_path, obs_log_path, task):
    """Video t=0 is the OBS 'START RECORDING' event, a different clock
    reference than PsychoPy's expStart used in task_windows.csv."""
    windows = pd.read_csv(task_windows_path)
    row = windows[windows["task"] == task].iloc[0]
    task_start_abs = datetime.fromisoformat(row["start_abs_iso"])

    with open(obs_log_path, encoding="utf-8") as f:
        content = f.read()
    start_line = next(l for l in content.splitlines() if l.startswith("EVENT:START RECORDING"))
    ts_str = start_line.split("@", 1)[1].strip()
    obs_start_naive = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    obs_start_abs = obs_start_naive.replace(tzinfo=task_start_abs.tzinfo)

    offset_s = (task_start_abs - obs_start_abs).total_seconds()
    return offset_s, float(row["duration_s"])


def cut_clip(ffmpeg_exe, video_path, start_s, duration_s, out_path):
    cmd = [ffmpeg_exe, "-y", "-ss", str(start_s), "-i", video_path, "-t", str(duration_s),
           "-an", "-c", "copy", out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed cutting clip: {result.stderr[-1000:]}")


# OpenFace is failing silently, it can exit cleanly while
# having written only a fraction of the expected frames. A cached file
# this short must never be reused, so completeness is checked by row
# count.
MIN_COMPLETENESS_RATIO = 0.95
NOMINAL_FPS = 60
MAX_OPENFACE_ATTEMPTS = 3
RETRY_DELAY_S = 30


def expected_frame_count(duration_s):
    return duration_s * NOMINAL_FPS


def check_completeness(df, duration_s, participant, task):
    expected = expected_frame_count(duration_s)
    ratio = len(df) / expected if expected else 0
    if ratio < MIN_COMPLETENESS_RATIO:
        return False, (f"{participant}/{task}: OpenFace output has {len(df)} rows but the {duration_s:.1f}s "
                        f"clip should yield ~{expected:.0f} frames ({ratio*100:.1f}% -- below the "
                        f"{MIN_COMPLETENESS_RATIO*100:.0f}% completeness floor). Likely a silent OpenFace "
                        f"failure (OOM or otherwise) that still exited cleanly.")
    return True, None


def run_openface(openface_exe, clip_path, out_dir, timeout_s=None):
    os.makedirs(out_dir, exist_ok=True)
    openface_dir = os.path.dirname(openface_exe)
    try:
        result = subprocess.run(
            [openface_exe, "-f", os.path.abspath(clip_path), "-out_dir", os.path.abspath(out_dir),
             "-fx", str(CAMERA_FX), "-fy", str(CAMERA_FY), "-cx", str(CAMERA_CX), "-cy", str(CAMERA_CY),
             "-2Dfp", "-3Dfp", "-pdmparams", "-pose", "-aus", "-gaze"],
            cwd=openface_dir, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        # The process finished writing a complete file but never exited.
        # Rather than discarding a long run, check whether it actually
        # finished before giving up on it.
        csv_candidates = glob.glob(os.path.join(out_dir, "*.csv"))
        if csv_candidates:
            df = pd.read_csv(csv_candidates[0])
            df.columns = [c.strip() for c in df.columns]
            return df
        raise RuntimeError(f"FeatureExtraction.exe did not exit within {timeout_s}s (killed) -- "
                            f"no output csv was found, so this looks like a genuine stall, not just a hang-after-finish.")
    if result.returncode != 0:
        raise RuntimeError(f"FeatureExtraction.exe exited {result.returncode}: {result.stderr[-1000:]}")
    csv_candidates = glob.glob(os.path.join(out_dir, "*.csv"))
    if not csv_candidates:
        raise RuntimeError("FeatureExtraction.exe ran but produced no output csv.")
    df = pd.read_csv(csv_candidates[0])
    df.columns = [c.strip() for c in df.columns]
    return df


def run_openface_with_retries(openface_exe, clip_path, out_dir, duration_s, participant, task):
    last_error = None
    # Multiple of the rough throughput estimate, to catch genuine
    # hangs without cutting off a legitimately slow run under memory pressure.
    timeout_s = max(900.0, duration_s * OPENFACE_SECONDS_PER_FRAME_ESTIMATE * NOMINAL_FPS * 4)
    for attempt in range(1, MAX_OPENFACE_ATTEMPTS + 1):
        try:
            df = run_openface(openface_exe, clip_path, out_dir, timeout_s=timeout_s)
            ok, msg = check_completeness(df, duration_s, participant, task)
            if ok:
                return df
            last_error = msg
            print(f"  [{task}] attempt {attempt}/{MAX_OPENFACE_ATTEMPTS} produced an incomplete output: {msg}")
        except Exception as e:
            last_error = str(e)
            print(f"  [{task}] attempt {attempt}/{MAX_OPENFACE_ATTEMPTS} failed: {e}")

        # Clean up any partial file before retrying, so no mistakes
        # on its own previous incomplete attempt.
        for f in glob.glob(os.path.join(out_dir, "*")):
            os.remove(f)
        if attempt < MAX_OPENFACE_ATTEMPTS:
            print(f"  [{task}] retrying in {RETRY_DELAY_S}s ...")
            time.sleep(RETRY_DELAY_S)

    raise RuntimeError(f"OpenFace failed to produce a complete output for {participant}/{task} after "
                        f"{MAX_OPENFACE_ATTEMPTS} attempts. Last error: {last_error}")


CAMERA_PARAMS_MARKER_FILENAME = "camera_params_used.json"


def _write_camera_params_marker(out_dir):
    import json
    with open(os.path.join(out_dir, CAMERA_PARAMS_MARKER_FILENAME), "w") as f:
        json.dump({"fx": CAMERA_FX, "fy": CAMERA_FY, "cx": CAMERA_CX, "cy": CAMERA_CY}, f)


def _cached_camera_params_match(out_dir):
    """A cached OpenFace output is only trustworthy to reuse if it was
    produced with the SAME camera calibration params this run would use --
    otherwise reusing it silently mixes calibrated and uncalibrated data
    under one cache path. Missing marker (e.g. a cache from before
    calibration became mandatory, 2026-09-07) counts as a mismatch, not a
    pass -- fail closed, not open."""
    marker_path = os.path.join(out_dir, CAMERA_PARAMS_MARKER_FILENAME)
    if not os.path.exists(marker_path):
        return False
    import json
    with open(marker_path) as f:
        params = json.load(f)
    return params == {"fx": CAMERA_FX, "fy": CAMERA_FY, "cx": CAMERA_CX, "cy": CAMERA_CY}


def load_or_run_openface_session_wide(args, video_path):
    """Run OpenFace ONCE on the full, uncut session video (or reuse a cached
    prior run) -- see the session-wide decision note in the module
    docstring. Returns (full_df, session_duration_s, of_elapsed)."""
    session_duration_s = get_video_duration_s(video_path)

    of_out_dir = os.path.join(args.openface_out_dir, f"{args.participant}_full_session")
    existing_csv = glob.glob(os.path.join(of_out_dir, "*.csv")) if os.path.isdir(of_out_dir) else []
    of_elapsed = None
    reuse_cache = bool(existing_csv) and not args.force_rerun_openface and args.max_clip_seconds is None

    if reuse_cache and not _cached_camera_params_match(of_out_dir):
        print(f"  Cached output at {of_out_dir} was produced with different (or no recorded) camera "
              f"calibration params than the current fx={CAMERA_FX} fy={CAMERA_FY} cx={CAMERA_CX} "
              f"cy={CAMERA_CY} -- treating as stale and re-running rather than silently mixing "
              f"calibrated/uncalibrated data.")
        reuse_cache = False

    if reuse_cache:
        df = pd.read_csv(existing_csv[0])
        df.columns = [c.strip() for c in df.columns]
        ok, msg = check_completeness(df, session_duration_s, args.participant, "full_session")
        if not ok:
            print(f"  WARNING: cached session-wide OpenFace output failed the completeness check -- "
                  f"discarding and re-running. {msg}")
            for f in glob.glob(os.path.join(of_out_dir, "*")):
                os.remove(f)
            reuse_cache = False
        else:
            print(f"  Reusing existing session-wide OpenFace output: {existing_csv[0]} "
                  f"({len(df)} rows, passed completeness check, matching camera params) "
                  f"(pass --force-rerun-openface to redo the ~3-4h extraction instead)")

    if reuse_cache:
        pass  # df already loaded and validated above
    else:
        run_path = video_path
        run_duration_s = session_duration_s
        temp_clip = None
        if args.max_clip_seconds is not None:
            # Smoke-test path, process only a prefix of the session so the
            # plumbing can be tested in seconds instead of hours.
            print(f"  --max-clip-seconds set: cutting a {args.max_clip_seconds:.0f}s prefix of the session "
                  f"for a smoke test (NOT a real run)")
            import imageio_ffmpeg
            temp_clip = os.path.join(args.clip_dir, f"{args.participant}_full_session_smoketest.mp4")
            cut_clip(imageio_ffmpeg.get_ffmpeg_exe(), video_path, 0.0, args.max_clip_seconds, temp_clip)
            run_path = temp_clip
            run_duration_s = args.max_clip_seconds

        est_frames = run_duration_s * 60  # nominal 60fps source
        est_openface_s = est_frames * OPENFACE_SECONDS_PER_FRAME_ESTIMATE
        print(f"  Session duration: {session_duration_s:.1f}s ({session_duration_s/60:.1f} min)")
        print(f"  Estimated OpenFace runtime: ~{est_openface_s/60:.1f} min ({est_frames:.0f} frames @ ~"
              f"{1/OPENFACE_SECONDS_PER_FRAME_ESTIMATE:.1f} fps effective throughput)")

        print(f"  Running OpenFace FeatureExtraction.exe on the full session video ...")
        t_of_start = time.time()
        df = run_openface_with_retries(args.openface_exe, run_path, of_out_dir, run_duration_s,
                                        args.participant, "full_session")
        of_elapsed = time.time() - t_of_start
        print(f"  OpenFace done: {len(df)} frames in {of_elapsed:.1f}s "
              f"({of_elapsed/max(len(df),1):.4f} s/frame)")
        _write_camera_params_marker(of_out_dir)

        if temp_clip is not None and not args.keep_clips:
            os.remove(temp_clip)

    return df, session_duration_s, of_elapsed


def slice_task_window(full_df, offset_s, window_duration_s):
    mask = (full_df["timestamp"] >= offset_s) & (full_df["timestamp"] < offset_s + window_duration_s)
    df_task = full_df.loc[mask].reset_index(drop=True)
    return df_task


def cache_task_slice(df_task, openface_out_dir, participant, task):
    """Materialize a task-window slice of the session-wide OpenFace output as
    its own OpenFace-format CSV, at the same {participant}_{task}/ path the
    OLD per-clip cache used to live at. This is purely so other tooling
    (e.g. 04_validate_features_blink_overlap.py) that reads that directory
    layout keeps working unmodified on session-wide-normalized data -- it is
    a re-slice of the ONE session-wide OpenFace run, not an independent
    OpenFace invocation."""
    task_dir = os.path.join(openface_out_dir, f"{participant}_{task}")
    os.makedirs(task_dir, exist_ok=True)
    for f in glob.glob(os.path.join(task_dir, "*.csv")):
        os.remove(f)
    out_path = os.path.join(task_dir, f"{participant}_{task}.csv")
    df_task.to_csv(out_path, index=False)
    return out_path


def compute_task_features(df, participant, task, window_duration_s, min_duration_ms, max_duration_ms):
    frame_time_s = float(np.median(np.diff(df["timestamp"].to_numpy())))
    duration_s = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]

    blink_mask = compute_blink_exclusion_mask(df["AU45_c"].to_numpy(), frame_time_s,
                                               BLINK_FILTER_MARGIN_FRAMES, min_duration_ms, max_duration_ms)
    low_conf_mask = (df["confidence"] < CONFIDENCE_EXCLUSION_THRESHOLD).to_numpy()
    exclude_mask = blink_mask | low_conf_mask
    df_clean = df.loc[~exclude_mask]

    row = {
        "participant": participant,
        "task": task,
        "n_frames": len(df),
        "video_duration_s": duration_s,
        "task_window_duration_s": window_duration_s,
        "confidence_mean": df["confidence"].mean(),
        "success_rate": df["success"].mean(),
        "blink_threshold_min_ms": min_duration_ms,
        "blink_threshold_max_ms": max_duration_ms,
        "n_frames_excluded_blink": int(blink_mask.sum()),
        "n_frames_excluded_low_confidence": int(low_conf_mask.sum()),
        "n_frames_excluded_total": int(exclude_mask.sum()),
        "pct_frames_excluded_blink": blink_mask.mean() * 100,
        "pct_frames_excluded_low_confidence": low_conf_mask.mean() * 100,
        "pct_frames_excluded_total": exclude_mask.mean() * 100,
        "n_frames_used": len(df_clean),
    }
    print(f"    [{task}] excluded {row['n_frames_excluded_total']}/{len(df)} frames "
          f"({row['pct_frames_excluded_total']:.1f}%): {row['n_frames_excluded_blink']} blink-adjacent "
          f"(+/-{BLINK_FILTER_MARGIN_FRAMES}f, {min_duration_ms:.0f}-{max_duration_ms:.0f}ms), "
          f"{row['n_frames_excluded_low_confidence']} low-confidence (<{CONFIDENCE_EXCLUSION_THRESHOLD})")

    if len(df_clean) < 30:
        print(f"    WARNING [{task}]: only {len(df_clean)} frames survive filtering -- "
              f"AU/gaze/pose statistics below may be unreliable.")

    for au in ("AU04_r", "AU07_r"):
        s = df_clean[au]
        prefix = au.lower()
        row[f"{prefix}_mean"] = s.mean()
        row[f"{prefix}_median"] = s.median()
        row[f"{prefix}_std"] = s.std()
        row[f"{prefix}_skew"] = s.skew()
        row[f"{prefix}_kurtosis"] = s.kurt()

    # Blink rate is measured over the blinks themselves, so it uses the
    # FULL (unfiltered) AU45_c signal, not df_clean.
    durations_ms = au45_blink_run_durations_ms(df["AU45_c"].to_numpy(), frame_time_s)
    n_raw = len(durations_ms)
    thresholded = durations_ms[(durations_ms >= min_duration_ms) & (durations_ms <= max_duration_ms)]
    n_thresholded = len(thresholded)
    row["blink_n_events_raw"] = n_raw
    row["blink_n_events_thresholded"] = n_thresholded
    row["blink_rate_per_min_raw"] = n_raw / duration_s * 60
    row["blink_rate_per_min"] = n_thresholded / duration_s * 60

    # gaze_1_z excluded from aggregation -- see EXCLUDED_CONTAMINATED_FEATURES.
    # Still present in df/df_clean (and the raw cached OpenFace csv) since
    # only the AGGREGATED feature table omits it.
    gaze_cols = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z",
                 "gaze_angle_x", "gaze_angle_y"]
    for col in gaze_cols:
        if col in EXCLUDED_CONTAMINATED_FEATURES:
            continue
        row[f"{col}_mean"] = df_clean[col].mean()
        row[f"{col}_std"] = df_clean[col].std()

    # OpenFace convention: pose_Rx = pitch, pose_Ry = yaw, pose_Rz = roll.
    # pose_Rx excluded from aggregation -- see EXCLUDED_CONTAMINATED_FEATURES.
    pose_map = {"pose_Rx": "pose_pitch", "pose_Ry": "pose_yaw", "pose_Rz": "pose_roll"}
    for src, label in pose_map.items():
        if src in EXCLUDED_CONTAMINATED_FEATURES:
            continue
        row[f"{label}_mean"] = df_clean[src].mean()
        row[f"{label}_std"] = df_clean[src].std()

    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("participant", help="Participant id, e.g. p20")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--video-dir", default=os.path.join("data", "raw", "video"))
    parser.add_argument("--windows-dir", default=os.path.join("data", "processed", "synced"))
    parser.add_argument("--obs-log-dir", default=os.path.join("data", "raw", "obs_logs"))
    parser.add_argument("--openface-exe", required=True,
                         help="Path to FeatureExtraction.exe from an extracted OpenFace_2.2.0_win_x64 build")
    parser.add_argument("--clip-dir", default=os.path.join("data", "processed", "video", "task_clips"),
                         help="Only used for the --max-clip-seconds smoke-test path (session-wide "
                              "processing no longer cuts per-task clips).")
    parser.add_argument("--openface-out-dir", default=os.path.join("data", "processed", "video", "openface_raw_session_wide"),
                         help="Session-wide OpenFace run is cached at {this}/{participant}_full_session/; "
                              "task-window slices of it are additionally materialized at "
                              "{this}/{participant}_{task}/ for tools expecting the old per-task layout.")
    parser.add_argument("--out-dir", default=os.path.join("data", "processed", "video"))
    parser.add_argument("--max-clip-seconds", type=float, default=None,
                         help="Smoke test only: process just a prefix of the session instead of the "
                              "whole thing (omit for the real run).")
    parser.add_argument("--keep-clips", action="store_true", help="Keep the smoke-test mp4 clip (large!); default deletes it after processing")
    parser.add_argument("--force-rerun-openface", action="store_true",
                         help="Re-run FeatureExtraction.exe on the full session even if a cached output csv "
                              "already exists for this participant (default: reuse it, since "
                              "OpenFace is the expensive ~3-4h step and re-running it is only needed "
                              "after a real data/config change, not a feature-aggregation change).")
    args = parser.parse_args()

    os.makedirs(args.clip_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    video_path = find_file(args.video_dir, args.participant, must_contain=".mp4")
    windows_path = os.path.join(args.windows_dir, f"{args.participant}_task_windows.csv")
    obs_log_path = find_file(args.obs_log_dir, args.participant)

    print(f"Video: {video_path}")
    print(f"Task windows: {windows_path}")
    print(f"OBS log: {obs_log_path}")
    print(f"OpenFace: {args.openface_exe}")

    t_script_start = time.time()

    # Pass 1: run OpenFace on the full session, slice out the 3 task
    # windows, then pool blink data across them to derive the adaptive
    # threshold before any feature aggregation happens.
    print("\n=== Pass 1/2: session-wide OpenFace, then slicing task windows ===")
    full_df, session_duration_s, of_elapsed = load_or_run_openface_session_wide(args, video_path)
    print(f"  Session-wide output: {len(full_df)} frames, timestamp range "
          f"{full_df['timestamp'].min():.1f}-{full_df['timestamp'].max():.1f}s")

    task_data = {}
    all_durations_ms = []
    frame_times_ms = []
    for task in args.tasks:
        offset_s, window_duration_s = compute_task_video_offset(windows_path, obs_log_path, task)
        df = slice_task_window(full_df, offset_s, window_duration_s)
        if len(df) == 0:
            raise ValueError(f"No frames found for task '{task}' in window "
                              f"[{offset_s:.1f}, {offset_s + window_duration_s:.1f}]s of the session-wide "
                              f"output -- check offset/sync (video may not cover this task window).")
        cache_path = cache_task_slice(df, args.openface_out_dir, args.participant, task)
        print(f"  [{task}] window {offset_s:.1f}-{offset_s + window_duration_s:.1f}s -> {len(df)} frames "
              f"(cached: {cache_path})")

        frame_time_s = float(np.median(np.diff(df["timestamp"].to_numpy())))
        durations_ms = au45_blink_run_durations_ms(df["AU45_c"].to_numpy(), frame_time_s)
        all_durations_ms.extend(durations_ms.tolist())
        frame_times_ms.append(frame_time_s * 1000)
        task_data[task] = {"df": df, "window_duration_s": window_duration_s, "openface_elapsed_s": of_elapsed}

    frame_time_ms = float(np.median(frame_times_ms))
    threshold_info = derive_adaptive_blink_thresholds(all_durations_ms, frame_time_ms)
    min_duration_ms = threshold_info["min_duration_ms"]
    max_duration_ms = threshold_info["max_duration_ms"]

    print(f"\nAdaptive blink-duration threshold for {args.participant} "
          f"(pooled across {args.tasks}, n={threshold_info['n_events']} AU45 episodes):")
    print(f"  min={min_duration_ms:.1f}ms (fixed 2-frame floor), max={max_duration_ms:.1f}ms "
          f"(Tukey Q3+1.5*IQR: Q1={threshold_info.get('q1_ms')}, Q3={threshold_info.get('q3_ms')}, "
          f"IQR={threshold_info.get('iqr_ms')})")
    if threshold_info["fallback_reason"]:
        print(f"  NOTE: {threshold_info['fallback_reason']}")

    # Glasses/AU04-bias check: a near-constant raw AU04 offset that a
    # temporal filter can't fix, since it isn't isolated bad frames
    # (CHECK this).
    all_au04_raw = np.concatenate([task_data[t]["df"]["AU04_r"].to_numpy() for t in args.tasks])
    au04_median_raw = float(np.median(all_au04_raw))
    likely_glasses_bias = au04_median_raw > GLASSES_AU04_MEDIAN_THRESHOLD
    print(f"\nGlasses/AU04-bias check: pooled raw AU04_r median = {au04_median_raw:.3f} "
          f"(threshold {GLASSES_AU04_MEDIAN_THRESHOLD}) -> "
          f"{'FLAGGED -- likely glasses or similar occlusion bias, see brow-region AUs (01/02/04) with caution' if likely_glasses_bias else 'OK'}")

    # --- Pass 2: compute features per task using the shared threshold. ---
    print("\n=== Pass 2/2: computing features per task ===")
    feature_rows = []
    for i, task in enumerate(args.tasks, 1):
        print(f"\n[{i}/{len(args.tasks)}] === Task '{task}' ===")
        d = task_data[task]
        row = compute_task_features(d["df"], args.participant, task, d["window_duration_s"],
                                     min_duration_ms, max_duration_ms)
        row["openface_elapsed_s"] = d["openface_elapsed_s"]
        row["blink_threshold_n_events_pooled"] = threshold_info["n_events"]
        row["au04_median_raw_pooled"] = au04_median_raw
        row["likely_glasses_au_bias"] = likely_glasses_bias
        row["extraction_mode"] = "session_wide"
        row["session_duration_s"] = session_duration_s
        feature_rows.append(row)
        print(f"  confidence_mean={row['confidence_mean']:.3f}  "
              f"au04_r_mean={row['au04_r_mean']:.4f}  au07_r_mean={row['au07_r_mean']:.4f}  "
              f"blink_rate={row['blink_rate_per_min']:.1f}/min (raw {row['blink_rate_per_min_raw']:.1f}/min)")

    features_df = pd.DataFrame(feature_rows)
    out_path = os.path.join(args.out_dir, f"{args.participant}_video_features.csv")
    features_df.to_csv(out_path, index=False)

    print(f"\nSaved video feature table: {out_path}")
    print(f"  {features_df.shape[0]} rows (tasks) x {features_df.shape[1]} columns")
    print(f"  Total elapsed: {(time.time()-t_script_start)/60:.1f} min")

    key_cols = ["participant", "task", "n_frames", "confidence_mean", "pct_frames_excluded_total",
                "au04_r_mean", "au04_r_std", "au07_r_mean", "au07_r_std",
                "blink_rate_per_min", "pose_pitch_mean", "pose_yaw_mean", "pose_roll_mean"]
    key_cols = [c for c in key_cols if c in features_df.columns]
    print("\n=== Summary (key features) ===")
    print(features_df[key_cols].to_string(index=False))
    print(f"\n(Full feature table has {features_df.shape[1]} columns — see {out_path} for everything, "
          "including per-gaze-column mean/std.)")


if __name__ == "__main__":
    main()
