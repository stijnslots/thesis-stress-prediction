"""
Compares three facial-video-analysis tools (OpenFace 2.0, LibreFace,
Py-Feat) on a short clip of the Stroop task, to decide which is
practically usable for the video pipeline.

Needed features: Action Units, gaze direction, head pose, blink rate.

What this script does:
    1. Cuts a short clip from the start of the Stroop task window.
    2. Runs OpenFace's FeatureExtraction.exe on the clip.
    3. Attempts LibreFace on the same clip, documenting the failure rather
       than silently skipping it if it doesn't run.
    4. Runs Py-Feat's detector on the same clip, frame by frame.
    5. Reports available feature families and processing speed for
       whichever tools ran successfully.

Usage:
    python 01_compare_facial_tools_p20.py
    python 01_compare_facial_tools_p20.py --clip-seconds 45 --openface-exe "C:/path/to/FeatureExtraction.exe"
"""

import argparse
import glob
import os
import shutil
import subprocess
import time
from datetime import datetime

import pandas as pd


def compute_clip_start_offset(task_windows_path, obs_log_path, task):
    """Video t=0 is the OBS 'START RECORDING' event, which is a different
    clock reference than PsychoPy's expStart. Convert the task-window start
    (absolute, from task_windows.csv) into an offset into the mp4 file."""
    windows = pd.read_csv(task_windows_path)
    row = windows[windows["task"] == task].iloc[0]
    task_start_abs = datetime.fromisoformat(row["start_abs_iso"])

    with open(obs_log_path, encoding="utf-8") as f:
        content = f.read()
    start_line = next(l for l in content.splitlines() if l.startswith("EVENT:START RECORDING"))
    # e.g. "EVENT:START RECORDING @ 2026-03-18 13:57:23"
    ts_str = start_line.split("@", 1)[1].strip()
    obs_start_naive = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    obs_start_abs = obs_start_naive.replace(tzinfo=task_start_abs.tzinfo)

    return (task_start_abs - obs_start_abs).total_seconds()


def cut_clip(ffmpeg_exe, video_path, start_s, duration_s, out_path):
    cmd = [ffmpeg_exe, "-y", "-ss", str(start_s), "-i", video_path, "-t", str(duration_s),
           "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", out_path]
    subprocess.run(cmd, capture_output=True, text=True, check=True)


def extract_frames(ffmpeg_exe, video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [ffmpeg_exe, "-y", "-i", video_path, os.path.join(out_dir, "frame_%04d.png")]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return sorted(glob.glob(os.path.join(out_dir, "*.png")))


def run_openface(openface_exe, clip_path, out_dir):
    if not openface_exe or not os.path.exists(openface_exe):
        return {"ok": False, "reason": f"FeatureExtraction.exe not found at '{openface_exe}'. "
                "Download OpenFace_2.2.0_win_x64.zip from "
                "github.com/TadasBaltrusaitis/OpenFace/releases and pass --openface-exe."}
    os.makedirs(out_dir, exist_ok=True)
    openface_dir = os.path.dirname(openface_exe)
    t0 = time.time()
    result = subprocess.run(
        [openface_exe, "-f", os.path.abspath(clip_path), "-out_dir", os.path.abspath(out_dir),
         "-2Dfp", "-3Dfp", "-pdmparams", "-pose", "-aus", "-gaze"],
        cwd=openface_dir, capture_output=True, text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        return {"ok": False, "reason": f"FeatureExtraction.exe exited {result.returncode}: {result.stderr[-500:]}"}

    csv_candidates = glob.glob(os.path.join(out_dir, "*.csv"))
    if not csv_candidates:
        return {"ok": False, "reason": "FeatureExtraction.exe ran but produced no output csv."}
    df = pd.read_csv(csv_candidates[0])
    df.columns = [c.strip() for c in df.columns]
    return {"ok": True, "elapsed_s": elapsed, "df": df, "csv_path": csv_candidates[0]}


def openface_blink_rate(df, duration_s):
    au45 = df["AU45_c"]
    blink_events = int(((au45.shift(1) == 0) & (au45 == 1)).sum())
    return blink_events, blink_events / duration_s * 60


def run_libreface(clip_path, work_dir):
    """Documents the expected failure rather than hiding it: libreface's
    pinned mediapipe==0.10.5 has no Python 3.12 wheel, and every mediapipe
    version that DOES install on 3.12 (checked: 0.10.13 through 1.0.1) has
    already dropped the `mp.solutions.face_mesh` API libreface's code
    imports, so face detection silently returns zero faces on every frame."""
    try:
        import mediapipe as mp
        has_legacy_api = hasattr(mp, "solutions")
    except ImportError:
        return {"ok": False, "reason": "mediapipe not installed."}

    if not has_legacy_api:
        return {
            "ok": False,
            "reason": (
                f"mediapipe {mp.__version__} is installed (required for libreface's face alignment) "
                "but no longer exposes `mp.solutions.face_mesh`, which libreface's code depends on. "
                "libreface pins mediapipe==0.10.5, which has no Windows/Python-3.12 wheel; every "
                "installable version on this system (0.10.13 - 1.0.1) has already removed that API. "
                "Result: libreface runs without crashing but detects zero faces on every frame."
            ),
        }

    try:
        import libreface
    except ImportError:
        return {"ok": False, "reason": "libreface not installed (pip install libreface)."}

    os.makedirs(work_dir, exist_ok=True)
    t0 = time.time()
    try:
        result = libreface.get_facial_attributes_video(
            clip_path, temp_dir=os.path.join(work_dir, "tmp"), device="cpu",
            weights_download_dir=os.path.join(work_dir, "weights"),
        )
        return {"ok": True, "elapsed_s": time.time() - t0, "df": result}
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}", "elapsed_s": time.time() - t0}


def run_pyfeat(frame_paths, out_csv):
    from feat import Detectorv1

    t0 = time.time()
    detector = Detectorv1(device="cpu")
    init_elapsed = time.time() - t0

    t1 = time.time()
    df = detector.detect(frame_paths, data_type="image", batch_size=8, progress_bar=False)
    detect_elapsed = time.time() - t1

    df.to_csv(out_csv, index=False)
    return {"ok": True, "init_elapsed_s": init_elapsed, "detect_elapsed_s": detect_elapsed, "df": df}


def pyfeat_blink_rate(df, duration_s):
    """Py-Feat's AU model (xgb, DISFA-trained) does not include AU45 (blink)
    in its output set — see the comparison doc. Blink rate is therefore not
    directly derivable from Py-Feat's AU columns the way it is from
    OpenFace's AU45_c; it would require a separate eye-aspect-ratio
    calculation from the landmark columns, which is out of scope here."""
    au_cols = [c for c in df.columns if c.startswith("AU")]
    has_au45 = any("45" in c for c in au_cols)
    return has_au45, au_cols


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--participant", default="p20")
    parser.add_argument("--task", default="stroop")
    parser.add_argument("--clip-seconds", type=float, default=45.0)
    parser.add_argument("--video-path", default=os.path.join("data", "raw", "video", "p20.mp4"))
    parser.add_argument("--windows-path", default=os.path.join("data", "processed", "synced", "p20_task_windows.csv"))
    parser.add_argument("--obs-log-path", default=os.path.join("data", "raw", "obs_logs", "p20"))
    parser.add_argument("--clip-out", default=os.path.join("data", "processed", "video", "p20_stroop_clip_45s.mp4"))
    parser.add_argument("--openface-exe", default=None,
                         help="Path to FeatureExtraction.exe from an extracted OpenFace_2.2.0_win_x64 build")
    parser.add_argument("--ffmpeg-exe", default=None, help="Defaults to imageio_ffmpeg's bundled binary")
    parser.add_argument("--out-dir", default=os.path.join("results", "video_exploration"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ffmpeg_exe = args.ffmpeg_exe
    if ffmpeg_exe is None:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    print(f"Computing clip offset for task '{args.task}' ...")
    offset_s = compute_clip_start_offset(args.windows_path, args.obs_log_path, args.task)
    print(f"  video offset: {offset_s:.2f}s, clip length: {args.clip_seconds}s")

    if not os.path.exists(args.clip_out):
        print(f"Cutting clip -> {args.clip_out}")
        cut_clip(ffmpeg_exe, args.video_path, offset_s, args.clip_seconds, args.clip_out)
    else:
        print(f"Clip already exists, reusing: {args.clip_out}")

    results = {}

    print("\n=== OpenFace 2.0 ===")
    of_out_dir = os.path.join(args.out_dir, "openface_out")
    of_result = run_openface(args.openface_exe, args.clip_out, of_out_dir)
    results["openface"] = of_result
    if of_result["ok"]:
        n_blinks, blink_rate = openface_blink_rate(of_result["df"], args.clip_seconds)
        of_result["n_blinks"] = n_blinks
        of_result["blink_rate_per_min"] = blink_rate
        print(f"  OK: {of_result['elapsed_s']:.1f}s, {len(of_result['df'])} frames, "
              f"{n_blinks} blinks -> {blink_rate:.1f}/min")
    else:
        print(f"  FAILED: {of_result['reason']}")

    print("\n=== LibreFace ===")
    lf_result = run_libreface(args.clip_out, os.path.join(args.out_dir, "libreface_work"))
    results["libreface"] = lf_result
    if lf_result["ok"]:
        print(f"  OK: {lf_result['elapsed_s']:.1f}s, {len(lf_result['df'])} rows")
    else:
        print(f"  FAILED: {lf_result['reason']}")

    print("\n=== Py-Feat ===")
    frames_dir = os.path.join(args.out_dir, "pyfeat_frames")
    print("  Extracting frames for image-mode processing (avoids the torchcodec/shared-FFmpeg dependency)...")
    frame_paths = extract_frames(ffmpeg_exe, args.clip_out, frames_dir)
    print(f"  {len(frame_paths)} frames extracted")
    pf_out_csv = os.path.join(args.out_dir, "pyfeat_out.csv")
    try:
        pf_result = run_pyfeat(frame_paths, pf_out_csv)
        has_au45, au_cols = pyfeat_blink_rate(pf_result["df"], args.clip_seconds)
        pf_result["has_au45"] = has_au45
        pf_result["au_cols"] = au_cols
        results["pyfeat"] = pf_result
        print(f"  OK: init {pf_result['init_elapsed_s']:.1f}s + detect {pf_result['detect_elapsed_s']:.1f}s "
              f"for {len(frame_paths)} frames, AU45 in output: {has_au45}")
    except Exception as e:
        results["pyfeat"] = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
        print(f"  FAILED: {results['pyfeat']['reason']}")

    print("\nDone. See results/video_exploration/tool_comparison.md for the write-up of these results.")
    return results


if __name__ == "__main__":
    main()
