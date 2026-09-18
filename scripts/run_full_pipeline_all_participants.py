"""
Runs the full pipeline overnight for the whole participant cohort: task
windows, PPG features, labels, and (once video exists) the full video
pipeline, then watches for new video files arriving later and processes
those automatically.

Phases:
    A. For every participant with PsychoPy and OBS-log data: task windows,
       PPG features, labels, and a partial master table (video columns
       left empty until video is available).
    B. For every participant that already has video: the full video
       pipeline and a complete master table.
    C. Watches the video folder for new files while the script keeps
       running, and processes each one as it appears.

Every step is wrapped individually: a failure is logged and the batch
moves on to the next participant or step rather than stopping, and the
status overview is rewritten after every participant so it stays current.

Usage:
    python scripts/run_full_pipeline_all_participants.py
    python scripts/run_full_pipeline_all_participants.py --watch-hours 6
"""

import argparse
import os
import re
import subprocess
import sys
import time
import traceback

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(REPO_ROOT)

PYTHON = sys.executable
OPENFACE_EXE = os.path.join(REPO_ROOT, "scratchpad_video_tools", "OpenFace_2.2.0_win_x64",
                             "OpenFace_2.2.0_win_x64", "FeatureExtraction.exe")

LOG_PATH = os.path.join("results", "full_pipeline_run_log.txt")
REPORT_PATH = os.path.join("results", "pipeline_status_overview.md")

PSYCHOPY_DIR = os.path.join("data", "raw", "psychopy")
PPG_DIR = os.path.join("data", "raw", "ppg")
VIDEO_DIR = os.path.join("data", "raw", "video")
OBS_DIR = os.path.join("data", "raw", "obs_logs")

_log_file = open(LOG_PATH, "a", encoding="utf-8")


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _log_file.write(line + "\n")
    _log_file.flush()


def discover_participants():
    def ids_from(d, pattern):
        ids = set()
        if not os.path.isdir(d):
            return ids
        for f in os.listdir(d):
            m = re.match(pattern, f, re.IGNORECASE)
            if m:
                ids.add(m.group(1).lower())
        return ids

    psychopy_ids = ids_from(PSYCHOPY_DIR, r"(p\d+)_")
    ppg_ids = ids_from(PPG_DIR, r"(p\d+)_")
    video_ids = ids_from(VIDEO_DIR, r"(p\d+)\.mp4")
    obs_ids = set(f.lower() for f in os.listdir(OBS_DIR)) if os.path.isdir(OBS_DIR) else set()
    return psychopy_ids, ppg_ids, video_ids, obs_ids


def run(cmd, timeout=3 * 3600):
    t0 = time.time()
    try:
        proc = subprocess.run([PYTHON] + cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            return False, (proc.stderr or proc.stdout or "")[-3000:], elapsed
        return True, proc.stdout[-1500:], elapsed
    except subprocess.TimeoutExpired:
        return False, f"TimeoutExpired after {time.time()-t0:.0f}s", time.time() - t0
    except Exception as e:
        return False, f"{e}\n{traceback.format_exc()}", time.time() - t0


STATE = {}   # participant -> dict of step statuses/info
ERRORS = []  # list of {participant, step, error}


def record_error(p, step, error):
    ERRORS.append({"participant": p, "step": step, "error": error})
    log(f"  [{p}] ERROR in '{step}': {error.splitlines()[0] if error else '(no message)'}")


def ensure_state(p):
    if p not in STATE:
        STATE[p] = {"steps": {}, "video_status": "no_video"}
    return STATE[p]


def phase_a_participant(p, has_ppg):
    st = ensure_state(p)

    ok, info, elapsed = run([os.path.join("scripts", "sync", "extract_task_windows.py"), p])
    st["steps"]["task_windows"] = f"OK ({elapsed:.0f}s)" if ok else "FAILED"
    if not ok:
        record_error(p, "task_windows", info)
        return

    if has_ppg:
        # Pin to the untrimmed output filename that the merge step expects,
        # trimming hasn't been adopted as the pipeline's default yet
        # (CHECK this).
        ok, info, elapsed = run([os.path.join("scripts", "ppg", "03_extract_ppg_features.py"), p,
                                  "--trim-seconds", "0"])
        st["steps"]["ppg_features"] = f"OK ({elapsed:.0f}s)" if ok else "FAILED"
        if not ok:
            record_error(p, "ppg_features", info)
    else:
        st["steps"]["ppg_features"] = "SKIPPED (no PPG file found)"
        log(f"  [{p}] SKIPPED ppg_features: no PPG file found in {PPG_DIR}")

    ok, info, elapsed = run([os.path.join("scripts", "labels", "01_extract_stress_labels.py"), p])
    st["steps"]["labels"] = f"OK ({elapsed:.0f}s)" if ok else "FAILED"
    if not ok:
        record_error(p, "labels", info)
        return

    if not has_ppg:
        st["steps"]["merge"] = "SKIPPED (no PPG features)"
        return

    merge_cmd = [os.path.join("scripts", "features", "01_merge_features.py"), p, "--allow-missing-video"]
    ok, info, elapsed = run(merge_cmd)
    st["steps"]["merge"] = f"OK partial (video_pending) ({elapsed:.0f}s)" if ok else "FAILED"
    if not ok:
        record_error(p, "merge_partial", info)
    else:
        st["video_status"] = "pending"


def phase_b_participant(p):
    """Full video pipeline + complete merge. Safe to call repeatedly --
    02_extract_video_features.py reuses cached OpenFace output."""
    st = ensure_state(p)
    log(f"  [{p}] running video pipeline (may take 10-170 min if OpenFace hasn't run for this participant yet)...")

    ok, info, elapsed = run(
        [os.path.join("scripts", "video", "02_extract_video_features.py"), p, "--openface-exe", OPENFACE_EXE],
        # Generous timeout, throughput varies with system load, still-running job.
        timeout=10 * 3600,
    )
    st["steps"]["video_features"] = f"OK ({elapsed:.0f}s)" if ok else "FAILED"
    if not ok:
        record_error(p, "video_features", info)
        st["video_status"] = "failed"
        return

    # Must point at the session cache, the script's own default is the
    # old per-clip cache dir, since the calibration switch.
    ok, info, elapsed = run([os.path.join("scripts", "video", "04_validate_features_blink_overlap.py"),
                              p, "--post-filter",
                              "--openface-raw-dir", os.path.join("data", "processed", "video", "openface_raw_session_wide")])
    st["steps"]["blink_validation"] = f"OK ({elapsed:.0f}s)" if ok else "FAILED"
    if not ok:
        record_error(p, "blink_validation", info)

    # Some participants have video but permanently unusable PPG. Harmless to
    # pass unconditionally when PPG is present.
    ok, info, elapsed = run([os.path.join("scripts", "features", "01_merge_features.py"), p,
                              "--allow-missing-ppg"])
    st["steps"]["merge"] = f"OK complete ({elapsed:.0f}s)" if ok else "FAILED"
    if not ok:
        record_error(p, "merge_complete", info)
        st["video_status"] = "failed"
    else:
        st["video_status"] = "complete"

    try:
        vf = pd.read_csv(os.path.join("data", "processed", "video", f"{p}_video_features.csv"))
        st["glasses_flag"] = bool(vf["likely_glasses_au_bias"].iloc[0]) if "likely_glasses_au_bias" in vf.columns else None
        st["blink_threshold"] = (vf["blink_threshold_min_ms"].iloc[0], vf["blink_threshold_max_ms"].iloc[0])
        st["pct_excluded"] = dict(zip(vf["task"], vf["pct_frames_excluded_total"].round(1)))
    except Exception as e:
        log(f"  [{p}] could not read back video_features.csv for report details: {e}")

    try:
        val = pd.read_csv(os.path.join("results", "video_exploration", f"{p}_feature_validation_post_filter.csv"))
        st["validation_flags"] = val["flag"].value_counts().to_dict()
    except Exception as e:
        log(f"  [{p}] could not read back validation flags: {e}")


def write_report(all_participant_ids, has_ppg_set, has_video_set):
    lines = []
    lines.append("# Pipeline-status — volledig cohort\n")
    lines.append("Autonoom, doorlopend bijgewerkt. Methodologische bron: `notes/methodology_notes.md`.\n")
    lines.append(f"Laatste update: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    n_total = len(all_participant_ids)
    n_ppg = sum(1 for p in all_participant_ids if p in has_ppg_set)
    n_video = len(has_video_set)
    n_complete = sum(1 for p in all_participant_ids if STATE.get(p, {}).get("video_status") == "complete")
    n_pending = sum(1 for p in all_participant_ids if STATE.get(p, {}).get("video_status") == "pending")

    lines.append(f"**{n_total} participanten totaal** ({n_ppg} met PPG, {n_video} met video). "
                 f"**{n_complete} mastertabellen compleet** (incl. video), **{n_pending} partieel** "
                 f"(video_pending), rest nog niet verwerkt of gefaald.\n")

    lines.append("## Overzicht per participant\n")
    lines.append("| Participant | PsychoPy | PPG | Video | task_windows | ppg_features | labels | video_features | Mastertabel |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for p in all_participant_ids:
        st = STATE.get(p, {"steps": {}, "video_status": "no_video"})
        steps = st["steps"]
        video_status = st["video_status"]
        mastertabel = {"complete": "COMPLEET", "pending": "PARTIEEL (video_pending)",
                       "failed": "GEFAALD", "no_video": "-"}[video_status]
        lines.append(
            f"| {p} | Ja | {'Ja' if p in has_ppg_set else 'Nee'} | {'Ja' if p in has_video_set else 'Nee'} | "
            f"{steps.get('task_windows','-')} | {steps.get('ppg_features','-')} | {steps.get('labels','-')} | "
            f"{steps.get('video_features','-')} | {mastertabel} |"
        )
    lines.append("")

    video_participants = [p for p in all_participant_ids if p in has_video_set]
    if video_participants:
        lines.append("## Video-detail (adaptieve drempels, brilcheck, validatie)\n")
        lines.append("| Participant | Blink-drempel (ms) | Uitsluiting % (Stroop/Arithm./FaceRecall) | Bril-flag (AU04) | Validatie (ROOD/GEEL/GROEN) |")
        lines.append("|---|---|---|---|---|")
        for p in video_participants:
            st = STATE.get(p, {})
            thr = st.get("blink_threshold")
            thr_str = f"{thr[0]:.0f}-{thr[1]:.0f}" if thr else "-"
            pct = st.get("pct_excluded")
            pct_str = " / ".join(f"{pct.get(t,'-')}" for t in ["stroop", "arithmetic", "face_recall"]) if pct else "-"
            glasses = st.get("glasses_flag")
            glasses_str = "JA" if glasses else ("Nee" if glasses is False else "-")
            flags = st.get("validation_flags")
            flags_str = f"{flags.get('ROOD',0)}/{flags.get('GEEL',0)}/{flags.get('GROEN',0)}" if flags else "-"
            lines.append(f"| {p} | {thr_str} | {pct_str} | {glasses_str} | {flags_str} |")
        lines.append("")

    lines.append("## Foutenlog\n")
    if not ERRORS:
        lines.append("Geen fouten opgetreden.\n")
    else:
        for e in ERRORS[-100:]:  # cap so the report doesn't explode
            lines.append(f"### {e['participant']} — stap '{e['step']}'")
            lines.append("```")
            lines.append((e["error"] or "")[:1500])
            lines.append("```\n")
        if len(ERRORS) > 100:
            lines.append(f"(+{len(ERRORS)-100} oudere fouten niet getoond -- zie {LOG_PATH} voor het volledige log)\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--watch-hours", type=float, default=5.5,
                         help="How long (total, including phases A/B) to keep watching for new video "
                              "files after the initial batch before exiting (default 5.5h).")
    parser.add_argument("--watch-interval-s", type=float, default=300,
                         help="How often to rescan data/raw/video/ for new files during the watch phase.")
    parser.add_argument("--exclude", nargs="+", default=[],
                         help="Participant id(s) to skip entirely (e.g. --exclude p1, for a participant "
                              "with permanently unusable raw data). Existing outputs for an excluded "
                              "participant are left untouched, not deleted.")
    args = parser.parse_args()
    exclude_set = {p.lower() for p in args.exclude}

    t_start = time.time()
    deadline = t_start + args.watch_hours * 3600

    psychopy_ids, ppg_ids, video_ids, obs_ids = discover_participants()
    if exclude_set:
        log(f"Excluding participant(s) from this run: {sorted(exclude_set)}")
        psychopy_ids -= exclude_set
        ppg_ids -= exclude_set
        video_ids -= exclude_set
    all_ids = sorted(psychopy_ids, key=lambda x: int(x[1:]))
    log(f"=== Full pipeline run starting: {len(all_ids)} participants with PsychoPy data, "
        f"{len(ppg_ids)} with PPG, {len(video_ids)} with video ===")
    missing_ppg = sorted(psychopy_ids - ppg_ids, key=lambda x: int(x[1:]))
    if missing_ppg:
        log(f"Participants with PsychoPy but NO PPG file (ppg_features/merge will be skipped): {missing_ppg}")
    missing_obs = sorted(psychopy_ids - obs_ids, key=lambda x: int(x[1:]))
    if missing_obs:
        log(f"Participants with PsychoPy but NO obs_log (task_windows will fail): {missing_obs}")

    write_report(all_ids, ppg_ids, video_ids)

    # --- Phase A: fast steps for everyone ---
    log("\n=== PHASE A: task windows / PPG / labels / partial merge for all participants ===")
    for i, p in enumerate(all_ids, 1):
        log(f"[A {i}/{len(all_ids)}] {p}")
        try:
            phase_a_participant(p, has_ppg=(p in ppg_ids))
        except Exception as e:
            record_error(p, "phase_a_uncaught", f"{e}\n{traceback.format_exc()}")
        write_report(all_ids, ppg_ids, video_ids)

    # --- Phase B: full video pipeline for participants that already have video ---
    processed_video = set()
    video_participants_initial = sorted(video_ids, key=lambda x: int(x[1:]))
    log(f"\n=== PHASE B: full video pipeline for {video_participants_initial} ===")
    for i, p in enumerate(video_participants_initial, 1):
        log(f"[B {i}/{len(video_participants_initial)}] {p}")
        try:
            phase_b_participant(p)
        except Exception as e:
            record_error(p, "phase_b_uncaught", f"{e}\n{traceback.format_exc()}")
        processed_video.add(p)
        write_report(all_ids, ppg_ids, video_ids)

    # --- Phase C: watch for new video files until the deadline ---
    log(f"\n=== PHASE C: watching {VIDEO_DIR} for new video files until "
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(deadline))} ===")
    while time.time() < deadline:
        _, _, current_video_ids, _ = discover_participants()
        current_video_ids -= exclude_set
        new_ones = sorted(current_video_ids - processed_video, key=lambda x: int(x[1:]))
        if new_ones:
            log(f"New video file(s) detected: {new_ones}")
            for p in new_ones:
                if p not in psychopy_ids:
                    log(f"  [{p}] has a video but no PsychoPy/OBS data -- cannot compute task windows, skipping")
                    processed_video.add(p)
                    continue
                if p not in STATE or STATE[p]["video_status"] not in ("pending",):
                    # make sure phase-A steps exist for this participant too
                    # (e.g. it may have arrived with PPG after phase A already ran)
                    try:
                        phase_a_participant(p, has_ppg=(p in ppg_ids) or os.path.exists(
                            os.path.join("data", "processed", "ppg", f"{p}_ppg_features.csv")))
                    except Exception as e:
                        record_error(p, "phase_a_uncaught_late", f"{e}\n{traceback.format_exc()}")
                try:
                    phase_b_participant(p)
                except Exception as e:
                    record_error(p, "phase_b_uncaught_late", f"{e}\n{traceback.format_exc()}")
                processed_video.add(p)
                write_report(all_ids, ppg_ids | {p}, current_video_ids)
        time.sleep(args.watch_interval_s)

    log("=== Full pipeline run complete (watch deadline reached) ===")
    write_report(all_ids, ppg_ids, discover_participants()[2] - exclude_set)


if __name__ == "__main__":
    main()
