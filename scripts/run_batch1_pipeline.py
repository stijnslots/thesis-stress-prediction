"""
Batch-runs the full per-participant pipeline (task windows, PPG features,
video features, blink-filter validation, labels, merged feature table) for
a list of participants, unattended.

If a step fails for one participant, the error is logged and the remaining
steps for that participant are skipped, but the batch continues with the
next one.

Usage:
    python scripts/run_batch1_pipeline.py p21 p22 p23
"""

import subprocess
import sys
import os
import time
import traceback

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + os.sep + ".."
REPO_ROOT = os.path.abspath(REPO_ROOT)
os.chdir(REPO_ROOT)

PYTHON = sys.executable
OPENFACE_EXE = os.path.join(REPO_ROOT, "scratchpad_video_tools", "OpenFace_2.2.0_win_x64",
                             "OpenFace_2.2.0_win_x64", "FeatureExtraction.exe")

LOG_PATH = os.path.join("results", "batch1_run_log.txt")
REPORT_PATH = os.path.join("results", "pipeline_status_batch1.md")

STEPS = [
    ("task_windows", [os.path.join("scripts", "sync", "extract_task_windows.py"), "{p}"]),
    ("ppg_features", [os.path.join("scripts", "ppg", "03_extract_ppg_features.py"), "{p}"]),
    ("video_features", [os.path.join("scripts", "video", "02_extract_video_features.py"), "{p}",
                         "--openface-exe", OPENFACE_EXE]),
    ("blink_validation", [os.path.join("scripts", "video", "04_validate_features_blink_overlap.py"), "{p}",
                           "--post-filter"]),
    ("labels", [os.path.join("scripts", "labels", "01_extract_stress_labels.py"), "{p}"]),
    ("merge", [os.path.join("scripts", "features", "01_merge_features.py"), "{p}"]),
]

P20_BLINK_EXCLUSION_PCT = {"stroop": 48.44, "arithmetic": 40.73, "face_recall": 27.40}

_log_file = open(LOG_PATH, "a", encoding="utf-8")


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _log_file.write(line + "\n")
    _log_file.flush()


def run_step(participant, step_name, cmd_template):
    cmd = [PYTHON] + [c.format(p=participant) for c in cmd_template]
    log(f"  [{participant}] running '{step_name}': {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3 * 3600)
        elapsed = time.time() - t0
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-3000:]
            log(f"  [{participant}] '{step_name}' FAILED (exit {proc.returncode}) after {elapsed:.0f}s")
            return False, tail, elapsed
        log(f"  [{participant}] '{step_name}' OK ({elapsed:.0f}s)")
        return True, proc.stdout[-2000:], elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        log(f"  [{participant}] '{step_name}' TIMED OUT after {elapsed:.0f}s")
        return False, "TimeoutExpired after 3h", elapsed
    except Exception as e:
        elapsed = time.time() - t0
        log(f"  [{participant}] '{step_name}' EXCEPTION: {e}\n{traceback.format_exc()}")
        return False, f"{e}\n{traceback.format_exc()}", elapsed


def collect_participant_data(participant, completed_steps):
    """Best-effort read of this participant's output files for the report.
    Never raises -- a read failure here just means less detail in the
    report, not a batch abort."""
    data = {"warnings": []}

    try:
        if "task_windows" in completed_steps:
            tw = pd.read_csv(os.path.join("data", "processed", "synced", f"{participant}_task_windows.csv"))
            data["tasks_found"] = tw["task"].tolist()
            data["task_durations_s"] = dict(zip(tw["task"], tw["duration_s"].round(1)))
            if set(tw["task"]) != {"stroop", "arithmetic", "face_recall"}:
                data["warnings"].append(f"unexpected task set: {set(tw['task'])}")
    except Exception as e:
        data["warnings"].append(f"could not read task_windows.csv: {e}")

    try:
        if "ppg_features" in completed_steps:
            ppg = pd.read_csv(os.path.join("data", "processed", "ppg", f"{participant}_ppg_features.csv"))
            plaus = {}
            for col, lo, hi in [("HRV_MeanNN", 300, 1600), ("HRV_SDNN", 0, 300), ("HRV_RMSSD", 0, 300)]:
                if col in ppg.columns:
                    bad = ppg[(ppg[col] < lo) | (ppg[col] > hi)]
                    if len(bad):
                        data["warnings"].append(f"{col} outside [{lo},{hi}] for tasks {bad['task'].tolist()}")
                    plaus[col] = ppg[col].round(1).tolist()
            data["ppg_plausibility"] = plaus
            data["ppg_nan_cols_all_missing"] = ppg.columns[ppg.isna().all()].tolist()
    except Exception as e:
        data["warnings"].append(f"could not read ppg_features.csv: {e}")

    try:
        if "video_features" in completed_steps:
            vid = pd.read_csv(os.path.join("data", "processed", "video", f"{participant}_video_features.csv"))
            data["blink_exclusion_pct"] = dict(zip(vid["task"], vid["pct_frames_excluded_blink"].round(1)))
            data["confidence_mean"] = dict(zip(vid["task"], vid["confidence_mean"].round(3)))
            data["blink_rate_per_min"] = dict(zip(vid["task"], vid["blink_rate_per_min"].round(1)))
            low_conf = vid[vid["confidence_mean"] < 0.9]
            if len(low_conf):
                data["warnings"].append(f"low confidence_mean (<0.9) for tasks {low_conf['task'].tolist()}")
    except Exception as e:
        data["warnings"].append(f"could not read video_features.csv: {e}")

    try:
        if "blink_validation" in completed_steps:
            val = pd.read_csv(os.path.join("results", "video_exploration",
                                            f"{participant}_feature_validation_post_filter.csv"))
            data["blink_validation_flags"] = val["flag"].value_counts().to_dict()
            reds = val[val["flag"] == "ROOD"]["feature"].tolist()
            data["blink_validation_red_features"] = reds
    except Exception as e:
        data["warnings"].append(f"could not read feature_validation_post_filter.csv: {e}")

    try:
        if "labels" in completed_steps:
            lbl = pd.read_csv(os.path.join("data", "processed", "labels", f"{participant}_labels.csv"))
            data["labels"] = dict(zip(lbl["task"], lbl["stress_label"]))
            oob = lbl[(lbl["stress_label"] < 0) | (lbl["stress_label"] > 9) | lbl["stress_label"].isna()]
            if len(oob):
                data["warnings"].append(f"label out of [0,9] or missing for tasks {oob['task'].tolist()}")
    except Exception as e:
        data["warnings"].append(f"could not read labels.csv: {e}")

    try:
        if "merge" in completed_steps:
            master = pd.read_csv(os.path.join("data", "processed", "features", f"{participant}_master_features.csv"))
            data["master_shape"] = master.shape
    except Exception as e:
        data["warnings"].append(f"could not read master_features.csv: {e}")

    return data


def write_report(participants, results, errors):
    lines = []
    lines.append("# Pipeline-status — batch 1 (p21, p22, p23)\n")
    lines.append("Autonoom uitgevoerd terwijl je weg was. Methodologische bron: `notes/methodology_notes.md`.\n")

    n_ok = sum(1 for p in participants if not results[p]["failed"])
    lines.append(f"**{n_ok}/{len(participants)} participanten volledig verwerkt zonder fouten.**\n")

    lines.append("## Overzicht per participant\n")
    for p in participants:
        r = results[p]
        status = "VOLLEDIG GESLAAGD" if not r["failed"] else f"GESTOPT bij stap '{r['failed_step']}'"
        lines.append(f"### {p} — {status}\n")
        lines.append("| Stap | Resultaat | Tijd |")
        lines.append("|---|---|---|")
        for step_name, _ in STEPS:
            info = r["steps"].get(step_name, "niet gestart")
            lines.append(f"| {step_name} | {info} |")
        lines.append("")

        d = r["data"]
        if d.get("warnings"):
            lines.append("**Waarschuwingen/afwijkingen gevonden:**")
            for w in d["warnings"]:
                lines.append(f"- {w}")
            lines.append("")

        if "task_durations_s" in d:
            lines.append(f"- Taken gevonden: {d['tasks_found']} (durations: {d['task_durations_s']})")
        if "blink_exclusion_pct" in d:
            lines.append(f"- Blink-filter uitsluiting: {d['blink_exclusion_pct']}  "
                          f"(p20 referentie: {P20_BLINK_EXCLUSION_PCT})")
        if "blink_validation_flags" in d:
            lines.append(f"- Blink-validatie (post-filter) vlaggen: {d['blink_validation_flags']}")
            if d.get("blink_validation_red_features"):
                lines.append(f"  - ROOD: {d['blink_validation_red_features']}")
        if "ppg_plausibility" in d:
            lines.append(f"- HRV-plausibiliteit (MeanNN/SDNN/RMSSD per taak): {d['ppg_plausibility']}")
        if "labels" in d:
            lines.append(f"- Stress-labels: {d['labels']}")
        if "master_shape" in d:
            lines.append(f"- Mastertabel shape: {d['master_shape']}")
        lines.append("")

    lines.append("## Foutenlog\n")
    if not errors:
        lines.append("Geen fouten opgetreden.\n")
    else:
        for e in errors:
            lines.append(f"### {e['participant']} — stap '{e['step']}'")
            lines.append("```")
            lines.append(e["error"][:2000])
            lines.append("```\n")

    lines.append("## Houden de p20-beslissingen stand op deze nieuwe data?\n")
    lines.append("**Blink-filter (drempel 42,5-450,5ms + marge ±15 frames):** zie de post-filter-validatievlaggen "
                  "hierboven per participant. Als die overwegend GEEL/GROEN blijven (geen nieuwe ROOD-explosie), "
                  "houdt de instelling stand.\n")
    lines.append("**NeuroKit2 voor PPG:** dit rapport bevestigt alleen dat NeuroKit2 foutloos draait en "
                  "fysiologisch plausibele HRV-waarden oplevert op de nieuwe signalen (zie plausibiliteit "
                  "hierboven) — dit is GEEN hernieuwde HeartPy-vs-NeuroKit2-vergelijking zoals destijds voor p20 "
                  "(dat zou een aparte analyse vereisen, bijv. met `scripts/ppg/01_explore_ppg_p20.py`, die nu "
                  "nog niet generiek is per participant). Behandel deze bevestiging dus als een lichtgewicht "
                  "plausibiliteitscheck, niet als een volledige herbevestiging van de methodekeuze.\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Report written to {REPORT_PATH}")


def main():
    participants = sys.argv[1:]
    if not participants:
        print("Usage: python run_batch1_pipeline.py <participant> [<participant> ...]")
        sys.exit(1)

    log(f"=== Batch run starting for {participants} ===")
    results = {}
    errors = []

    for p in participants:
        log(f"=== Participant {p} ===")
        results[p] = {"steps": {}, "failed": False, "failed_step": None, "data": {}}
        completed_steps = []
        for step_name, cmd_template in STEPS:
            ok, info, elapsed = run_step(p, step_name, cmd_template)
            if ok:
                results[p]["steps"][step_name] = f"OK ({elapsed:.0f}s)"
                completed_steps.append(step_name)
            else:
                results[p]["steps"][step_name] = f"FAILED ({elapsed:.0f}s) — zie foutenlog"
                results[p]["failed"] = True
                results[p]["failed_step"] = step_name
                errors.append({"participant": p, "step": step_name, "error": info})
                log(f"  [{p}] stopping remaining steps for this participant, continuing batch with next participant")
                break

        results[p]["data"] = collect_participant_data(p, completed_steps)
        log(f"=== Finished participant {p} (failed={results[p]['failed']}) ===")

        # Write the report after EVERY participant, not just at the end, so
        # a crash partway through the batch still leaves a usable report.
        write_report(list(results.keys()), results, errors)

    log("=== Batch run complete ===")


if __name__ == "__main__":
    main()
