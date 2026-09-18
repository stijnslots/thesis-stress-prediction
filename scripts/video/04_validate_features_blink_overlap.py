"""
Systematic blink-contamination check for every OpenFace feature used in the
video pipeline (17 Action Unit intensities, 8 gaze columns, 3 head-pose
columns).

For each feature and task, takes the top-20 most extreme values (or, for
gaze and pose, the top-20 largest frame-to-frame changes) and checks what
fraction land close to a detected blink, compared against a chance
baseline. Features are flagged red, yellow or green depending on how far
above chance that overlap is.

A "post-filter" mode re-runs this check on the same frames the pipeline
actually used after its own blink and confidence filtering, using a wider
window than the exclusion margin: even after removing the immediate blink
vicinity, do the remaining peaks still cluster suspiciously close to a
blink? (Not finished yet, manual control still needed.)

Usage:
    python 04_validate_features_blink_overlap.py p20
    python 04_validate_features_blink_overlap.py p20 --tasks stroop arithmetic
    python 04_validate_features_blink_overlap.py p20 --post-filter --out-dir results/video_exploration
"""

import argparse
import os

import numpy as np
import pandas as pd

from blink_utils import (au45_blink_run_indices, derive_adaptive_blink_thresholds,
                          compute_blink_exclusion_mask, EXCLUDED_CONTAMINATED_FEATURES)

AU_COLS = ["AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r", "AU10_r",
           "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r"]
# The two excluded features aren't in these lists.
GAZE_COLS = [c for c in ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z",
                          "gaze_angle_x", "gaze_angle_y"] if c not in EXCLUDED_CONTAMINATED_FEATURES]
POSE_COLS = [c for c in ["pose_Rx", "pose_Ry", "pose_Rz"] if c not in EXCLUDED_CONTAMINATED_FEATURES]

DEFAULT_TASKS = ["stroop", "arithmetic", "face_recall"]
WINDOW_FRAMES = 5
EXCLUSION_MARGIN_FRAMES = 15  # must match BLINK_FILTER_MARGIN_FRAMES in 02_extract_video_features.py
CHECK_MARGIN_FRAMES = 25      # wider than the exclusion margin -- see --post-filter note above
CONFIDENCE_EXCLUSION_THRESHOLD = 0.7  # must match 02_extract_video_features.py
N_TOP = 20


def load_openface_csv(openface_raw_dir, participant, task):
    task_dir = os.path.join(openface_raw_dir, f"{participant}_{task}")
    candidates = [f for f in os.listdir(task_dir) if f.endswith(".csv")]
    df = pd.read_csv(os.path.join(task_dir, candidates[0]))
    df.columns = [c.strip() for c in df.columns]
    return df.reset_index(drop=True)


def get_participant_blink_thresholds(participant, tasks, openface_raw_dir):
    """Read the adaptive thresholds the pipeline actually used from
    {participant}_video_features.csv; re-derive independently (with a
    warning) if that file isn't available yet."""
    video_features_path = os.path.join("data", "processed", "video", f"{participant}_video_features.csv")
    if os.path.exists(video_features_path):
        vf = pd.read_csv(video_features_path)
        if {"blink_threshold_min_ms", "blink_threshold_max_ms"}.issubset(vf.columns):
            return float(vf["blink_threshold_min_ms"].iloc[0]), float(vf["blink_threshold_max_ms"].iloc[0])

    print(f"  WARNING: could not read thresholds from {video_features_path} -- re-deriving independently "
          "(may not exactly match what the pipeline used).")
    all_durations_ms, frame_times_ms = [], []
    for task in tasks:
        df = load_openface_csv(openface_raw_dir, participant, task)
        frame_time_s = float(np.median(np.diff(df["timestamp"].to_numpy())))
        from blink_utils import au45_blink_run_durations_ms
        all_durations_ms.extend(au45_blink_run_durations_ms(df["AU45_c"].to_numpy(), frame_time_s).tolist())
        frame_times_ms.append(frame_time_s * 1000)
    info = derive_adaptive_blink_thresholds(all_durations_ms, float(np.median(frame_times_ms)))
    return info["min_duration_ms"], info["max_duration_ms"]


def blink_exclusion_pool_mask(au45c, confidence, frame_time_s, min_duration_ms, max_duration_ms,
                               margin=EXCLUSION_MARGIN_FRAMES):
    """True = frame is a valid candidate (NOT blink-adjacent AND NOT
    low-confidence) -- the same pool 02_extract_video_features.py
    aggregates over post-filter."""
    blink_excluded = compute_blink_exclusion_mask(au45c, frame_time_s, margin, min_duration_ms, max_duration_ms)
    low_conf_excluded = confidence < CONFIDENCE_EXCLUSION_THRESHOLD
    return ~(blink_excluded | low_conf_excluded)


def chance_baseline_pct(au45c, window=WINDOW_FRAMES, pool_mask=None):
    n = len(au45c)
    near = np.zeros(n, dtype=bool)
    for i in np.where(au45c == 1)[0]:
        lo, hi = max(0, i - window), min(n, i + window + 1)
        near[lo:hi] = True
    if pool_mask is not None:
        near = near[pool_mask]
    return near.mean() * 100


def overlap_for_series(values, au45c, window=WINDOW_FRAMES, n_top=N_TOP, pool_mask=None):
    n = len(au45c)
    candidates = values if pool_mask is None else values[pool_mask]
    top_idx = candidates.sort_values(ascending=False).head(n_top).index
    exact = sum(au45c[idx] == 1 for idx in top_idx)
    near = sum((au45c[max(0, idx - window):min(n, idx + window + 1)] == 1).any() for idx in top_idx)
    return exact, near, len(top_idx)


def flag_and_recommendation(feature, mean_ratio, n_tasks_ge_1_5x, post_filter=False):
    if feature == "AU45_r":
        return "N.V.T.", "Triviaal — dit IS de blink-intensiteit; geen validatie nodig."
    if mean_ratio >= 1.8 or n_tasks_ge_1_5x == 3:
        reco = ("Blink-filter alleen loste dit niet op: nog steeds sterk blink-gerelateerd net buiten de "
                "uitgesloten zone. Overweeg een breder filter of deze feature laten vervallen." if post_filter else
                "Niet aanbevolen zonder correctie: filter frames met AU45_c==1 (+/-5 frames) "
                "eruit voor je deze feature aggregeert, of laat de feature vervallen.")
        return "ROOD", reco
    if mean_ratio >= 1.3 or n_tasks_ge_1_5x >= 1:
        reco = ("Verbeterd maar niet volledig schoon — nog lichte blink-nabijheid net buiten de "
                "uitgesloten zone. Bruikbaar, met dat voorbehoud." if post_filter else
                "Bruikbaar met correctie: blink-filter aanraden vóór aggregatie, vooral voor "
                "skew/kurtosis (piekgevoelig). Mean/std waarschijnlijk minder kwetsbaar.")
        return "GEEL", reco
    reco = ("Filter werkte: overlap zit nu op/onder kansniveau, ook net buiten de uitgesloten zone." if post_filter else
            "Bruikbaar zonder correctie: overlap met blinks ligt op of onder kansniveau.")
    return "GROEN", reco


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("participant")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--openface-raw-dir", default=os.path.join("data", "processed", "video", "openface_raw"))
    parser.add_argument("--out-dir", default=os.path.join("results", "video_exploration"))
    parser.add_argument("--post-filter", action="store_true",
                         help="Validate the ALREADY-blink-filtered pipeline output instead of the raw "
                              "signal -- see the 'Post-filter mode' note in the module docstring.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = "_post_filter" if args.post_filter else ""
    window = CHECK_MARGIN_FRAMES if args.post_filter else WINDOW_FRAMES

    min_duration_ms = max_duration_ms = None
    if args.post_filter:
        min_duration_ms, max_duration_ms = get_participant_blink_thresholds(
            args.participant, args.tasks, args.openface_raw_dir)
        print(f"Using adaptive thresholds for {args.participant}: "
              f"{min_duration_ms:.1f}-{max_duration_ms:.1f}ms\n")

    long_rows = []
    for task in args.tasks:
        df = load_openface_csv(args.openface_raw_dir, args.participant, task)
        au45c = df["AU45_c"].to_numpy()
        frame_time_s = float(np.median(np.diff(df["timestamp"].to_numpy())))
        pool_mask = (blink_exclusion_pool_mask(au45c, df["confidence"].to_numpy(), frame_time_s,
                                                min_duration_ms, max_duration_ms)
                     if args.post_filter else None)
        baseline = chance_baseline_pct(au45c, window=window, pool_mask=pool_mask)
        if args.post_filter:
            print(f"{task}: n={len(df)}, {pool_mask.sum()} frames survive the +/-{EXCLUSION_MARGIN_FRAMES}-frame "
                  f"blink filter ({pool_mask.mean()*100:.1f}%); of those, baseline within +/-{window} "
                  f"frames of a blink = {baseline:.1f}%")
        else:
            print(f"{task}: n={len(df)}, chance baseline (±{window} frames of a blink) = {baseline:.1f}%")

        for col in AU_COLS:
            exact, near, n = overlap_for_series(df[col], au45c, window=window, pool_mask=pool_mask)
            long_rows.append({"feature": col, "feature_type": "AU", "task": task,
                               "near_pct": near / n * 100, "baseline_pct": baseline,
                               "ratio_near": (near / n * 100) / baseline})

        for col in GAZE_COLS + POSE_COLS:
            diff = df[col].diff().abs()
            diff.iloc[0] = 0
            exact, near, n = overlap_for_series(diff, au45c, window=window, pool_mask=pool_mask)
            ftype = "gaze_change" if col in GAZE_COLS else "pose_change"
            long_rows.append({"feature": col, "feature_type": ftype, "task": task,
                               "near_pct": near / n * 100, "baseline_pct": baseline,
                               "ratio_near": (near / n * 100) / baseline})

    long_df = pd.DataFrame(long_rows)
    long_path = os.path.join(args.out_dir, f"{args.participant}_feature_validation_raw{suffix}.csv")
    long_df.to_csv(long_path, index=False)

    wide_rows = []
    for (ftype, feat), g in long_df.groupby(["feature_type", "feature"]):
        g = g.set_index("task")
        row = {"feature": feat, "feature_type": ftype}
        ratios = []
        for t in args.tasks:
            row[f"overlap_pct_{t}"] = g.loc[t, "near_pct"]
            row[f"baseline_pct_{t}"] = g.loc[t, "baseline_pct"]
            row[f"ratio_{t}"] = g.loc[t, "ratio_near"]
            ratios.append(g.loc[t, "ratio_near"])
        row["mean_ratio"] = float(np.mean(ratios))
        row["min_ratio"] = float(np.min(ratios))
        row["max_ratio"] = float(np.max(ratios))
        row["n_tasks_ge_1.5x"] = sum(r >= 1.5 for r in ratios)
        row["flag"], row["recommendation"] = flag_and_recommendation(
            feat, row["mean_ratio"], row["n_tasks_ge_1.5x"], post_filter=args.post_filter)
        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows).sort_values(["feature_type", "mean_ratio"], ascending=[True, False])
    wide_path = os.path.join(args.out_dir, f"{args.participant}_feature_validation{suffix}.csv")
    wide_df.to_csv(wide_path, index=False)

    print(f"\nSaved: {long_path}")
    print(f"Saved: {wide_path}")
    print("\n=== Summary ===")
    summary_cols = ["feature", "feature_type"] + [f"overlap_pct_{t}" for t in args.tasks] + ["mean_ratio", "flag"]
    print(wide_df[summary_cols].round(1).to_string(index=False))
    print("\nFlag counts:")
    print(wide_df["flag"].value_counts().to_string())


if __name__ == "__main__":
    main()
