"""
Extract per-task PPG/HRV features for one participant, using NeuroKit2.

For each task window (Stroop, Arithmetic, Face Recall):
    1. Cut the matching slice out of the raw Shimmer PPG signal.
    2. Clean the signal and detect heartbeats.
    3. Compute the HRV metric set (SDNN, RMSSD, pNN50, LF/HF, etc.) from
       the detected peaks.
    4. Save the underlying beat-to-beat intervals separately, so summary
       statistics can be recomputed later without reprocessing the raw
       signal.

The first 30 seconds of each task window are trimmed before processing, to
exclude the cardiac orienting response at task onset. Pass --trim-seconds 0
to get the untrimmed version for comparison; both versions are saved under
different filenames so they can be compared.

Usage:
    python 03_extract_ppg_features.py p20                       # trimmed (default 30s)
    python 03_extract_ppg_features.py p20 --trim-seconds 0       # untrimmed
    python 03_extract_ppg_features.py p20 --tasks stroop arithmetic
"""

import argparse
import os
import re

import neurokit2 as nk
import numpy as np
import pandas as pd

PPG_COL = "S5E1C_PPG_A13_CAL"
TS_COL = "S5E1C_Timestamp_Unix_CAL"
DEFAULT_TASKS = ["stroop", "arithmetic", "face_recall"]


def find_ppg_csv(ppg_dir, participant):
    """Match by prefix (case-insensitive, participant id not followed by
    another digit) rather than a hardcoded '{participant}_Shimmer.csv'
    pattern, since the observed naming (P20_Shimmer.csv for participant
    'p20') capitalizes the id — this stays robust to that or other casing.
    IMPORTANT: a plain substring check ("participant in filename") is NOT
    safe here -- 'p2' is a substring of 'P20_Shimmer.csv', 'P21_...', ...,
    'P29_...', so participants p1-p7 were silently matching a random file
    from their own decade (e.g. p1 -> P10_Shimmer.csv) until this was
    caught. The (?!\\d) lookahead rejects those false matches."""
    pattern = re.compile(rf"^{re.escape(participant)}(?!\d)", re.IGNORECASE)
    matches = [
        f for f in os.listdir(ppg_dir)
        if pattern.match(f) and "shimmer" in f.lower() and f.lower().endswith(".csv")
    ]
    if not matches:
        raise FileNotFoundError(f"No Shimmer PPG csv found for participant '{participant}' in {ppg_dir}")
    if len(matches) > 1:
        print(f"Warning: multiple PPG csv files match '{participant}' in {ppg_dir}, using the first: {sorted(matches)[0]}")
    return os.path.join(ppg_dir, sorted(matches)[0])


def load_ppg(ppg_path):
    """Column names are prefixed with the Shimmer device ID, which is NOT
    consistent across participants -- most use "S5E1C_..." but p66/p68 (at
    least) were recorded on a different device ("Shimmer4_..."). Find the
    timestamp/PPG columns by their SUFFIX instead of hardcoding the device
    prefix, then rename to the canonical TS_COL/PPG_COL so the rest of this
    script doesn't need to care which device recorded a given participant."""
    df = pd.read_csv(ppg_path, sep="\t", skiprows=[0, 2])

    ts_matches = [c for c in df.columns if c.endswith("Timestamp_Unix_CAL")]
    ppg_matches = [c for c in df.columns if c.endswith("PPG_A13_CAL")]
    if len(ts_matches) != 1 or len(ppg_matches) != 1:
        raise ValueError(f"{ppg_path}: expected exactly one timestamp and one PPG column by suffix match, "
                          f"found timestamp={ts_matches}, ppg={ppg_matches}. Columns present: {list(df.columns)}")
    df = df.rename(columns={ts_matches[0]: TS_COL, ppg_matches[0]: PPG_COL})

    df[TS_COL] = df[TS_COL].astype(float)
    df[PPG_COL] = df[PPG_COL].astype(float)
    return df


def estimate_sample_rate(timestamps_ms):
    return 1000.0 / np.median(np.diff(timestamps_ms))


def load_task_windows(windows_path):
    return pd.read_csv(windows_path)


def cut_window(df, start_unix_ms, end_unix_ms):
    mask = (df[TS_COL] >= start_unix_ms) & (df[TS_COL] <= end_unix_ms)
    return df.loc[mask].reset_index(drop=True)


def process_task_signal(signal, sample_rate):
    signals, info = nk.ppg_process(signal, sampling_rate=sample_rate)
    peaks = np.where(signals["PPG_Peaks"].values == 1)[0]
    hrv_df = nk.hrv(peaks, sampling_rate=sample_rate, show=False)

    rr_ms = np.diff(peaks) / sample_rate * 1000.0
    rr_table = pd.DataFrame({
        "rr_index": np.arange(1, len(peaks)),
        "peak_sample_index": peaks[1:],
        "peak_time_s": peaks[1:] / sample_rate,
        "rr_interval_ms": rr_ms,
    })
    return hrv_df, rr_table, len(peaks)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("participant", help="Participant id, e.g. p20")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                         help=f"Task names to process (default: {DEFAULT_TASKS})")
    parser.add_argument("--ppg-dir", default=os.path.join("data", "raw", "ppg"))
    parser.add_argument("--windows-dir", default=os.path.join("data", "processed", "synced"))
    parser.add_argument("--out-dir", default=os.path.join("data", "processed", "ppg"))
    parser.add_argument("--trim-seconds", type=float, default=30.0,
                         help="Seconds to cut from the START of each task window before PPG/HRV "
                              "processing, to exclude the cardiac orienting response at task onset "
                              "(see notes/methodology_notes.md). Default 30. Pass 0 to reproduce the "
                              "original untrimmed behavior. Output filename reflects this: trim_seconds=0 "
                              "-> {participant}_ppg_features.csv (unchanged, original file), "
                              "trim_seconds>0 -> {participant}_ppg_features_trimmed.csv (never overwrites "
                              "the untrimmed file, so both can be compared side by side).")
    args = parser.parse_args()
    if args.trim_seconds < 0:
        raise ValueError("--trim-seconds must be >= 0")

    os.makedirs(args.out_dir, exist_ok=True)

    ppg_path = find_ppg_csv(args.ppg_dir, args.participant)
    print(f"Loading PPG data from {ppg_path} ...")
    ppg_df = load_ppg(ppg_path)
    sample_rate = estimate_sample_rate(ppg_df[TS_COL].values)
    print(f"  {len(ppg_df)} samples, estimated sample rate: {sample_rate:.2f} Hz")

    windows_path = os.path.join(args.windows_dir, f"{args.participant}_task_windows.csv")
    print(f"Loading task windows from {windows_path} ...")
    windows_df = load_task_windows(windows_path)

    feature_rows = []
    for task in args.tasks:
        matches = windows_df[windows_df["task"] == task]
        if matches.empty:
            available = ", ".join(windows_df["task"].unique())
            print(f"  WARNING: task '{task}' not found in {windows_path} (available: {available}) — skipping")
            continue
        window = matches.iloc[0]

        trim_ms = args.trim_seconds * 1000.0
        effective_duration_s = window["duration_s"] - args.trim_seconds
        if effective_duration_s <= 0:
            print(f"  WARNING: --trim-seconds ({args.trim_seconds}s) >= task window duration "
                  f"({window['duration_s']:.1f}s) for '{task}' -- nothing would be left. Skipping this task.")
            continue

        start_unix_ms = float(window["start_abs_unix"]) * 1000.0 + trim_ms
        end_unix_ms = float(window["end_abs_unix"]) * 1000.0
        segment = cut_window(ppg_df, start_unix_ms, end_unix_ms)
        signal = segment[PPG_COL].to_numpy()

        trim_note = f" (first {args.trim_seconds:.0f}s trimmed -> {effective_duration_s:.1f}s effective)" if args.trim_seconds > 0 else ""
        print(f"\nProcessing task '{task}': {window['duration_s']:.1f}s window{trim_note}, {len(signal)} samples")

        # Flag rather than skip when trimming leaves a short window,
        # probably the shortest task.
        if effective_duration_s < 60:
            print(f"  NOTE: effective window after trim is only {effective_duration_s:.1f}s (<60s) for '{task}' -- "
                  "HRV estimates from this few beats may be less reliable than for longer tasks.")

        # A near-empty segment crashes the bandpass filter with an error.
        # A clear message instead of losing the whole participant to one
        # bad task.
        min_expected_samples = int(sample_rate * 5)  # 5s floor, well below any real task
        if len(signal) < min_expected_samples:
            print(f"  WARNING: only {len(signal)} samples for '{task}' (expected ~{window['duration_s']*sample_rate:.0f}) "
                  f"-- raw PPG recording likely doesn't cover this task window. Skipping this task.")
            continue

        try:
            hrv_df, rr_table, n_peaks = process_task_signal(signal, sample_rate)
        except Exception as e:
            print(f"  WARNING: NeuroKit2 processing failed for task '{task}': {type(e).__name__}: {e} -- skipping this task.")
            continue

        rr_suffix = "_rr_intervals_trimmed.csv" if args.trim_seconds > 0 else "_rr_intervals.csv"
        rr_path = os.path.join(args.out_dir, f"{args.participant}_{task}{rr_suffix}")
        rr_table.to_csv(rr_path, index=False)
        print(f"  {n_peaks} peaks detected, {len(rr_table)} RR intervals -> {rr_path}")

        row = {"participant": args.participant, "task": task, "n_peaks": n_peaks,
               "trim_seconds": args.trim_seconds, "effective_duration_s": effective_duration_s}
        row.update(hrv_df.iloc[0].to_dict())
        feature_rows.append(row)

    if not feature_rows:
        raise ValueError(f"No task could be processed for {args.participant} -- the raw PPG recording likely "
                          "doesn't overlap any task window (e.g. truncated/corrupted Shimmer file). "
                          "Check data/raw/ppg/ for this participant.")
    if len(feature_rows) < len(args.tasks):
        done = [r["task"] for r in feature_rows]
        print(f"\nWARNING: only {len(feature_rows)}/{len(args.tasks)} tasks succeeded ({done}) -- "
              "the output csv will have fewer than 3 rows for this participant.")

    features_df = pd.DataFrame(feature_rows)
    out_name = f"{args.participant}_ppg_features_trimmed.csv" if args.trim_seconds > 0 else f"{args.participant}_ppg_features.csv"
    out_path = os.path.join(args.out_dir, out_name)
    features_df.to_csv(out_path, index=False)

    print(f"\nSaved PPG feature table: {out_path}")
    print(f"  {features_df.shape[0]} rows (tasks) x {features_df.shape[1]} columns")

    key_cols = ["participant", "task", "n_peaks", "HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD",
                "HRV_pNN50", "HRV_LF", "HRV_HF", "HRV_LFHF"]
    key_cols = [c for c in key_cols if c in features_df.columns]
    print("\n=== Summary (key HRV metrics) ===")
    print(features_df[key_cols].to_string(index=False))

    print(f"\n(Full feature table has {features_df.shape[1]} columns — see {out_path} for the complete "
          "NeuroKit2 time/frequency/nonlinear HRV set.)")


if __name__ == "__main__":
    main()
