"""
Exploratory. Extract per-task GSR/EDA features for one participant, using
NeuroKit2. Structured like the PPG extraction script, but not yet part of
the main pipeline (working on it).

For each task window:
    1. Cut the matching slice out of the raw Shimmer GSR (skin conductance)
       signal.
    2. Clean the signal and decompose it into a tonic and a phasic
       component, and detect skin-conductance-response peaks.
    3. Extract tonic and phasic summary features.

Sample rate is estimated per participant rather than assumed, since GSR and
PPG are recorded on the same timestamp column and should share one rate.
Uses the same 30-second start-of-task trim as PPG, justified independently
for EDA.

Usage:
    python 01_extract_gsr_features.py p20
    python 01_extract_gsr_features.py p20 --trim-seconds 0
    python 01_extract_gsr_features.py p20 --tasks stroop arithmetic
"""

import argparse
import os
import re

import neurokit2 as nk
import numpy as np
import pandas as pd

TS_SUFFIX = "Timestamp_Unix_CAL"
COND_SUFFIX = "GSR_Skin_Conductance_CAL"
TS_COL = "Timestamp_Unix_CAL"
COND_COL = "GSR_Skin_Conductance_CAL"
DEFAULT_TASKS = ["stroop", "arithmetic", "face_recall"]


def find_ppg_csv(ppg_dir, participant):
    """Same matching logic as scripts/ppg/03_extract_ppg_features.py --
    prefix match, not substring, to avoid 'p2' matching 'P20_...'."""
    pattern = re.compile(rf"^{re.escape(participant)}(?!\d)", re.IGNORECASE)
    matches = [
        f for f in os.listdir(ppg_dir)
        if pattern.match(f) and "shimmer" in f.lower() and f.lower().endswith(".csv")
    ]
    if not matches:
        raise FileNotFoundError(f"No Shimmer csv found for participant '{participant}' in {ppg_dir}")
    if len(matches) > 1:
        print(f"Warning: multiple csv files match '{participant}' in {ppg_dir}, using the first: {sorted(matches)[0]}")
    return os.path.join(ppg_dir, sorted(matches)[0])


def load_gsr(shimmer_path):
    """Column names are device-prefixed and vary (S5E1C_/Shimmer_610A_/
    Shimmer4_) -- find by SUFFIX, same approach as the PPG loader, then
    rename to canonical names."""
    df = pd.read_csv(shimmer_path, sep="\t", skiprows=[0, 2])

    ts_matches = [c for c in df.columns if c.endswith(TS_SUFFIX)]
    cond_matches = [c for c in df.columns if c.endswith(COND_SUFFIX)]
    if len(ts_matches) != 1 or len(cond_matches) != 1:
        raise ValueError(f"{shimmer_path}: expected exactly one timestamp and one GSR conductance column, "
                          f"found timestamp={ts_matches}, conductance={cond_matches}")
    df = df.rename(columns={ts_matches[0]: TS_COL, cond_matches[0]: COND_COL})
    df[TS_COL] = df[TS_COL].astype(float)
    df[COND_COL] = df[COND_COL].astype(float)
    return df


def estimate_sample_rate(timestamps_ms):
    """Same estimator as PPG -- median inter-sample interval, not a
    hardcoded constant, since GSR/PPG share one timestamp stream but this
    keeps the two scripts independently correct if that ever changes."""
    return 1000.0 / np.median(np.diff(timestamps_ms))


def load_task_windows(windows_path):
    return pd.read_csv(windows_path)


def cut_window(df, start_unix_ms, end_unix_ms):
    mask = (df[TS_COL] >= start_unix_ms) & (df[TS_COL] <= end_unix_ms)
    return df.loc[mask].reset_index(drop=True)


def process_task_signal(signal, sample_rate):
    signals, info = nk.eda_process(signal, sampling_rate=sample_rate, method="neurokit", method_phasic="cvxeda")
    return signals, info


def compute_features(signals, info, duration_s):
    scl = signals["EDA_Tonic"]
    n_scr = int(signals["SCR_Peaks"].sum())
    nscr_per_min = n_scr / duration_s * 60.0

    amp = np.asarray(info.get("SCR_Amplitude", []), dtype=float)
    rise = np.asarray(info.get("SCR_RiseTime", []), dtype=float)
    recovery = np.asarray(info.get("SCR_RecoveryTime", []), dtype=float)

    def nanmean(a):
        return float(np.nanmean(a)) if len(a) and not np.all(np.isnan(a)) else np.nan

    def nanmedian(a):
        return float(np.nanmedian(a)) if len(a) and not np.all(np.isnan(a)) else np.nan

    return {
        "scl_mean": float(scl.mean()),
        "scl_std": float(scl.std()),
        "n_scr": n_scr,
        "nscr_per_min": nscr_per_min,
        "scr_amplitude_mean": nanmean(amp),
        "scr_amplitude_median": nanmedian(amp),
        "scr_risetime_mean_s": nanmean(rise),
        "scr_recoverytime_mean_s": nanmean(recovery),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("participant", help="Participant id, e.g. p20")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--ppg-dir", default=os.path.join("data", "raw", "ppg"))
    parser.add_argument("--windows-dir", default=os.path.join("data", "processed", "synced"))
    parser.add_argument("--out-dir", default=os.path.join("data", "processed", "gsr"))
    parser.add_argument("--trim-seconds", type=float, default=30.0,
                         help="Seconds to cut from the START of each task window before GSR processing "
                              "(default 30, matches PPG's default -- see module docstring for the "
                              "independent justification). Pass 0 to reproduce untrimmed behavior.")
    args = parser.parse_args()
    if args.trim_seconds < 0:
        raise ValueError("--trim-seconds must be >= 0")

    os.makedirs(args.out_dir, exist_ok=True)

    shimmer_path = find_ppg_csv(args.ppg_dir, args.participant)
    print(f"Loading GSR data from {shimmer_path} ...")
    gsr_df = load_gsr(shimmer_path)
    sample_rate = estimate_sample_rate(gsr_df[TS_COL].values)
    print(f"  {len(gsr_df)} samples, estimated sample rate: {sample_rate:.2f} Hz")

    windows_path = os.path.join(args.windows_dir, f"{args.participant}_task_windows.csv")
    print(f"Loading task windows from {windows_path} ...")
    windows_df = load_task_windows(windows_path)

    feature_rows = []
    for task in args.tasks:
        matches = windows_df[windows_df["task"] == task]
        if matches.empty:
            print(f"  WARNING: task '{task}' not found -- skipping")
            continue
        window = matches.iloc[0]

        trim_ms = args.trim_seconds * 1000.0
        effective_duration_s = window["duration_s"] - args.trim_seconds
        if effective_duration_s <= 0:
            print(f"  WARNING: trim >= task duration for '{task}' -- skipping")
            continue

        start_unix_ms = float(window["start_abs_unix"]) * 1000.0 + trim_ms
        end_unix_ms = float(window["end_abs_unix"]) * 1000.0
        segment = cut_window(gsr_df, start_unix_ms, end_unix_ms)
        signal = segment[COND_COL].to_numpy()

        print(f"\nProcessing task '{task}': {effective_duration_s:.1f}s effective, {len(signal)} samples")

        min_expected_samples = int(sample_rate * 5)
        if len(signal) < min_expected_samples:
            print(f"  WARNING: only {len(signal)} samples for '{task}' -- skipping (recording likely doesn't cover this window)")
            continue

        try:
            signals, info = process_task_signal(signal, sample_rate)
        except Exception as e:
            print(f"  WARNING: eda_process failed for task '{task}': {type(e).__name__}: {e} -- skipping this task.")
            continue

        feats = compute_features(signals, info, effective_duration_s)
        print(f"  SCL mean={feats['scl_mean']:.3f} uS, {feats['n_scr']} SCR peaks "
              f"({feats['nscr_per_min']:.2f}/min), amplitude mean={feats['scr_amplitude_mean']:.3f} uS")

        row = {"participant": args.participant, "task": task, "sample_rate_hz": sample_rate,
               "trim_seconds": args.trim_seconds, "effective_duration_s": effective_duration_s}
        row.update(feats)
        feature_rows.append(row)

        # Cache the processed per-task signal (not just the summary row) so
        # validation/plotting scripts can re-use it without re-running
        # eda_process.
        sig_out = signals.copy()
        sig_out["participant"] = args.participant
        sig_out["task"] = task
        sig_path = os.path.join(args.out_dir, f"{args.participant}_{task}_eda_signal.csv")
        sig_out.to_csv(sig_path, index=False)

    if not feature_rows:
        raise ValueError(f"No task could be processed for {args.participant}.")

    features_df = pd.DataFrame(feature_rows)
    out_path = os.path.join(args.out_dir, f"{args.participant}_gsr_features.csv")
    features_df.to_csv(out_path, index=False)
    print(f"\nSaved GSR feature table: {out_path}")
    print(features_df[["task", "scl_mean", "scl_std", "n_scr", "nscr_per_min",
                        "scr_amplitude_mean", "scr_amplitude_median"]].to_string(index=False))


if __name__ == "__main__":
    main()
