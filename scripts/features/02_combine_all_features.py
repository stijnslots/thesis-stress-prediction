"""
Combines every participant's master feature table into one cohort-wide
table. Every table shares the same column layout, so this is a
straightforward concatenation, not a fuzzy merge. Rows are included as-is
even when only PPG is done and video or GSR are not yet available.

Usage:
    python 02_combine_all_features.py
    python 02_combine_all_features.py --out-name combined_features_partial.csv
"""

import argparse
import glob
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-dir", default=os.path.join("data", "processed", "features"))
    parser.add_argument("--out-name", default="combined_features_partial.csv")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.features_dir, "p*_master_features.csv")),
                   key=lambda p: int(os.path.basename(p).split("_")[0][1:]))
    if not paths:
        raise FileNotFoundError(f"No *_master_features.csv files found in {args.features_dir}")

    print(f"Found {len(paths)} participant master tables:")
    frames = []
    col_sets = {}
    for p in paths:
        df = pd.read_csv(p)
        pid = os.path.basename(p).split("_master_features")[0]
        col_sets[pid] = tuple(df.columns)
        frames.append(df)
        print(f"  {pid}: {df.shape[0]} rows x {df.shape[1]} cols "
              f"(ppg_pending={df['ppg_pending'].iloc[0]}, video_pending={df['video_pending'].iloc[0]})")

    # Schema should be identical across all tables.
    unique_schemas = set(col_sets.values())
    if len(unique_schemas) > 1:
        print(f"\nWARNING: {len(unique_schemas)} different column schemas found across participant tables "
              "(expected 1) -- concatenation will still work but will create extra NaN columns where "
              "schemas differ. Investigate before treating this as a clean dataset.")
        ref = col_sets[list(col_sets)[0]]
        for pid, cols in col_sets.items():
            if cols != ref:
                print(f"  {pid} differs from the first table's schema "
                      f"(+{set(cols)-set(ref)}, -{set(ref)-set(cols)})")

    combined = pd.concat(frames, ignore_index=True, sort=False)

    if "likely_glasses_au_bias" not in combined.columns:
        print("\nWARNING: 'likely_glasses_au_bias' column not found in any table -- "
              "glasses-check results will not be present in the combined output.")

    out_path = os.path.join(args.features_dir, args.out_name)
    combined.to_csv(out_path, index=False)

    # Writes .xlsx
    xlsx_path = os.path.splitext(out_path)[0] + ".xlsx"
    combined.to_excel(xlsx_path, index=False)

    print(f"\nSaved combined table: {out_path}")
    print(f"Saved Excel-friendly version: {xlsx_path}")
    print(f"  {combined.shape[0]} rows x {combined.shape[1]} columns "
          f"({combined['participant'].nunique()} participants)")
    print(f"  ppg_pending: {int(combined['ppg_pending'].sum())}/{len(combined)} rows")
    print(f"  video_pending: {int(combined['video_pending'].sum())}/{len(combined)} rows")
    if "likely_glasses_au_bias" in combined.columns:
        flagged = combined[combined["likely_glasses_au_bias"] == True]["participant"].nunique()
        print(f"  likely_glasses_au_bias: flagged for {flagged} participant(s)")


if __name__ == "__main__":
    main()
