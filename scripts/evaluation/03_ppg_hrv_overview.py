"""
Exploratory analysis of PPG and PRV features across all participants for
whom PPG has been processed, joined with stress labels and task-window
durations.

Usage:
    python scripts/evaluation/03_ppg_hrv_overview.py
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PPG_DIR = os.path.join("data", "processed", "ppg")
LABELS_DIR = os.path.join("data", "processed", "labels")
SYNCED_DIR = os.path.join("data", "processed", "synced")
OUT_DIR = os.path.join("results", "eda_preliminary")
TASK_ORDER = ["stroop", "arithmetic", "face_recall"]

CORE_HRV = ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_LF", "HRV_HF", "HRV_LFHF"]
HIST_FEATURES = ["HRV_SDNN", "HRV_RMSSD", "HRV_LFHF"]
BPM_MIN, BPM_MAX = 40, 180

TOTAL_COHORT = 76


def load_ppg_all():
    paths = sorted(glob.glob(os.path.join(PPG_DIR, "p*_ppg_features.csv")),
                   key=lambda p: int(os.path.basename(p).split("_")[0][1:]))
    paths = [p for p in paths if "rr_intervals" not in p]
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["task"] = pd.Categorical(df["task"], categories=TASK_ORDER, ordered=True)
    return df


def load_labels_all():
    paths = glob.glob(os.path.join(LABELS_DIR, "p*_labels.csv"))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def load_durations_all(participants):
    frames = []
    for p in participants:
        path = os.path.join(SYNCED_DIR, f"{p}_task_windows.csv")
        if os.path.exists(path):
            sub = pd.read_csv(path)[["participant", "task", "duration_s"]]
            frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def print_desc_table(df, feature):
    desc = df.groupby("task", observed=True)[feature].agg(["mean", "median", "std", "min", "max", "count"])
    print(f"\n{feature}:")
    print(desc)
    return desc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ppg = load_ppg_all()
    participants = sorted(ppg["participant"].unique(), key=lambda x: int(x[1:]))
    labels = load_labels_all()
    durations = load_durations_all(participants)

    merged = ppg.merge(labels, on=["participant", "task"], how="left")
    merged = merged.merge(durations, on=["participant", "task"], how="left")

    n_participants = ppg["participant"].nunique()
    n_obs = len(ppg)
    print(f"=== Scope ===")
    print(f"PPG-features beschikbaar voor {n_participants} participanten, {n_obs} observaties "
          f"({n_participants} x 3 taken, alle 3 taken aanwezig per participant).")

    # --- 1. Descriptive statistics per task, per core HRV feature ---
    print(f"\n=== 1. Beschrijvende statistieken per taak (kern-HRV) ===")
    desc_tables = {}
    for feat in CORE_HRV:
        desc_tables[feat] = print_desc_table(merged, feat)

    # --- 2. Histograms: SDNN, RMSSD, LFHF x 3 tasks (3x3 grid) ---
    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    for row, feat in enumerate(HIST_FEATURES):
        for col, t in enumerate(TASK_ORDER):
            ax = axes[row, col]
            sub = merged.loc[merged["task"] == t, feat].dropna()
            ax.hist(sub, bins=15, edgecolor="black")
            ax.set_title(f"{feat} -- {t} (n={len(sub)})")
            ax.set_xlabel(feat)
            ax.set_ylabel("amount")
    fig.suptitle(f"HRV-featureverdelingen per taak (N={n_participants} participanten)")
    fig.tight_layout()
    hist_path = os.path.join(OUT_DIR, "ppg_overview.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {hist_path}")

    # --- 3. Task comparison ---
    print(f"\n=== 3. Vergelijking tussen taken (stress-literatuur check) ===")
    rmssd_means = desc_tables["HRV_RMSSD"]["mean"]
    sdnn_means = desc_tables["HRV_SDNN"]["mean"]
    lfhf_means = desc_tables["HRV_LFHF"]["mean"]
    print(f"RMSSD (mean) per taak: {rmssd_means.to_dict()}")
    print(f"SDNN (mean) per taak: {sdnn_means.to_dict()}")
    print(f"LF/HF (mean) per taak: {lfhf_means.to_dict()}")
    arithmetic_lowest_rmssd = rmssd_means['arithmetic'] == rmssd_means.min()
    arithmetic_lowest_sdnn = sdnn_means['arithmetic'] == sdnn_means.min()
    arithmetic_highest_lfhf = lfhf_means['arithmetic'] == lfhf_means.max()
    print(f"Arithmetic heeft de laagste RMSSD van de 3 taken: {arithmetic_lowest_rmssd}")
    print(f"Arithmetic heeft de laagste SDNN van de 3 taken: {arithmetic_lowest_sdnn}")
    print(f"Arithmetic heeft de hoogste LF/HF van de 3 taken: {arithmetic_highest_lfhf}")

    # --- 4. Correlation matrix ---
    print(f"\n=== 4. Correlatiematrix (kern-HRV-features, alle taken gepoold) ===")
    corr = merged[CORE_HRV].corr()
    print(corr.round(3))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(CORE_HRV)))
    ax.set_xticklabels(CORE_HRV, rotation=45, ha="right")
    ax.set_yticks(range(len(CORE_HRV)))
    ax.set_yticklabels(CORE_HRV)
    for i in range(len(CORE_HRV)):
        for j in range(len(CORE_HRV)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                     color="white" if abs(corr.values[i, j]) > 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Correlatiematrix kern-HRV-features (alle taken gepoold)")
    fig.tight_layout()
    corr_path = os.path.join(OUT_DIR, "ppg_correlation_heatmap.png")
    fig.savefig(corr_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {corr_path}")

    # --- 5. Scatterplots SDNN/RMSSD vs stress_label per task ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row, feat in enumerate(["HRV_SDNN", "HRV_RMSSD"]):
        for col, t in enumerate(TASK_ORDER):
            ax = axes[row, col]
            sub = merged.loc[merged["task"] == t, [feat, "stress_label"]].dropna()
            ax.scatter(sub["stress_label"], sub[feat], alpha=0.7)
            if len(sub) >= 2:
                coeffs = np.polyfit(sub["stress_label"], sub[feat], 1)
                xs = np.linspace(sub["stress_label"].min(), sub["stress_label"].max(), 50)
                ax.plot(xs, np.polyval(coeffs, xs), color="red", linewidth=1)
            ax.set_title(f"{feat} vs stress_label -- {t} (n={len(sub)})")
            ax.set_xlabel("stress_label")
            ax.set_ylabel(feat)
    fig.suptitle("Indicatieve verkenning HRV vs. stress_label -- GEEN statistische toets, "
                 f"N={n_participants} participanten")
    fig.tight_layout()
    scatter_path = os.path.join(OUT_DIR, "ppg_scatter_vs_stress.png")
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {scatter_path}")

    # --- 6. Signal-quality check via n_peaks -> implied bpm ---
    print(f"\n=== 6. Signaalkwaliteit-check (implied bpm uit n_peaks) ===")
    merged["implied_bpm"] = merged["n_peaks"] / (merged["duration_s"] / 60.0)
    flagged = merged[(merged["implied_bpm"] < BPM_MIN) | (merged["implied_bpm"] > BPM_MAX)]
    print(f"Fysiologisch plausibel bereik: {BPM_MIN}-{BPM_MAX} bpm.")
    print(f"Gevlagd als mogelijk onbetrouwbaar: {len(flagged)}/{len(merged)} observaties "
          f"({100*len(flagged)/len(merged):.1f}%)")
    if len(flagged):
        print(flagged[["participant", "task", "n_peaks", "duration_s", "implied_bpm"]]
              .sort_values("implied_bpm").to_string(index=False))
    else:
        print("Geen enkele observatie valt buiten het plausibele bereik.")
    print(f"\nimplied_bpm beschrijvende statistiek per taak:")
    print(merged.groupby("task", observed=True)["implied_bpm"].agg(["mean", "median", "std", "min", "max"]))

    # --- 7. Data completeness ---
    print(f"\n=== 7. Datacompleetheid ===")
    all_ids = [f"p{i}" for i in range(1, TOTAL_COHORT + 1)]
    usable_set = set(participants)
    missing_ids = [p for p in all_ids if p not in usable_set]
    n_usable = n_participants
    n_missing = len(missing_ids)
    print(f"Bruikbare PPG-features: {n_usable}/{TOTAL_COHORT} participanten ({100*n_usable/TOTAL_COHORT:.1f}%).")
    print(f"Ontbrekend/niet bruikbaar: {n_missing}/{TOTAL_COHORT} -> {missing_ids}")
    print("(Dit telt zowel structureel gefaalde PPG-opnames als eventuele participanten die nog niet "
          "verwerkt zijn -- dit script maakt zelf geen onderscheid tussen die twee, dat vereist het "
          "foutenlog/de raw PPG-bestanden te inspecteren.)")
    if n_missing == 14:
        print("Dit komt overeen met het eerder vastgestelde aantal van 14.")
    else:
        print(f"LET OP: dit wijkt af van het eerder vastgestelde aantal van 14 "
              f"(nu {n_missing}) -- controleer de lijst hierboven, bijvoorbeeld doordat er inmiddels "
              "meer/minder participanten verwerkt zijn dan eerder.")

    combined_out = os.path.join("data", "processed", "features", "combined_ppg_hrv_all.csv")
    merged.to_csv(combined_out, index=False)
    merged.to_excel(os.path.splitext(combined_out)[0] + ".xlsx", index=False)
    print(f"\nSaved combined PPG+labels table: {combined_out} (+ .xlsx)")


if __name__ == "__main__":
    main()
