#!/usr/bin/env python3
"""
Aggregate experiment results across seeds for the PhoBERT-Ontology paper.
Calculates sample mean and sample standard deviation (ddof=1) per dataset and model.
"""

import os
import glob
import pandas as pd
import numpy as np

# Paper Published Values for Verification (Tables 4–8)
# Format: { (Table, Dataset, Model): (Macro_F1_Mean, Macro_F1_Std) }
PUBLISHED_TABLES = {
    # Table 4: PhoBERT Fusion Ablation on PhoBERT-base-v2
    ("Table 4", "UIT-VSFC", "Baseline"): (76.30, 0.45),
    ("Table 4", "UIT-VSFC", "RawConcat"): (75.15, 0.62),
    ("Table 4", "UIT-VSFC", "RawGate"): (76.40, 0.38),
    ("Table 4", "UIT-VSFC", "DenseConcat"): (75.87, 0.51),
    ("Table 4", "UIT-VSFC", "DenseGate"): (76.64, 0.42),
    
    ("Table 4", "VSFC-Ekman", "Baseline"): (65.01, 0.72),
    ("Table 4", "VSFC-Ekman", "RawConcat"): (63.80, 0.85),
    ("Table 4", "VSFC-Ekman", "RawGate"): (65.93, 0.61),
    ("Table 4", "VSFC-Ekman", "DenseConcat"): (64.50, 0.78),
    ("Table 4", "VSFC-Ekman", "DenseGate"): (66.10, 0.65),

    ("Table 4", "UIT-VSMEC", "Baseline"): (65.13, 0.88),
    ("Table 4", "UIT-VSMEC", "RawConcat"): (64.20, 0.95),
    ("Table 4", "UIT-VSMEC", "RawGate"): (65.28, 0.75),
    ("Table 4", "UIT-VSMEC", "DenseConcat"): (64.80, 0.91),
    ("Table 4", "UIT-VSMEC", "DenseGate"): (65.50, 0.82),

    # Table 5: ViSoBERT Fusion Ablation
    ("Table 5", "UIT-VSFC", "Baseline"): (90.02, 0.15),
    ("Table 5", "UIT-VSFC", "RawConcat"): (88.34, 0.42),
    ("Table 5", "UIT-VSFC", "RawGate"): (89.73, 0.28),
    ("Table 5", "UIT-VSFC", "DenseConcat"): (90.02, 0.20),
    ("Table 5", "UIT-VSFC", "DenseGate"): (90.24, 0.18),
    ("Table 5", "UIT-VSFC", "Residual gate"): (90.21, 0.22),

    ("Table 5", "VSFC-Ekman", "Baseline"): (60.00, 0.92),
    ("Table 5", "VSFC-Ekman", "RawConcat"): (58.40, 1.10),
    ("Table 5", "VSFC-Ekman", "RawGate"): (59.22, 0.85),
    ("Table 5", "VSFC-Ekman", "DenseConcat"): (58.90, 0.98),
    ("Table 5", "VSFC-Ekman", "DenseGate"): (59.50, 0.88),
    ("Table 5", "VSFC-Ekman", "Residual gate"): (55.60, 1.05),

    ("Table 5", "UIT-VSMEC", "Baseline"): (58.87, 0.79),
    ("Table 5", "UIT-VSMEC", "RawConcat"): (57.10, 0.95),
    ("Table 5", "UIT-VSMEC", "RawGate"): (58.39, 0.82),
    ("Table 5", "UIT-VSMEC", "DenseConcat"): (58.20, 0.88),
    ("Table 5", "UIT-VSMEC", "DenseGate"): (58.60, 0.80),
    ("Table 5", "UIT-VSMEC", "Residual gate"): (57.80, 0.90),

    # Table 6: Knowledge Source Comparison (PhoBERT & ViSoBERT)
    ("Table 6", "VSFC-Ekman (PhoBERT)", "Baseline"): (64.72, 0.70),
    ("Table 6", "VSFC-Ekman (PhoBERT)", "VnEmoLex Concat"): (63.18, 0.82),
    ("Table 6", "VSFC-Ekman (PhoBERT)", "RawGate"): (65.93, 0.61),

    ("Table 6", "VSFC-Ekman (ViSoBERT)", "Baseline"): (57.04, 0.90),
    ("Table 6", "VSFC-Ekman (ViSoBERT)", "VnEmoLex Concat"): (57.30, 0.88),
    ("Table 6", "VSFC-Ekman (ViSoBERT)", "RawGate"): (59.49, 0.75),

    ("Table 6", "UIT-VSMEC (PhoBERT)", "Baseline"): (65.13, 0.88),
    ("Table 6", "UIT-VSMEC (PhoBERT)", "VnEmoLex Concat"): (65.28, 0.75),
    ("Table 6", "UIT-VSMEC (PhoBERT)", "RawGate"): (63.26, 0.92),

    ("Table 6", "UIT-VSMEC (ViSoBERT)", "Baseline"): (57.13, 0.85),
    ("Table 6", "UIT-VSMEC (ViSoBERT)", "VnEmoLex Concat"): (58.28, 0.80),
    ("Table 6", "UIT-VSMEC (ViSoBERT)", "RawGate"): (58.39, 0.82),

    # Table 7: Hybrid Fusion Baselines on ViSoBERT
    ("Table 7", "VSFC-Ekman", "Baseline"): (58.11, 0.88),
    ("Table 7", "VSFC-Ekman", "ALDONAr"): (59.05, 0.76),
    ("Table 7", "VSFC-Ekman", "KEAHT"): (60.92, 0.65),
    ("Table 7", "VSFC-Ekman", "CombViSA"): (54.64, 1.12),

    ("Table 7", "UIT-VSMEC", "Baseline"): (55.92, 0.95),
    ("Table 7", "UIT-VSMEC", "ALDONAr"): (56.98, 0.82),
    ("Table 7", "UIT-VSMEC", "KEAHT"): (57.87, 0.75),
    ("Table 7", "UIT-VSMEC", "CombViSA"): (58.69, 0.70),
}


def load_all_per_seed_csvs(root_dir="results"):
    pattern = os.path.join(root_dir, "*", "*", "per_seed.csv")
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)


def aggregate_results(df):
    if df.empty:
        print("No per_seed.csv files found to aggregate.")
        return pd.DataFrame()

    grouped = df.groupby(["dataset", "group", "model"]).agg(
        n_seeds=("seed", "count"),
        acc_mean=("accuracy", lambda x: x.mean() * 100),
        acc_std=("accuracy", lambda x: x.std(ddof=1) * 100),
        f1_mean=("macro_f1", lambda x: x.mean() * 100),
        f1_std=("macro_f1", lambda x: x.std(ddof=1) * 100),
    ).reset_index()

    return grouped


def main():
    results_dir = os.path.dirname(os.path.abspath(__file__))
    df = load_all_per_seed_csvs(results_dir)
    
    print("=== PER-SEED AGGREGATION (Sample std, ddof=1) ===")
    if df.empty:
        print("NOTE: No per_seed.csv files present in results/ subdirectories yet.")
        print("Incoming .docx log files have not been placed in results/_incoming/.")
    else:
        agg = aggregate_results(df)
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
