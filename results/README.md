# Experiment Results & Log Ingestion

This directory contains aggregated experiment results, per-seed CSV records, and verification tools matching the paper's published tables.

---

## 📂 Mapping CSV Files to Paper Tables & Figures

| CSV File Path | Feeds Paper Table / Section | Description |
| :--- | :--- | :--- |
| **`01-phobert-fusion-ablation/*/per_seed.csv`** | **Table 4** | Knowledge fusion ablation on PhoBERT-base-v2 |
| **`02-visobert-fusion-ablation/*/per_seed.csv`** | **Table 5** | Knowledge fusion ablation on ViSoBERT |
| **`03-lexicon-vs-ontology/*/per_seed.csv`** | **Table 6** | Knowledge source comparison (VnEmoLex vs. Ontology) |
| **`04-hybrid-baselines/*/per_seed.csv`** | **Table 7** | Re-implemented hybrid fusion baselines (ALDONAr, KEAHT, CombViSA) |
| **`*/*/per_class.csv`** | **Tables 8–10** | Per-class Precision, Recall, F1, and Support metrics |
| **`*/*/significance.csv`** | **Section 4.4** | Non-parametric paired permutation test $p$-values ($10,000$ resamples) |
| **`05-shap-lime/`** | **Figure 5 / Section 5.7.2** | Post-hoc SHAP and LIME feature attribution outputs |

---

## ⚙️ Seed Execution & Statistical Methodology

- **Individual Seed Runs:** Each notebook is executed **one seed at a time** in the exact order: `0, 123, 1234, 2025, 42`.
- **Sample Standard Deviation ($\text{ddof}=1$):** All mean $\pm$ std figures reported in paper Tables 4–8 represent **sample standard deviation ($s$, $\text{ddof}=1$)** calculated across the 5 separate seed runs using `results/aggregate.py`.
- **Note on Log Summaries:** Raw summary `.docx` documents calculate population standard deviation ($\sigma$, $\text{ddof}=0$). For $n=5$, population std and sample std differ by a factor of $\sqrt{\frac{5}{4}} \approx 1.11803$.

---

## 📊 Automated Verification Tool

Run the aggregation script to recalculate sample statistics across all per-seed CSV files:

```bash
python results/aggregate.py
```

See [`results/VERIFICATION.md`](VERIFICATION.md) for the complete cell-by-cell comparison table between reconstructed metrics and published paper values.

---

## 🚨 Status of Log Files & Priority TODOs

> **PROMINENT TODO (Author Action Required):**  
> 1. **Group 01 Logs (Table 4):** Upload the per-seed console output `.docx` files for **`01-phobert-fusion-ablation`** (PhoBERT fusion ablation on UIT-VSFC, VSFC-Ekman, and UIT-VSMEC) to `results/_incoming/`. These back Table 4 of the paper.  
> 2. **Groups 02, 03, 04 Logs:** Place the remaining incoming `.docx` run logs into `results/_incoming/` for automated conversion via `aggregate.py`.
