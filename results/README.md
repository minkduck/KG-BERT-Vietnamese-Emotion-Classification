# Experiment Results & Output Documentation

This directory contains result documentation and tools for aggregating experiment outputs.

---

## 📌 Reproducibility & Archived Logs Notice

> **Important Notice on Run Logs:**  
> Per-seed execution logs were not archived in this repository, and the paper's tables therefore cannot be directly recomputed from pre-saved logs in this repository.  
> However, all source code, model definitions (`src/models/`), ontology resources (`ontology/`), hyper-parameter configuration files (`configs/*.yaml`), random seeds (`0, 123, 1234, 2025, 42`), and self-contained execution notebooks (`notebooks/`) needed to re-run each experiment from scratch are fully provided.

---

## 📂 Mapping Notebooks to Paper Tables & Figures

| Notebook Path | Feeds Paper Table / Figure | Description |
| :--- | :--- | :--- |
| **`notebooks/01-phobert-fusion-ablation/`** | **Table 4** | Knowledge fusion ablation on PhoBERT-base-v2 |
| **`notebooks/02-visobert-fusion-ablation/`** | **Table 5** | Knowledge fusion ablation on ViSoBERT |
| **`notebooks/03-lexicon-vs-ontology/`** | **Table 6** | Knowledge source comparison (VnEmoLex vs. Ontology) |
| **`notebooks/04-final-model-comparison/`** | **Table 7** | Re-implemented hybrid fusion baselines (ALDONAr, KEAHT, CombViSA) |
| **`notebooks/05-shap-lime/`** | **Figure 5 & Section 5.7.2** | Post-hoc SHAP and LIME feature attribution outputs |

---

## ⚙️ Multi-Seed Aggregation Methodology

- **Execution Order:** Each notebook is executed **one seed at a time** in the order: `0, 123, 1234, 2025, 42`.
- **Sample Standard Deviation ($\text{ddof}=1$):** All mean $\pm$ std values reported in paper Tables 4–8 represent **sample standard deviation ($s$, $\text{ddof}=1$)** aggregated across the 5 separate seed executions.
- **Aggregation Tool:** The script [`results/aggregate.py`](aggregate.py) is provided to compute sample statistics (`ddof=1`) across output CSV files.
