# RESULTS VERIFICATION REPORT — Paper Tables 4–8

> **Verification Standard:**  
> - Comparison metric: Macro-F1 (Mean ± Sample Standard Deviation $s$, $\text{ddof}=1$).  
> - Tolerance threshold: $0.01$ percentage points.  
> - Note on Standard Deviation: `.docx` summary logs report population standard deviation ($\text{ddof}=0$, $\sigma$), whereas the paper reports sample standard deviation ($\text{ddof}=1$, $s$). For $n=5$ seeds, these differ by a factor of $\sqrt{\frac{5}{4}} \approx 1.11803$.

---

## 1. Summary of Incoming Log Files (`results/_incoming/`)

| Incoming Log File | Target Experiment Group | Status / Verification State |
| :--- | :--- | :--- |
| **`01-phobert-fusion-ablation`** | Group 01 (PhoBERT Fusion Ablation) | ❌ **NO DATA** — Log files not provided in `results/_incoming/` |
| **`02-visobert-fusion-ablation`** | Group 02 (ViSoBERT Fusion Ablation) | ❌ **NO DATA** — Log files pending upload in `results/_incoming/` |
| **`03-lexicon-vs-ontology`** | Group 03 (Knowledge Source Comparison) | ❌ **NO DATA** — Log files pending upload in `results/_incoming/` |
| **`04-hybrid-baselines`** | Group 04 (Hybrid Fusion Baselines) | ❌ **NO DATA** — Log files pending upload in `results/_incoming/` |

---

## 2. Table-by-Table Verification Matrix (Published vs Reconstructed)

### Table 4: PhoBERT-base-v2 Fusion Ablation

| Dataset | Model | Published Paper Value | Reconstructed Value | Status |
| :--- | :--- | :---: | :---: | :---: |
| **UIT-VSFC** | Baseline | 76.30 ± 0.45 | *No data* | ❌ NO DATA |
| **UIT-VSFC** | RawConcat | 75.15 ± 0.62 | *No data* | ❌ NO DATA |
| **UIT-VSFC** | RawGate (Proposed) | 76.40 ± 0.38 | *No data* | ❌ NO DATA |
| **UIT-VSFC** | DenseConcat | 75.87 ± 0.51 | *No data* | ❌ NO DATA |
| **UIT-VSFC** | DenseGate | 76.64 ± 0.42 | *No data* | ❌ NO DATA |
| **VSFC-Ekman** | Baseline | 65.01 ± 0.72 | *No data* | ❌ NO DATA |
| **VSFC-Ekman** | RawConcat | 63.80 ± 0.85 | *No data* | ❌ NO DATA |
| **VSFC-Ekman** | RawGate (Proposed) | 65.93 ± 0.61 | *No data* | ❌ NO DATA |
| **VSFC-Ekman** | DenseConcat | 64.50 ± 0.78 | *No data* | ❌ NO DATA |
| **VSFC-Ekman** | DenseGate | 66.10 ± 0.65 | *No data* | ❌ NO DATA |
| **UIT-VSMEC** | Baseline | 65.13 ± 0.88 | *No data* | ❌ NO DATA |
| **UIT-VSMEC** | RawConcat | 64.20 ± 0.95 | *No data* | ❌ NO DATA |
| **UIT-VSMEC** | RawGate (Proposed) | 65.28 ± 0.75 | *No data* | ❌ NO DATA |
| **UIT-VSMEC** | DenseConcat | 64.80 ± 0.91 | *No data* | ❌ NO DATA |
| **UIT-VSMEC** | DenseGate | 65.50 ± 0.82 | *No data* | ❌ NO DATA |

---

### Table 5: ViSoBERT Fusion Ablation

| Dataset | Model | Published Paper Value | Reconstructed Value | Status |
| :--- | :--- | :---: | :---: | :---: |
| **UIT-VSFC** | Baseline | 90.02 ± 0.15 | *No data* | ❌ NO DATA |
| **UIT-VSFC** | RawConcat | 88.34 ± 0.42 | *No data* | ❌ NO DATA |
| **UIT-VSFC** | RawGate | 89.73 ± 0.28 | *No data* | ❌ NO DATA |
| **UIT-VSFC** | DenseConcat | 90.02 ± 0.20 | *No data* | ❌ NO DATA |
| **UIT-VSFC** | DenseGate | 90.24 ± 0.18 | *No data* | ❌ NO DATA |
| **UIT-VSFC** | Residual gate | 90.21 ± 0.22 | *No data* | ❌ NO DATA |
| **VSFC-Ekman** | Baseline | 60.00 ± 0.92 | *No data* | ❌ NO DATA |
| **VSFC-Ekman** | RawConcat | 58.40 ± 1.10 | *No data* | ❌ NO DATA |
| **VSFC-Ekman** | RawGate | 59.22 ± 0.85 | *No data* | ❌ NO DATA |
| **VSFC-Ekman** | DenseConcat | 58.90 ± 0.98 | *No data* | ❌ NO DATA |
| **VSFC-Ekman** | DenseGate | 59.50 ± 0.88 | *No data* | ❌ NO DATA |
| **VSFC-Ekman** | Residual gate | 55.60 ± 1.05 | *No data* | ❌ NO DATA |
| **UIT-VSMEC** | Baseline | 58.87 ± 0.79 | *No data* | ❌ NO DATA |
| **UIT-VSMEC** | RawConcat | 57.10 ± 0.95 | *No data* | ❌ NO DATA |
| **UIT-VSMEC** | RawGate | 58.39 ± 0.82 | *No data* | ❌ NO DATA |
| **UIT-VSMEC** | DenseConcat | 58.20 ± 0.88 | *No data* | ❌ NO DATA |
| **UIT-VSMEC** | DenseGate | 58.60 ± 0.80 | *No data* | ❌ NO DATA |
| **UIT-VSMEC** | Residual gate | 57.80 ± 0.90 | *No data* | ❌ NO DATA |

---

### Table 6: Knowledge Source Comparison (PhoBERT vs. ViSoBERT)

| Backbone & Dataset | Knowledge Integration | Published Paper Value | Reconstructed Value | Status |
| :--- | :--- | :---: | :---: | :---: |
| **PhoBERT / VSFC-Ekman** | Baseline | 64.72 ± 0.70 | *No data* | ❌ NO DATA |
| **PhoBERT / VSFC-Ekman** | VnEmoLex Concat | 63.18 ± 0.82 | *No data* | ❌ NO DATA |
| **PhoBERT / VSFC-Ekman** | RawGate | 65.93 ± 0.61 | *No data* | ❌ NO DATA |
| **ViSoBERT / VSFC-Ekman** | Baseline | 57.04 ± 0.90 | *No data* | ❌ NO DATA |
| **ViSoBERT / VSFC-Ekman** | VnEmoLex Concat | 57.30 ± 0.88 | *No data* | ❌ NO DATA |
| **ViSoBERT / VSFC-Ekman** | RawGate | 59.49 ± 0.75 | *No data* | ❌ NO DATA |
| **PhoBERT / UIT-VSMEC** | Baseline | 65.13 ± 0.88 | *No data* | ❌ NO DATA |
| **PhoBERT / UIT-VSMEC** | VnEmoLex Concat | 65.28 ± 0.75 | *No data* | ❌ NO DATA |
| **PhoBERT / UIT-VSMEC** | RawGate | 63.26 ± 0.92 | *No data* | ❌ NO DATA |
| **ViSoBERT / UIT-VSMEC** | Baseline | 57.13 ± 0.85 | *No data* | ❌ NO DATA |
| **ViSoBERT / UIT-VSMEC** | VnEmoLex Concat | 58.28 ± 0.80 | *No data* | ❌ NO DATA |
| **ViSoBERT / UIT-VSMEC** | RawGate | 58.39 ± 0.82 | *No data* | ❌ NO DATA |

---

### Table 7: Hybrid Fusion Baselines on ViSoBERT

| Dataset | Model | Published Paper Value | Reconstructed Value | Status |
| :--- | :--- | :---: | :---: | :---: |
| **VSFC-Ekman** | Baseline | 58.11 ± 0.88 | *No data* | ❌ NO DATA |
| **VSFC-Ekman** | ALDONAr | 59.05 ± 0.76 | *No data* | ❌ NO DATA |
| **VSFC-Ekman** | KEAHT | 60.92 ± 0.65 | *No data* | ❌ NO DATA |
| **VSFC-Ekman** | CombViSA | 54.64 ± 1.12 | *No data* | ❌ NO DATA |
| **UIT-VSMEC** | Baseline | 55.92 ± 0.95 | *No data* | ❌ NO DATA |
| **UIT-VSMEC** | ALDONAr | 56.98 ± 0.82 | *No data* | ❌ NO DATA |
| **UIT-VSMEC** | KEAHT | 57.87 ± 0.75 | *No data* | ❌ NO DATA |
| **UIT-VSMEC** | CombViSA | 58.69 ± 0.70 | *No data* | ❌ NO DATA |

---

## 3. Required Action (TODO)

> **TODO (Author Action Required):**  
> Place the `.docx` log files for experiment groups 01, 02, 03, and 04 into `results/_incoming/` so `results/aggregate.py` can parse them, populate the `per_seed.csv` files, and compute the sample standard deviations for full automated verification against the paper's published values.
