# An Explainable Neuro-Symbolic PhoBERT–Ontology Framework for Vietnamese Emotion Classification

Minh-Duc Huynh, Thi-Thu-Thuy Pham  
*Under review at the Journal of Intelligent Information Systems.*

---

## 📌 Overview & Experiment Groups

This repository provides the official implementation and ontology resources for our neuro-symbolic emotion classification framework across three Vietnamese benchmark datasets (**UIT-VSFC**, **VSFC-Ekman**, and **UIT-VSMEC**).

The **proposed method** is **RawGate**: an input-conditioned sigmoid gate, $g = \sigma(W h + b)$, applied to a 24-dimensional ontology vector before concatenation with the sentence representation $h \in \mathbb{R}^{768}$:
$$h_{\text{fused}} = [h; g \odot o]$$

This is combined with a **two-tier explainability framework**:
- **Tier-1 (Intrinsic Symbol Trace):** Sub-symbolic mapping from text tokens to ontology lexical entries, appraisal dimensions, intensity, and polarity via `OntologyEngine.getvector(text, debug=True)`.
- **Tier-2 (Post-hoc Explanations):** Feature attribution and local feature importance explanations via SHAP and LIME (`notebooks/05-shap-lime/`).

The experimental pipeline is organized into **5 evaluation groups**:

1. **`01-phobert-fusion-ablation/`** — Fusion ablation on **PhoBERT-base-v2**. The variants reported in the paper are Baseline, RawConcat, **RawGate (proposed)**, DenseConcat, DenseGate, and RawGate + rule layer (paper Table 4). The notebooks also contain additional exploratory variants (deep projection, wide projection) that are **not** reported in the paper.
2. **`02-visobert-fusion-ablation/`** — The same ablation on **ViSoBERT**, plus a bottleneck residual gate (paper Table 5). An ontology concept cross-attention variant in these notebooks is exploratory and **not** reported in the paper.
3. **`03-lexicon-vs-ontology/`** — Comparison of knowledge sources on both backbones: no external knowledge (Baseline), **VnEmoLex** (Doãn & Lưu, 2022) concatenation, and our ontology with RawGate (paper Table 6).
4. **`04-final-model-comparison/`** — **Re-implemented hybrid fusion baselines**. Re-implementations of the core fusion mechanisms of **ALDONAr**, **KEAHT**, and **CombViSA** on a common ViSoBERT encoder fed our 24-dimensional ontology vector. These are re-implemented baselines for comparison, not reproductions of the original systems, and not our contribution (paper Table 7).
5. **`05-shap-lime/`** — Post-hoc explainability notebooks producing SHAP and LIME feature attributions and multi-class ROC curves (paper Figure 5 and Section 5.7.2).

---

## 🗺️ Paper ↔ Code Mapping Table

| Paper Section / Table / Fig | Experiment / Concept | Notebook Path | Model Class / Flags | Associated Config |
| :--- | :--- | :--- | :--- | :--- |
| **Table 4** | PhoBERT Fusion Ablation | `notebooks/01-phobert-fusion-ablation/*.ipynb` | `PhoBERT_Fusion_V2` (`fusion_type`: `none`, `concat`, `gate`) | `configs/{vsfc, vsfc_ekman, vsmec}.yaml` |
| **Table 5** | ViSoBERT Fusion Ablation | `notebooks/02-visobert-fusion-ablation/*.ipynb` | `ViSoBERT_Fusion` (`none`, `concat`, `gate`), `ViSoBERT_DeepOntology_V2` / `ViSoBERT_Residual_Fusion` (`residual_gate`) | `configs/{vsfc, vsfc_ekman, vsmec}.yaml` |
| **Table 6** | Lexicon vs. Ontology | `notebooks/03-lexicon-vs-ontology/*.ipynb` | `Model_Baseline`, `Model_VnEmoLex_Concat`, `Model_Ontology_RawGate` | `configs/{vsfc, vsfc_ekman, vsmec}.yaml` |
| **Table 7** | Re-implemented Fusion Baselines | `notebooks/04-final-model-comparison/*.ipynb` | `ViSoBERT_Baseline`, `ALDONAr_Model`, `KEAHT_Model`, `CombViSA_Model` | `configs/{vsfc, vsfc_ekman, vsmec}.yaml` |
| **Tables 8–10** | Symbolic Rule Post-Processing | Notebook rule evaluation cells | Inline `NeuroSymbolicReasoner` classes | `configs/{vsfc, vsfc_ekman, vsmec}.yaml` |
| **Figure 4 & Sec 5.7.1** | Intrinsic Symbol Trace | Diagnostic cells & tests | `OntologyEngine.getvector(text, debug=True)` | `configs/*.yaml` |
| **Figure 5 & Sec 5.7.2** | Post-Hoc SHAP & LIME Explanations | `notebooks/05-shap-lime/*.ipynb` | `PhoBERT_Fusion_V2` (`fusion_type="gate"`) | `configs/{vsfc, vsfc_ekman, vsmec}.yaml` |

---

## 📂 Repository Structure

```
.
├── README.md                     # Project overview and reproduction instructions
├── AUDIT_REPORT.md               # Technical audit report matching code to paper
├── LICENSE                       # MIT License
├── CITATION.cff                  # Citation metadata (CFF format)
├── requirements.txt              # Python dependencies
├── .gitignore                    # Version control ignore list
│
├── ontology/
│   ├── README.md                 # Knowledge Graph release notes & version mapping
│   └── ekman_appraisal_ontology.rdf  # Ekman & Cognitive Appraisal Knowledge Graph (OWL/RDF)
│
├── data/
│   ├── README.md                 # Dataset documentation, provenance & licensing notices
│   ├── vsfc/vsfc.zip             # UIT-VSFC Sentiment Dataset (UIT-NLP)
│   ├── vsmec/uit-vsmec.zip       # UIT-VSMEC Emotion Dataset (UIT-NLP)
│   └── vsfc-ekman/
│       ├── vsfcekman.zip         # Derived VSFC-Ekman Emotion Dataset (Authors)
│       └── LABELING.md           # Annotation provenance & derivation protocol
│
├── src/
│   ├── __init__.py
│   ├── utils.py                  # Reproducibility utilities (seed_everything, config loading)
│   ├── ontology_engine.py        # Ekman ontology query engine & vectorizer (VSFC & Ekman/VSMEC)
│   ├── significance_test.py      # Non-parametric paired permutation test module
│   ├── data_loading.py           # Unified data loaders (direct .zip / directory support)
│   └── models/
│       ├── __init__.py
│       ├── phobert_fusion_models.py    # PhoBERT fusion & deep projection architectures
│       ├── visobert_fusion_models.py   # ViSoBERT fusion & cross-attention architectures
│       ├── lexicon_ontology_models.py  # VnEmoLex vs. Ontology comparative models
│       └── final_comparison_models.py  # ViSoBERT Baseline, ALDONAr, KEAHT, and CombViSA
│
├── configs/
│   ├── vsfc.yaml                 # Configuration & parameters for UIT-VSFC (3-class)
│   ├── vsfc_ekman.yaml           # Configuration & parameters for VSFC-Ekman (7-class)
│   └── vsmec.yaml                # Configuration & parameters for UIT-VSMEC (7-class)
│
├── notebooks/
│   ├── 01-phobert-fusion-ablation/     # Group 1: PhoBERT-base-v2 ablation experiments
│   ├── 02-visobert-fusion-ablation/    # Group 2: ViSoBERT ablation experiments
│   ├── 03-lexicon-vs-ontology/         # Group 3: VnEmoLex vs. Ontology experiments
│   ├── 04-final-model-comparison/      # Group 4: Re-implemented hybrid fusion baselines
│   └── 05-shap-lime/                   # Group 5: Post-hoc SHAP & LIME explainability
│
└── results/
    ├── README.md                 # Documentation for experiment logs & reproduction status
    ├── aggregate.py              # Automated per-seed aggregation script (sample std ddof=1)
    └── VERIFICATION.md           # Cell-by-cell paper table verification matrix
```

> **Data Attribution Notice:** UIT-VSFC and UIT-VSMEC belong to their original authors (UIT-NLP); please cite and follow their original terms of use.

---

## ⚙️ Installation & Setup

### 1. Hardware & Environment
- Experiments were run on Kaggle notebooks with 2x NVIDIA T4 GPUs.
- **Python:** $\ge 3.10$
- **PyTorch:** $\ge 2.0$

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/minkduck/KG-BERT-Vietnamese-Emotion-Classification.git
cd KG-BERT-Vietnamese-Emotion-Classification

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Ontology Hyper-Parameters, Random Seeds & Significance Testing

### Hyper-Parameters & Single Committed Run Seeds
The exact ontology vectorizer hyper-parameters and example single-run seeds are preserved in `configs/*.yaml`:

| Configuration | Default Conf | Manual Boost * | Negation Attenuation | Alpha Similarity | Single-Run Seed (PhoBERT) | Single-Run Seed (ViSoBERT) | Single-Run Seed (Lexicon/Onto) | Single-Run Seed (Baselines) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`vsfc.yaml`** | 0.90 | *Not applied* | 0.40 | 0.20 | 123 | 2025 | 42 | 1234 |
| **`vsfc_ekman.yaml`** | 0.85 | *Not applied* | 0.50 | 0.20 | 0 | 2025 | 1234 | 42 |
| **`vsmec.yaml`** | 0.85 | *Not applied* | 0.50 | 0.20 | 1234 | 42 | 42 | 2025 |

> **Notes on Hyper-Parameters and Seeds:**  
> - **Manual Boost \***: Defined in the configuration (`1.2` / `null`) but not applied by the ontology engine, so it has no effect on the outputs.  
> - **Per-Model Seed Columns**: The seed columns in the table above record the seed of the **single committed example run** for each respective group notebook, not the full set.  
> - **Multi-Seed Results**: All results reported in the paper are the **mean and sample standard deviation ($\text{ddof}=1$) over the five seeds `0, 123, 1234, 2025, and 42`**, which were run as separate executions and aggregated afterwards.

### Statistical Significance Testing
- `src/significance_test.py` implements a non-parametric paired sample/prediction-level permutation test ($10,000$ resamples, two-sided test statistic $| \text{metric}_B - \text{metric}_A |$, reporting both Macro-F1 and Accuracy $p$-values).
- **Note:** These tests are exploratory. The paper reports mean and standard deviation over five seeds and does not base any claim on p-values (see Section 4.4 of the paper).

---

## 🌌 Ontology Knowledge Graph Release

The repository includes `ontology/ekman_appraisal_ontology.rdf`:
- **Triple Count:** **91,005** triples
- **Named Individuals:** **11,733**
- **OWL Classes:** **46**
- **Object Properties:** 14 | **Datatype Properties:** 10

> **Note on Ontology Release:** The archived file (`ontology/ekman_appraisal_ontology.rdf`) is the release used for UIT-VSFC; the VSFC-Ekman and UIT-VSMEC experiments used an earlier release that is not archived here. See [`ontology/README.md`](ontology/README.md) for details.

---

## 🔒 Code Availability

Source code and ontology resources: https://github.com/minkduck/KG-BERT-Vietnamese-Emotion-Classification