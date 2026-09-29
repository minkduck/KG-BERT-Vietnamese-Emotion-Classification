# An Explainable Neuro-Symbolic PhoBERT–Ontology Framework for Vietnamese Emotion Classification

Official source code and ontology resources for the paper:
> **An Explainable Neuro-Symbolic PhoBERT–Ontology Framework for Vietnamese Emotion Classification**  
> Minh-Duc Huynh, Thi-Thu-Thuy Pham  
> *Submitted to Journal of Intelligent Information Systems*

### Paper Citation (BibTeX Placeholder)
```bibtex
@article{huynh2026explainable,
  title={An Explainable Neuro-Symbolic PhoBERT--Ontology Framework for Vietnamese Emotion Classification},
  author={Huynh, Minh-Duc and Pham, Thi-Thu-Thuy},
  journal={Submitted to Journal of Intelligent Information Systems},
  year={2026}
}
```

---

## 📌 Overview & Method Summary

This repository provides a modular, reproducible implementation of our proposed neuro-symbolic framework alongside comparative baselines across three Vietnamese benchmark datasets (**UIT-VSFC**, **VSFC-Ekman**, and **UIT-VSMEC**).

The **proposed method** in the paper is **RawGate** — an input-conditioned gated fusion mechanism that dynamically weights a 24-dimensional ontology vector $o \in \mathbb{R}^{24}$ using the transformer text representation $h \in \mathbb{R}^{768}$:
$$g = \sigma(W_g h + b_g), \quad h_{\text{fused}} = [h; g \odot o]$$

This is paired with a **two-tier explainability framework**:
- **Tier-1 (Intrinsic Symbol Trace):** Sub-symbolic mapping from text tokens to ontology lexical entries, appraisal dimensions, intensity, and polarity. Accessible via `OntologyEngine.getvector(text, debug=True)`.
- **Tier-2 (Post-hoc Explanations):** Feature attribution and local feature importance explanations via SHAP and LIME (`notebooks/05-shap-lime/`).

---

## 🧪 Evaluation Groups

The experimental pipeline is organized into **5 core evaluation groups**:

1. **`01-phobert-fusion-ablation/`** — Knowledge fusion ablation on **PhoBERT-base-v2**: Baseline, RawConcat, **RawGate (proposed)**, DenseConcat, DenseGate, and RawGate + rule layer (paper Table 4).
2. **`02-visobert-fusion-ablation/`** — Knowledge fusion ablation on **ViSoBERT**, including Residual Gate fusion (paper Table 5).
3. **`03-lexicon-vs-ontology/`** — Knowledge-source comparison across backbones: Baseline (no external knowledge), **VnEmoLex** (Doãn & Lưu, 2022) raw concatenation, and our formal Ekman appraisal ontology with RawGate (paper Table 6).
4. **`04-final-model-comparison/`** — Re-implemented hybrid fusion baselines (**ALDONAr**, **KEAHT**, **CombViSA**) evaluated on ViSoBERT. All three re-implementations are fed the exact same 24-dimensional ontology vector as our models (paper Table 7).
5. **`05-shap-lime/`** — Post-hoc explainability notebooks producing SHAP and LIME feature attributions and multi-class ROC curves (paper Figure 5 & Section 5.7.2).

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

## 🔬 Architecture Details: Residual Gate Variants

The **Residual Gate** ablation in Group 02 uses dataset-specific projection pipelines suited to each dataset's taxonomy depth:

- **UIT-VSFC (`ViSoBERT_DeepOntology_V2`):** Two-layer bottleneck projection ($24 \to 64 \to 768$) with LayerNorm:  
  $h_{\text{fused}} = h_{\text{text}} + \text{Dropout}(z \odot h_{\text{ont}})$, where $z = \sigma(W [h_{\text{text}}; h_{\text{ont}}] + b)$.
- **UIT-VSMEC (`ViSoBERT_DeepOntology`):** Two-layer bottleneck projection ($24 \to 64 \to 768$) with BatchNorm1d.
- **VSFC-Ekman (`ViSoBERT_Residual_Fusion`):** Two-step projection ($24 \to 256 \to 768$) with LayerNorm on the intermediate 256-d layer:  
  $h_{\text{fused}} = \text{LayerNorm}(h_{\text{text}} + \text{Sigmoid}(W [h_{\text{text}}; h_{\text{ont}}]) \odot h_{\text{ont}})$.

---

## 🔬 Additional Exploratory Ablations (Not Reported in Paper)

Some notebooks contain extra architectural variants created during exploration that are retained for completeness:

- **Deep Projection Adapter (`PhoBERT_DeepOntology` / `ViSoBERT_DeepOntology`):** Multi-layer non-linear bottleneck projection ($24 \to 64 \to 768$) prior to fusion.
- **Wide Projection Adapter (`PhoBERT_WideProjection`):** Direct single-layer wide projection ($24 \to 768$) with dynamic gating.
- **Ontology Concept Cross-Attention (`ViSoBERT_Ontology_CrossAttention` / OCA):** Slices the 24-d ontology vector into 5 group concept tokens ($768$-d each via LayerNorm+ReLU) and applies 8-head multi-head cross-attention over full token sequences.
- **Deep Gated Interpolation (`fusion_type="gate"` in deep/wide models):** Soft convex interpolation gating $z \odot h_{\text{text}} + (1 - z) \odot h_{\text{ont}}$.

---

## 📂 Repository Structure

```
.
├── README.md                     # Project overview and reproduction instructions
├── AUDIT_REPORT.md               # Comprehensive technical audit report matching code to paper
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
    ├── README.md                 # Documentation for experiment logs & CSV artifacts
    ├── aggregate.py              # Automated per-seed aggregation script (sample std ddof=1)
    └── VERIFICATION.md           # Cell-by-cell paper table verification matrix
```

> **Data Attribution Notice:** UIT-VSFC and UIT-VSMEC belong to their original authors (UIT-NLP); please cite and follow their original terms of use.

---

## ⚙️ Installation & Setup

### 1. Hardware & Environment
- **Hardware used for paper experiments:** Kaggle Notebooks environment with 2× NVIDIA T4 GPUs (CUDA acceleration).
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

## 📊 Hyper-Parameters, Random Seeds & Significance Testing

### Hyper-Parameters
The exact ontology vectorizer hyper-parameters are preserved in `configs/*.yaml`:

| Configuration | Default Conf | Manual Boost | Negation Attenuation | Alpha Similarity |
| :--- | :---: | :---: | :---: | :---: |
| **`vsfc.yaml`** | 0.90 * | *Defined (null)* | 0.40 | 0.20 |
| **`vsfc_ekman.yaml`** | 0.85 * | *Defined (1.2)* | 0.50 | 0.20 |
| **`vsmec.yaml`** | 0.85 * | *Defined (1.2)* | 0.50 | 0.20 |

> **Audit Notes on Hyper-Parameters:**  
> - `default_conf` *: Fallback confidence for evocation triples lacking `confidenceScore`. Audit A9.1 confirmed 100% of RDF evocation triples possess an explicit `confidenceScore`, so `default_conf` is unused fallback logic in practice.  
> - `manual_boost`: Parameter defined in config files (`1.2` / `null`) but not applied in `OntologyEngine` vectorizer logic (dead parameter; no effect on output vectors).

### Multi-Seed Execution Protocol
- Each notebook is executed **one seed at a time** in the exact order: **`0, 123, 1234, 2025, 42`**.
- To reproduce all 5 runs for a given experiment, edit the seed variable in the configuration cell at the top of the notebook:  
  `seed = 0` $\to$ `123` $\to$ `1234` $\to$ `2025` $\to$ `42` (or set `seeds.<group>` in `configs/*.yaml`).
- All main paper results (mean $\pm$ sample standard deviation $s$, $\text{ddof}=1$) are aggregated manually across those 5 separate runs using [`results/aggregate.py`](results/aggregate.py).

### Statistical Significance Testing
- `src/significance_test.py` implements a non-parametric paired sample/prediction-level permutation test ($10,000$ resamples, two-sided test statistic $| \text{metric}_B - \text{metric}_A |$, reporting both Macro-F1 and Accuracy $p$-values).
- **Note:** These tests are exploratory; the paper reports mean $\pm$ standard deviation over five random seeds and does not base primary claims on $p$-values (see Section 4.4 of the paper).

---

## 🌌 Ontology Knowledge Graph Release

The repository includes `ontology/ekman_appraisal_ontology.rdf`:
- **Triple Count:** **91,005** triples
- **Named Individuals:** **11,733**
- **OWL Classes:** **46**
- **Object Properties:** 14 | **Datatype Properties:** 10

This file corresponds to the **UIT-VSFC revision release** (91,005 triples). See [`ontology/README.md`](ontology/README.md) for details regarding release history.

---

## 🛠️ Components & Availability Status

### Included in Repository
- ✅ Ontology vectorizer & query engine (`src/ontology_engine.py`)
- ✅ Model architectures (`src/models/*.py`)
- ✅ Data loaders (`src/data_loading.py`)
- ✅ Significance testing module (`src/significance_test.py`)
- ✅ Self-contained experiment notebooks (`notebooks/*/*.ipynb`)
- ✅ Post-hoc SHAP & LIME explainability notebooks (`notebooks/05-shap-lime/`)
- ✅ Automated results aggregation & verification tools (`results/aggregate.py`, `results/VERIFICATION.md`)

### Not Yet Included (TODOs)
- ❌ **Standalone Tier-1 Trace Export CLI:** TODO: add command-line tool to export structured XML/JSON traces (currently available in-memory via `engine.getvector(text, debug=True)`).
- ❌ **Standalone Lexicon Build Script:** TODO: add offline RDF-to-lexicon exporter (currently executed at runtime during `OntologyEngine` initialization).
- ❌ **VSFC-Ekman Derivation Script & Prompt:** TODO: add exact LLM prompt text for negative sentence mapping (see `data/vsfc-ekman/LABELING.md`).

---

## 🔒 Code Availability

Source code and ontology resources: https://github.com/minkduck/KG-BERT-Vietnamese-Emotion-Classification