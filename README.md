# An Explainable Neuro-Symbolic PhoBERT–Ontology Framework for Vietnamese Emotion Classification

Official source code and ontology resources for the paper of the same title (Huynh & Pham), submitted to the *Journal of Intelligent Information Systems*.

---

## 📌 Overview & Experiment Groups

This repository provides a modular, reproducible implementation of our proposed ontology injection framework alongside comparative baselines across three Vietnamese benchmark datasets (**UIT-VSFC**, **VSFC-Ekman**, and **UIT-VSMEC**).

The experimental pipeline is organized into **4 core evaluation groups**:

1. **`01-phobert-fusion-ablation/`** — Fusion ablation on **PhoBERT-base-v2**: baseline, RawConcat, **RawGate (proposed)**, DenseConcat, DenseGate, and RawGate + rule layer (paper Table 4).
2. **`02-visobert-fusion-ablation/`** — The same ablation on **ViSoBERT**, plus a residual gate (paper Table 5).
3. **`03-lexicon-vs-ontology/`** — Knowledge-source comparison on both backbones: no external knowledge, **VnEmoLex** (Doãn & Lưu, 2022) concatenation, and our ontology with RawGate (paper Table 6).
4. **`04-final-model-comparison/`** — Re-implementations of the fusion mechanisms of **ALDONAr**, **KEAHT** and **CombViSA** on ViSoBERT. All three are fed the same 24-dimensional ontology vector as our models. These are re-implemented baselines, not the original systems (paper Table 7).

Some notebooks contain additional exploratory variants (e.g. deep or wide ontology projections) that are not reported in the paper.

---

## 📂 Repository Structure

```
.
├── README.md                     # Project overview and reproduction instructions
├── LICENSE                       # MIT License
├── CITATION.cff                  # Citation metadata
├── requirements.txt              # Python dependencies
├── .gitignore                    # Version control ignore list
│
├── ontology/
│   └── ekman_appraisal_ontology.rdf  # Ekman & Cognitive Appraisal Ontology (OWL/RDF)
│
├── data/
│   ├── README.md                 # Dataset documentation and provenance
│   ├── vsfc/vsfc.zip             # UIT-VSFC Sentiment Dataset (UIT-NLP)
│   ├── vsmec/uit-vsmec.zip       # UIT-VSMEC Emotion Dataset (UIT-NLP)
│   └── vsfc-ekman/vsfcekman.zip  # Derived VSFC-Ekman Emotion Dataset (Authors)
│
├── src/
│   ├── __init__.py
│   ├── utils.py                  # Reproducibility utilities (seed_everything, config loading)
│   ├── ontology_engine.py        # Ekman ontology query engine & vectorizer (VSFC & Ekman/VSMEC)
│   ├── significance_test.py      # Non-parametric paired permutation test
│   ├── data_loading.py           # Unified data loaders (direct .zip / directory support)
│   └── models/
│       ├── __init__.py
│       ├── phobert_fusion_models.py    # PhoBERT fusion & deep projection architectures
│       ├── visobert_fusion_models.py   # ViSoBERT fusion & cross-attention architectures
│       ├── lexicon_ontology_models.py  # VnEmoLex vs. Ontology comparative models
│       └── final_comparison_models.py  # ViSoBERT Baseline, ALDONAr, KEAHT, and CombViSA
│
├── configs/
│   ├── vsfc.yaml                 # Configuration & seeds for UIT-VSFC (3-class)
│   ├── vsfc_ekman.yaml           # Configuration & seeds for VSFC-Ekman (7-class)
│   └── vsmec.yaml                # Configuration & seeds for UIT-VSMEC (7-class)
│
├── notebooks/
│   ├── 01-phobert-fusion-ablation/     # Group 1: PhoBERTv2 ablation experiments
│   │   ├── vsfc.ipynb
│   │   ├── vsfc_ekman.ipynb
│   │   └── vsmec.ipynb
│   ├── 02-visobert-fusion-ablation/    # Group 2: ViSoBERT ablation experiments
│   │   ├── vsfc.ipynb
│   │   ├── vsfc_ekman.ipynb
│   │   └── vsmec.ipynb
│   ├── 03-lexicon-vs-ontology/         # Group 3: VnEmoLex vs. Ontology experiments
│   │   ├── vsfc.ipynb
│   │   ├── vsfc_ekman.ipynb
│   │   └── vsmec.ipynb
│   └── 04-final-model-comparison/      # Group 4: Final model vs. SOTA literature
│       ├── vsfc.ipynb
│       ├── vsfc_ekman.ipynb
│       └── vsmec.ipynb
│
└── results/
    └── .gitkeep                  # Experiment artifacts and output logs
```

UIT-VSFC and UIT-VSMEC belong to their original authors (UIT-NLP); please cite and follow their terms of use.

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python >= 3.10
- PyTorch >= 2.0 with CUDA support (for GPU acceleration during training)

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

## 🚀 Running Experiments

Each notebook in `notebooks/` is self-contained and pre-configured with exact experiment hyper-parameters and random seeds corresponding to `configs/*.yaml`:

| Experiment Category | Notebook Path | Associated Config |
| :--- | :--- | :--- |
| **01. PhoBERT Fusion Ablation** | `notebooks/01-phobert-fusion-ablation/{vsfc, vsfc_ekman, vsmec}.ipynb` | `configs/{vsfc, vsfc_ekman, vsmec}.yaml` |
| **02. ViSoBERT Fusion Ablation** | `notebooks/02-visobert-fusion-ablation/{vsfc, vsfc_ekman, vsmec}.ipynb` | `configs/{vsfc, vsfc_ekman, vsmec}.yaml` |
| **03. Lexicon vs. Ontology** | `notebooks/03-lexicon-vs-ontology/{vsfc, vsfc_ekman, vsmec}.ipynb` | `configs/{vsfc, vsfc_ekman, vsmec}.yaml` |
| **04. Final Model Comparison** | `notebooks/04-final-model-comparison/{vsfc, vsfc_ekman, vsmec}.ipynb` | `configs/{vsfc, vsfc_ekman, vsmec}.yaml` |

> [!NOTE]
> **Data Handling:** Notebooks use `src.data_loading` which automatically parses `.zip` archives in `data/` or extracted directories. No manual unzipping is required.
>
> **Statistical testing:** `src/significance_test.py` implements a paired permutation test (10,000 resamples), used during exploration in groups 02–04. The paper reports mean ± standard deviation over random seeds and does not report p-values; see Section 4.4 of the paper.

---

## 📊 Ontology Hyper-Parameters & Random Seeds

The exact values used across all experiments are preserved in `configs/*.yaml`:

| Configuration | Default Conf | Negation Attenuation | Alpha Similarity |
| :--- | :---: | :---: | :---: |
| **`vsfc.yaml`** | 0.90 | 0.40 | 0.20 |
| **`vsfc_ekman.yaml`** | 0.85 | 0.50 | 0.20 |
| **`vsmec.yaml`** | 0.85 | 0.50 | 0.20 |

All reported results are averaged over the seeds 0, 42, 123, 1234 and 2025; the ViSoBERT ablation uses fewer seeds on two datasets, as stated in the caption of paper Table 5.

---

## 🔒 Code Availability

The source code, ontology resources, experiment configurations, and reproduction instructions are available at: https://github.com/minkduck/KG-BERT-Vietnamese-Emotion-Classification.