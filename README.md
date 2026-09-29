# Ontology-Guided Knowledge Injection for Vietnamese Sentiment and Emotion Analysis

Official source code and ontology resources accompanying the research paper on neuro-symbolic knowledge integration for Vietnamese sentiment analysis and emotion classification.

---

## 📌 Overview & Experiment Groups

This repository provides a modular, reproducible implementation of our proposed ontology injection framework alongside comparative baselines across three Vietnamese benchmark datasets (**UIT-VSFC**, **VSFC-Ekman**, and **UIT-VSMEC**).

The experimental pipeline is organized into **4 core evaluation groups**:

1. **`01-phobert-fusion-ablation/` (PhoBERT Fusion Ablation):**
   Ablation study of knowledge fusion strategies on the **PhoBERTv2** backbone (evaluating Baseline, Concatenation, Dynamic Gating, Dense 64d, Deep Ontology Projection, and Wide Projection).
2. **`02-visobert-fusion-ablation/` (ViSoBERT Fusion Ablation):**
   Ablation study of fusion strategies on the **ViSoBERT** backbone (evaluating Baseline, Raw xAI 24d, Dense Adapter, Deep Gated Residual Fusion, and Ontology Concept Cross-Attention).
3. **`03-lexicon-vs-ontology/` (Lexicon vs. Ontology Comparison):**
   Direct empirical comparison between a pre-existing emotion lexicon (**VnEmoLex**, Doãn & Lưu, 2022) and our formal Ekman appraisal ontology, evaluated across both PhoBERT and ViSoBERT architectures (this evaluates knowledge representation quality, not a separate backbone).
4. **`04-final-model-comparison/` (Proposed Model vs. SOTA Literature):**
   Comprehensive benchmarking comparing the baseline ViSoBERT model and established literature SOTA methods (**ALDONAr**, **KEAHT**) against our proposed model (**CombViSA** / *Ontology-Guided Cross-Attention* — the primary contribution of the paper).

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

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python >= 3.10
- PyTorch >= 2.0 with CUDA support (for GPU acceleration during training)

### 2. Environment Setup

```bash
# Clone the repository
git clone [GitHub URL]
cd [repo-name]

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
> **Statistical Significance Testing:** Paired permutation testing (`src.significance_test`) with 10,000 resamples is performed in `02-visobert-fusion-ablation`, `03-lexicon-vs-ontology`, and `04-final-model-comparison`. The initial exploration notebooks in `01-phobert-fusion-ablation` do not include permutation testing as per the original experimental design.

---

## 📊 Ontology Hyper-Parameters & Random Seeds

The exact values used across all experiments are preserved in `configs/*.yaml`:

| Configuration | Default Conf | Manual Boost | Negation Attenuation | Alpha Similarity | PhoBERT Seed | ViSoBERT Seed | Emo Seed | CombViSA Seed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`vsfc.yaml`** | 0.90 | *None* | 0.40 | 0.20 | 123 | 2025 | 42 | 1234 |
| **`vsfc_ekman.yaml`** | 0.85 | 1.20 | 0.50 | 0.20 | 0 | 2025 | 1234 | 42 |
| **`vsmec.yaml`** | 0.85 | 1.20 | 0.50 | 0.20 | 1234 | 42 | 42 | 2025 |

---

## 🔒 Code Availability

The source code, ontology resources, experiment configurations, and reproduction instructions are available at: [GitHub URL].
"# KG-BERT-Vietnamese-Emotion-Classification" 
