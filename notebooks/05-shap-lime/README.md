# Group 05: Post-Hoc Explainability (SHAP & LIME)

This directory contains notebooks performing post-hoc local feature attribution and ROC evaluation using **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)**.

---

## 🗺️ Mapping to Paper Figures & Sections

| Notebook Path | Target Dataset | Figure / Section in Paper | Analyzed Sentence & Target Class |
| :--- | :--- | :--- | :--- |
| **`vsfcekman-roc-lime (2).ipynb`** | **VSFC-Ekman** (7-class) | **Figure 5 (Top)** & Section 5.7.2 | `"giảng viên nhiệt tình trong công tác giảng dạy ."` $\to$ Predicted: `Happiness` (1.00) |
| **`vsmec-lime-roc (1).ipynb`** | **UIT-VSMEC** (7-class) | **Figure 5 (Bottom)** & Section 5.7.2 | `"mấy ai được như vậy ??"` $\to$ Predicted: `Other` (0.94) |
| **`vsfc-lime-roc.ipynb`** | **UIT-VSFC** (3-class) | Section 5.7.2 & ROC Evaluation | `"giảng viên nhiệt tình trong công tác giảng dạy ."` $\to$ Predicted: `Positive` (1.00) |

---

## ⚙️ Explainer Settings & Methodology

- **Model Variant Explained:** `PhoBERT_Fusion_V2(n_classes, fusion_type="gate", ontology_dim=None)` (**RawGate on PhoBERT**).
- **Tokenisation Strategy:**  
  - LIME: `LimeTextExplainer(split_expression=r"\W+")` (splits text on whitespace and non-alphanumeric punctuation into space-separated Vietnamese syllables).
  - SHAP: `shap.maskers.Text(r"\W+")` (whitespace/punctuation regex masker).
- **Sampling Parameters:**
  - SHAP `max_evals`: `150` evaluations per sentence (`batch_size=16`).
  - LIME `num_features`: `10` top features per local explanation.
  - Class explained: Top predicted class (`argmax` of model output probabilities).
