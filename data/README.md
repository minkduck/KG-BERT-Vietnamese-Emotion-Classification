# Data Documentation & Attribution

This directory contains the datasets used in the paper, organized into three subsets.

---

## 1. UIT-VSFC (Vietnamese Students' Feedback Corpus)

- **Path:** `data/vsfc/vsfc.zip`
- **Task:** 3-class Sentiment Analysis (`0: Negative`, `1: Neutral`, `2: Positive`)
- **Format:** Text files split into `train/`, `dev/`, and `test/` subfolders, each containing `sents.txt` and `sentiments.txt`.
- **Attribution & Provenance:**
  UIT-VSFC was created and published by the **UIT-NLP Group** (University of Information Technology, VNU-HCM). It is NOT created by the authors of this repository.
  - *Citation:*
    > Nguyen, K. T. V., Nguyen, V. O., & Nguyen, N. L. T. (2018). *UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment Analysis*. In Proceedings of the 2018 10th International Conference on Knowledge and Systems Engineering (KSE), pp. 19-24. IEEE.
- **License / Terms Notice:** Users should verify the original licensing terms and redistribution policies of the UIT-NLP group before public redistribution.

---

## 2. UIT-VSMEC (Vietnamese Social Media Emotion Corpus)

- **Path:** `data/vsmec/uit-vsmec.zip`
- **Task:** 7-class Emotion Classification (`Anger`, `Disgust`, `Fear`, `Happiness`, `Neutral`, `Sadness`, `Surprise`)
- **Format:** Excel spreadsheets: `train_nor_811.xlsx`, `valid_nor_811.xlsx`, `test_nor_811.xlsx`.
- **Attribution & Provenance:**
  UIT-VSMEC was created and published by the **UIT-NLP Group** (University of Information Technology, VNU-HCM). It is NOT created by the authors of this repository.
  - *Citation:*
    > Ho, V. A., Nguyen, D. H., Nguyen, D. V., Pham, Q. T., Nguyen, K. T. V., & Nguyen, N. L. T. (2020). *Emotion Recognition for Vietnamese Social Media Text using Pre-trained Language Models*. In Proceedings of the 7th International Conference on Asian Language Processing (IALP), pp. 197-202. IEEE.
- **License / Terms Notice:** Users should verify the original licensing terms and redistribution policies of the UIT-NLP group before public redistribution.

---

## 3. VSFC-Ekman (Derived Emotion Dataset)

- **Path:** `data/vsfc-ekman/vsfcekman.zip`
- **Task:** 7-class Emotion Classification on Student Feedback Text.
- **Format:** CSV files: `train.csv`, `dev.csv`, `test.csv` (Columns: `Sentence`/`Text`, `Emotion`/`Ekman_Label`, `Original_Sentiment`).
- **Attribution & Provenance:**
  VSFC-Ekman is a **derived dataset created by the authors of this study** by mapping and annotating the UIT-VSFC sentences into 7 Ekman emotion categories using the Ekman appraisal ontology and human verification.
- **Availability:** This derived dataset is freely available for academic research and reproducible evaluation in connection with our paper.

---

## Data Loading Utility

All dataset splits can be loaded seamlessly via `src/data_loading.py`:

```python
from src.data_loading import load_vsfc, load_vsfc_ekman, load_vsmec

# Loads train, valid, test DataFrames directly from .zip archives or extracted folders
df_train, df_val, df_test = load_vsfc("data/vsfc")
df_train, df_val, df_test = load_vsfc_ekman("data/vsfc-ekman")
df_train, df_val, df_test = load_vsmec("data/vsmec")
```
