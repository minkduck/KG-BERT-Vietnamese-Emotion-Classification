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
- **Licensing & Redistribution Notice:**
  Original source: [UIT-VSFC Github Repository / Publication](https://github.com/uitnlp/UIT-VSFC).  
  *Note:* Redistribution terms for the bundled UIT-VSFC archives should be confirmed against UIT-NLP's terms of use.

---

## 2. UIT-VSMEC (Vietnamese Social Media Emotion Corpus)

- **Path:** `data/vsmec/uit-vsmec.zip`
- **Task:** 7-class Emotion Classification (`Anger`, `Disgust`, `Fear`, `Happiness`, `Neutral`, `Sadness`, `Surprise`)
- **Format:** Excel spreadsheets: `train_nor_811.xlsx`, `valid_nor_811.xlsx`, `test_nor_811.xlsx`.
- **Attribution & Provenance:**
  UIT-VSMEC was created and published by the **UIT-NLP Group** (University of Information Technology, VNU-HCM). It is NOT created by the authors of this repository. It serves as the **primary emotion classification benchmark** in this paper.
  - *Citation:*
    > Ho, V. A., Nguyen, D. H., Nguyen, D. V., Pham, Q. T., Nguyen, K. T. V., & Nguyen, N. L. T. (2020). *Emotion Recognition for Vietnamese Social Media Text using Pre-trained Language Models*. In Proceedings of the 7th International Conference on Asian Language Processing (IALP), pp. 197-202. IEEE.
- **Licensing & Redistribution Notice:**
  Original source: [UIT-VSMEC Publication](https://ieeexplore.ieee.org/document/9308544).  
  *Note:* Redistribution terms for the bundled UIT-VSMEC archives should be confirmed against UIT-NLP's terms of use.

---

## 3. VSFC-Ekman (Derived Secondary Emotion Dataset)

- **Path:** `data/vsfc-ekman/vsfcekman.zip`
- **Task:** 7-class Ekman Emotion Classification on Student Feedback Text.
- **Format:** CSV files: `train.csv`, `dev.csv`, `test.csv` (Columns: `Sentence`/`Text`, `Emotion`/`Ekman_Label`, `Original_Sentiment`).
- **Provenance & Derivation Protocol:**
  - VSFC-Ekman re-labels **UIT-VSFC** keeping the original splits (Train: 11,426; Dev: 1,583; Test: 3,166).
  - Positive was mapped to `Happiness` and Neutral kept as `Neutral`.
  - For Negative sentences, a candidate emotion label was first suggested by a commercial large language model (Google Gemini) accessed through its public web interface, then accepted or corrected by the first author.
  - There were no written annotation guidelines, no trained annotators and no measured inter-annotator agreement, so **UIT-VSMEC is the primary emotion benchmark** and VSFC-Ekman a secondary, taxonomy-aligned setting.
  - > **TODO:** Record the exact Gemini model version and access month.
- **Test-Set Label Distribution:**
  - `Happiness`: 1,586
  - `Sadness`: 680
  - `Anger`: 576
  - `Neutral`: 167
  - `Disgust`: 137
  - `Fear`: 13
  - `Surprise`: 7
  - *Total Test Samples:* 3,166
- **Further Documentation:** See `data/vsfc-ekman/LABELING.md`.

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
