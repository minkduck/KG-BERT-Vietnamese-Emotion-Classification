# VSFC-Ekman Annotation & Derivation Documentation

This document describes the derivation methodology used to convert the 3-class sentiment corpus **UIT-VSFC** into the 7-class emotion dataset **VSFC-Ekman**.

---

## 1. Overview & Dataset Splits

VSFC-Ekman retains the exact sample text and train/dev/test partition of UIT-VSFC:
- **Train:** 11,426 samples
- **Dev:** 1,583 samples
- **Test:** 3,166 samples

---

## 2. Mapping & Labeling Protocol

1. **Mechanical Mapping (Positive & Neutral):**
   - All sentences labeled `Positive` (label `2`) in UIT-VSFC were automatically mapped to `Happiness`.
   - All sentences labeled `Neutral` (label `1`) in UIT-VSFC were automatically mapped to `Neutral`.

2. **Fine-Grained Mapping (Negative):**
   - All sentences labeled `Negative` (label `0`) in UIT-VSFC were mapped into one of five negative/reactive Ekman emotion classes (`Anger`, `Disgust`, `Fear`, `Sadness`, `Surprise`).
   - Candidate emotion suggestions were generated using LLM prompting (Gemini).
   - Final label assignments were determined by the first author's subjective judgment of the primary emotion conveyed by each feedback sentence.

> **TODO (Author Action Required):**  
> Insert the exact Gemini prompt text, model version (e.g., Gemini 1.5 Pro / Flash), API temperature, and access date used for generating initial negative emotion candidates below:
>
> ```markdown
> ### Gemini Labeling Prompt (TODO)
> [Insert exact prompt template here]
> ```

---

## 3. Test Set Label Distribution

The resulting 7-class emotion distribution on the 3,166 test sentences is:

| Emotion Label | Test Sample Count | Mapping Source |
| :--- | :---: | :--- |
| **Happiness** | 1,586 | Mechanical (from UIT-VSFC `Positive`) |
| **Sadness** | 680 | LLM Candidate + Author Judgement (from `Negative`) |
| **Anger** | 576 | LLM Candidate + Author Judgement (from `Negative`) |
| **Neutral** | 167 | Mechanical (from UIT-VSFC `Neutral`) |
| **Disgust** | 137 | LLM Candidate + Author Judgement (from `Negative`) |
| **Fear** | 13 | LLM Candidate + Author Judgement (from `Negative`) |
| **Surprise** | 7 | LLM Candidate + Author Judgement (from `Negative`) |
| **Total** | **3,166** | |

---

## 4. Methodological Note & Usage Guidance

- **No Written Annotation Guidelines:** There was no formal annotation manual.
- **No Trained Annotators / IAA:** Annotations were produced by a single author assisted by LLM suggestions; inter-annotator agreement (e.g., Cohen's $\kappa$ or Krippendorff's $\alpha$) was not calculated.
- **Recommended Usage:** VSFC-Ekman is provided as a secondary, domain-specific evaluation setting for student feedback. **UIT-VSMEC** should be treated as the primary benchmark for 7-class Vietnamese emotion classification.
