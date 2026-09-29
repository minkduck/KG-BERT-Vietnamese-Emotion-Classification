# AUDIT REPORT — An Explainable Neuro-Symbolic PhoBERT–Ontology Framework for Vietnamese Emotion Classification

> Generated 2026-09-29.  
> Ground rule: code is the source of truth for what was run; paper is the source of truth for naming and claims.

---

## A1. Model ↔ Paper Name Mapping

### Mapping table

| Paper name | Code class | Flags / constructor | Notes |
|---|---|---|---|
| **PhoBERT-base-v2 Baseline** | `PhoBERT_Fusion_V2` | `fusion_type="none"` | Uses `pooler_output` (not CLS) |
| **ViSoBERT Baseline** | `ViSoBERT_Fusion` | `fusion_type="none"` | Uses `last_hidden_state[:,0,:]` (CLS) |
| **ViSoBERT Baseline** (group 04) | `ViSoBERT_Baseline` | — | Distinct class; also uses CLS |
| **ViSoBERT Baseline** (group 03) | `Model_Baseline` | `model_name="uitnlp/visobert"` | Fallback chain: CLS / pooler / outputs[0][:,0,:] |
| **RawConcat** (raw 24-d concat) | `PhoBERT_Fusion_V2` / `ViSoBERT_Fusion` | `fusion_type="concat"`, `ontology_dim=None` | ont_adapter = Identity(); comb = cat([h_text, o], dim=1) |
| **RawGate (proposed)** | `PhoBERT_Fusion_V2` / `ViSoBERT_Fusion` / `Model_Ontology_RawGate` | `fusion_type="gate"`, `ontology_dim=None` | g = sigmoid(Linear(768→24)(h_text)); gated_o = o * g; comb = cat([h_text, gated_o]) |
| **DenseConcat** (64-d adapter + concat) | `PhoBERT_Fusion_V2` / `ViSoBERT_Fusion` | `fusion_type="concat"`, `ontology_dim=64` | ont_adapter = Linear(24→64)+BN+ReLU+Drop; comb = cat([h_text, o_64]) |
| **DenseGate** (64-d adapter + gate) | `PhoBERT_Fusion_V2` / `ViSoBERT_Fusion` | `fusion_type="gate"`, `ontology_dim=64` | same gate formula on 64-d adapted ontology |
| **Residual gate** (ViSoBERT/VSFC) | `ViSoBERT_DeepOntology_V2` | `fusion_type="residual_gate"`, `hidden_dim=64` | 24→64→768 via LayerNorm; h_fused = h_text + Dropout(z ⊙ h_ont); z = sigmoid(W·[h_text;h_ont]) |
| **Residual gate** (ViSoBERT/VSMEC) | `ViSoBERT_DeepOntology` | `fusion_type="residual_gate"`, `hidden_dim=64` | 24→64→768 via BatchNorm1d; same residual formula |
| **Residual gate** (ViSoBERT/VSFC-Ekman) | `ViSoBERT_Residual_Fusion` | — | 2-step projection 24→256→768 (LayerNorm on 256); h_fused = LayerNorm(h_text + gate ⊙ h_ont) then Dropout |
| **VnEmoLex Concat** | `Model_VnEmoLex_Concat` | `lex_dim=7` | Raw 7-d lexicon concat, no projection |
| **ALDONAr** (re-impl. baseline) | `ALDONAr_Model` | — | Projects ont 24→768 (Linear+LN+ReLU+Drop); learns 2-way softmax attention weights w_t, w_o; fused = w_t·h_text + w_o·h_ont |
| **KEAHT** (re-impl. baseline) | `KEAHT_Model` | `n_heads=8` | **4 concept tokens**: emo(7→768), app(9→768), pol(3→768), neg+int merged(5→768) via bare Linear; 8-head cross-attn query=text_seq, key/val=k_concepts; residual CLS |
| **CombViSA** (re-impl. baseline) | `CombViSA_Model` | — | BiLSTM hidden=256, bidirectional, 1 layer; max-pool → 512; ont branch 24→128 (BN+ReLU+Drop); concat→640; head Linear(640→256)+ReLU+Drop(0.2)+Linear(256→n_classes) |

### Paper description vs code — flag discrepancies

| Model | Paper description | Code reality | Match? |
|---|---|---|---|
| RawGate | g = sigmoid(W h + b); fused = [h; g ⊙ o] | g = sigmoid(Linear(768→24)(h_text)); gated_o = o * g; comb = cat([h_text, gated_o]) — **identical** | ✅ |
| Residual gate (VSFC paper) | h_fused = h_text + Dropout(z ⊙ h_ont), z = sigmoid(W[h_text; h_ont]+b), h_ont = Linear(o)→768 | `ViSoBERT_DeepOntology_V2`: 24→64→768 (two layers), then z = sigmoid(Linear(1536→768)(cat)), h_fused = h_text + Dropout(z⊙h_ont). Paper says "Linear(o)→768" (one step); code uses two steps (24→64→768). | ⚠️ PARTIAL |
| Residual gate (VSFC-Ekman) | same | `ViSoBERT_Residual_Fusion`: 24→256→768 (two steps via Sequential), gate = Sequential(Linear(1536→768), Sigmoid()), h_fused = LayerNorm(h_txt + gate ⊙ h_ont); no Dropout on ont_info (unlike V1/V2). Different from paper formula. | ⚠️ PARTIAL |
| KEAHT | "4 concept tokens + 8-head cross-attention" | **4 tokens**, 8 heads. BUT intensity(3-dim) and negation(2-dim) are **merged** into one 5-dim token (c4 = proj_neg(cat(int,neg))), not kept separate. | ⚠️ NOTE |
| CombViSA | "BiLSTM 256/dir + max pool, 128-unit ontology branch, 2-layer head 256" | BiLSTM hidden=256 bidirectional, max-pool (✅); ont branch Linear(24→128)+BN+ReLU+Drop (✅); head Linear(640→256)+ReLU+Drop+Linear(256→n_classes) (✅) | ✅ |
| ALDONAr | "attention-weighted text–ontology sum" | Softmax 2-weight over cat([h_text,h_ont])→2; fused = w_t·h_text + w_o·h_ont. Matches description. | ✅ |

### Extra ablations NOT in the paper

These classes exist in `src/models/` and were run in exploratory notebook cells but **are not reported** in any paper table:

| Class | File | Description |
|---|---|---|
| `PhoBERT_DeepOntology` | `phobert_fusion_models.py` | 24→64→768 two-layer projection, fusion_type ∈ {concat, gate, add} |
| `PhoBERT_WideProjection` | `phobert_fusion_models.py` | 24→768 single-step wide projection, fusion_type ∈ {concat, gate} |
| `ViSoBERT_Fusion` (as a class) | `visobert_fusion_models.py` | The ViSoBERT counterpart of `PhoBERT_Fusion_V2`; also used for reported models m1–m5 in group 02 |
| `OntologyConceptProjector` | `visobert_fusion_models.py` | 5-head sub-vector projector; helper for OCA |
| `ViSoBERT_Ontology_CrossAttention` (OCA) | `visobert_fusion_models.py` | 5-concept-token cross-attention ablation; appears in group 02 notebook but NOT in any paper table |
| `ViSoBERT_DeepOntology` / `_V2` with `fusion_type="gate"` | `visobert_fusion_models.py` | Soft-interpolation gate (z·text + (1-z)·ont); only the `residual_gate` branch is the paper's Residual gate |
| `ViSoBERT_DeepOntology` / `_V2` with `fusion_type="concat"` | `visobert_fusion_models.py` | Deep concat branch; not reported |

---

## A2. Significance Test

**Source:** `src/significance_test.py` (54 lines) + notebook cells.

| Property | Value |
|---|---|
| **Unit of permutation** | **Sample-level (prediction-level).** For each of 10,000 permutations, a Bernoulli(0.5) swap vector of length n_samples is drawn; individual predictions from model A and model B are independently swapped position-by-position. |
| **Test statistic** | Absolute difference of metric scores: `obs_diff = |metric(y_true, pred_b) − metric(y_true, pred_a)|`. Same absolute criterion applies to each permuted pair. |
| **Number of resamples** | 10,000 (default; all notebook calls use the default). |
| **One- vs two-sided** | **Two-sided** — `np.abs(...)` applied to both observed and permuted differences. |
| **p-value formula** | `p = (count_where_perm_diff ≥ obs_diff + 1) / (10000 + 1)` — Phipson & Smyth +1 correction. |
| **Metric** | Macro-averaged F1 (via `macro_f1()` helper) and Accuracy (via `accuracy_score`). Both reported in notebooks. |
| **General or fixed pairs** | General: accepts any two prediction arrays and any metric callable. |

---

## A3. Ontology Engine

### Default conf

`default_conf` is the **fallback confidence score** for RDF `Evocation` triples that lack an explicit `ekman:confidenceScore` property.

```python
entry_score = float(r.ev or self.default_conf) * float(r.base or 1.0)
```

- `OntologyEngine` default: **0.85** (Ekman/VSMEC tasks).
- `VSFCOntologyEngine` default: **0.90** (UIT-VSFC sentiment task).
- Matches configs: `vsfc.yaml: 0.9`, `vsfc_ekman.yaml: 0.85`, `vsmec.yaml: 0.85`.

### Manual boost

`manual_boost` is stored as `self.manual_boost` but is **never applied anywhere in the codebase**. The parameter is assigned in `__init__` but no code path references it during vectorization. Config values (1.2 for Ekman/VSMEC, `null`/1.0 for VSFC) have no effect on outputs.

### Negation attenuation

- `OntologyEngine`: multiplier **0.5** (Ekman/VSMEC).
- `VSFCOntologyEngine`: multiplier **0.4** (VSFC).
- When a negation cue is matched, `vn[0]` is set to 1.0 and the cue entry is skipped.
- ALL subsequent matched tokens in the sentence have their score multiplied by the attenuation factor.
- There is **no window limit** — `vn[0]` is never reset, so attenuation applies to all remaining tokens.

### Alpha (alpha_similarity)

Value: **0.2** (all datasets, all configs — confirmed).

```python
ve = np.clip(ve + alpha * (ve @ sim_matrix), 0, None)
```

Applied AFTER all token accumulation, BEFORE tanh. The 7×7 similarity matrix is row-normalized before use, so the product is a weighted neighbour-smoothed version of ve. Alpha controls the additive weight of this smoothing.

### 24-dimension breakdown

| Dims | Variable | Content |
|---|---|---|
| 0–6 | ve (7) | Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise |
| 7–15 | va (9) | GoalObstruction, Pleasantness, Unpleasantness, Dangerousness, Loss, Suddenness, LowPredictability, Unpredictability, Unfairness |
| 16–18 | vi (3) | HighIntensity, MediumIntensity, LowIntensity |
| 19–21 | vp (3) | PositivePolarity, NegativePolarity, NeutralPolarity |
| 22–23 | vn (2) | vn[0] = negation flag (1.0 if cue seen); vn[1] = always 0 |

---

## A4. Rule Layer

The rule layer is implemented as an inline `NeuroSymbolicReasoner` / `NeuroSymbolicReasonerVSMEC` class, defined per-notebook (not in `src/`). Thresholds:

| Dataset | Condition for override | Threshold | Config key |
|---|---|---|---|
| UIT-VSFC | `pred == Neutral` AND `confidence < model_conf_limit` AND `polarity_signal > ont_threshold` | `model_conf_limit=0.85`, `ont_threshold=0.55` | In-notebook hardcode |
| VSFC-Ekman | `pred == Neutral` AND `confidence < model_conf_limit` AND `top_emotion_slot > ont_threshold` | `model_conf_limit=0.90`, `ont_threshold=0.60` | In-notebook hardcode |
| UIT-VSMEC | `pred == Other` OR `confidence < model_conf_limit` AND `top_emotion_slot > ont_threshold` | `model_conf_limit=0.65`, `ont_threshold=0.40` | In-notebook hardcode |

---

## A5. Training Setup vs Paper Table 3

| Parameter | Code / Config | Paper Table 3 | Match? |
|---|---|---|---|
| Learning rate | `2.0e-5` (all configs) | 2e-5 | ✅ |
| Optimizer | AdamW (in notebooks: `torch.optim.AdamW`) | AdamW | ✅ |
| Weight decay | `weight_decay=0.01` (found in group 03 notebooks); **not in configs** | "AdamW default weight decay" — PyTorch default is **0.01** | Confirmed 0.01 in group 03 |
| Scheduler | None found in notebooks | "no scheduler" | ✅ |
| Batch size | 32 (all configs) | 32 | ✅ |
| Max epochs | 15 (all configs) | 15 | ✅ |
| Early stopping metric | Macro-F1 on dev set | Dev Macro-F1 | ✅ |
| Patience | **3** (VSFC, VSFC-Ekman); **4** (VSMEC group 02/vsmec; group 01; group 03) | Varies: 3 or 4 per dataset | ✅ |
| Dropout (BERT out) | `Dropout(p=0.3)` | 0.3 | ✅ |
| Dropout (adapter) | `Dropout(p=0.1)` | 0.1 | ✅ |
| Max length | 128 (all configs) | 128 | ✅ |
| Loss function | `nn.CrossEntropyLoss(weight=class_weights)` | class-weighted CE | ✅ |
| Seeds | 0, 42, 123, 1234, 2025 | 0, 42, 123, 1234, 2025 | ✅ |
| Deterministic cuDNN | `cudnn.deterministic=True`, `benchmark=False` | "deterministic cuDNN" | ✅ |

---

## A6. Ontology File

**File:** `ontology/ekman_appraisal_ontology.rdf` (9.76 MB)

| Element | Count |
|---|---|
| Total RDF triples | **91,005** |
| OWL Named Classes | 46 |
| Object Properties | 14 |
| Datatype Properties | 10 |
| Named Individuals | 11,733 |

- The repo contains **91,005 triples** — this is the **revision release for UIT-VSFC**.
- Per author clarification, only the 91,005-triple revision release will be archived in the repository; the earlier 90,420-triple release is no longer available.

---

## A7. Presence Check

| Component | Status | Location |
|---|---|---|
| SHAP / LIME notebooks | **Found** | `notebooks/05-shap-lime/` |
| Rule-based refinement layer | **Found (inline, not in src/)** | `NeuroSymbolicReasoner` class defined in notebook cells |
| Tier-1 trace export | **Partial** | `engine.getvector(text, debug=True)` returns `hit_trace`; diagnostic cells present |
| Lexicon construction script | **Not found as standalone script** | Built dynamically in `OntologyEngine._build_strict_lexicon()` |
| VSFC-Ekman labeling prompt/script | **Not found** | Documented in `data/vsfc-ekman/LABELING.md` with TODO |

---

## A8. Seeds in Config Table & Multi-Seed Run Order

Per author clarification:
- The five random seeds are **always run one at a time, in the exact order `0, 123, 1234, 2025, 42`**.
- The reported mean $\pm$ standard deviation in the paper were aggregated manually across those five separate executions.
- The single seed in each `configs/*.yaml` specifies the single representative seed used for the committed notebook execution in that group.

---

## A9. Extended Deep Audit

### A9.1 Evocation Confidence Audit
Querying `ontology/ekman_appraisal_ontology.rdf` with SPARQL / `rdflib`:
- **Total `Evocation` individuals:** **6,554**
  - **Evocations WITH `ekman:confidenceScore`:** **6,554** (100%)
  - **Evocations WITHOUT `ekman:confidenceScore`:** **0** (0%)
- **Distribution of `confidenceScore` grouped by `evidenceType`:**
  - `VnEmoLex_Background` (6,084 triples): min = **0.50**, max = **0.50** (all exact 0.50)
  - `Manual_Review` (263 triples): values $\in \{0.60, 0.70, 0.75, 0.85\}$
  - `Social_Lexicon_Import` (148 triples): values $\in \{0.70, 0.75, 0.80, 0.85, 0.90, 0.95\}$
  - `EducationDomainLexicon` (45 triples): values $\in \{0.70, 0.80\}$
  - `None` (14 triples): values $\in \{0.60, 0.70, 0.85, 0.90, 0.95\}$
- **`LexiconEntry` base intensity scores:**
  - Total `LexiconEntry` individuals: **10,925**
  - Entries WITH `baseIntensityScore`: **10,923**
  - Entries WITHOUT `baseIntensityScore`: **2** (only 2 entries use fallback `base=1.0`)
- **Is `default_conf` (0.85 / 0.90) ever actually used?**  
  **No.** `entry_score = float(r.ev or self.default_conf) * float(r.base or 1.0)`. Because 100% of `Evocation` individuals possess an explicit `confidenceScore`, `r.ev` is never falsy during lexicon creation. `default_conf` is unused dead fallback logic.

### A9.2 Slot Activation Audit
Executing `OntologyEngine.getvector` on the VSFC-Ekman test set (3,166 sentences) and `VSFCOntologyEngine.getvector` on the UIT-VSFC test set (3,166 sentences):

- **Share of sentences with $\ge 1$ non-zero dimension:**
  - VSFC-Ekman: **79.41%** (Paper claims 79.3% — confirmed ✅)
  - UIT-VSFC: **79.41%**
- **Mean number of non-zero dimensions per sentence:**
  - VSFC-Ekman: **2.00** (Paper claims ~1.99 — confirmed ✅)
  - UIT-VSFC: **2.01**
- **Per-dimension non-zero share breakdown (VSFC-Ekman & UIT-VSFC):**
  - Emotion slots (`ve_0..6`): Happiness 43.15%, Anger 29.66%, Disgust 8.72%, Sadness 8.37%, Surprise 4.55%, Fear 3.25%, Neutral 1.20%.
  - Polarity slots (`vp_0..2`): Positive 46.43%, Negative 41.28%, Neutral 0.00% (VSFC-Ekman) / 1.20% (UIT-VSFC).
  - Negation cue (`vn_0`): 13.36%.
  - **Appraisal dimensions 7–15 (`va_*`)**: **0.00%** (always zero).
  - **Intensity dimensions 16–18 (`vi_*`)**: **0.00%** (always zero).
  - **Unused negation dimension 23 (`vn[1]`)**: **0.00%** (always zero).
- **Reason for dead slots:** `_build_strict_lexicon` populates emotion mixtures (`mix_cache`) and direct emotion categories (`sim_cache`), but NO `evokesAppraisal` or `hasIntensity` triples are linked to the matched `LexiconEntry` nodes in `_infer_attributes()`. Thus, 13 out of 24 slots (dims 7–18 and dim 23) are permanently inactive dead slots in the current RDF file.

### A9.3 Per-Notebook Seed & Per-Class Analysis Audit

| Notebook | Dataset | Committed Seed | Per-Class / Confusion Matrix Source? |
|---|---|:---:|---|
| `01-phobert-fusion-ablation/vsfc.ipynb` | UIT-VSFC | **123** | **Yes** — matches paper claim for UIT-VSFC PhoBERT per-class analysis (seed 123). |
| `01-phobert-fusion-ablation/vsfc_ekman.ipynb` | VSFC-Ekman | **0** | Committed run uses seed 0. Paper claims seed 2025 was used for VSFC-Ekman Fig. 4. |
| `01-phobert-fusion-ablation/vsmec.ipynb` | UIT-VSMEC | **1234** | Committed run uses seed 1234. Paper claims seed 2025 was used for UIT-VSMEC Fig. 4. |
| `02-visobert-fusion-ablation/vsfc.ipynb` | UIT-VSFC | **2025** | Committed run uses seed 2025. |
| `02-visobert-fusion-ablation/vsfc_ekman.ipynb` | VSFC-Ekman | **2025** | Committed run uses seed 2025. |
| `02-visobert-fusion-ablation/vsmec.ipynb` | UIT-VSMEC | **42** | Committed run uses seed 42. |
| `03-lexicon-vs-ontology/vsfc.ipynb` | UIT-VSFC | **42** | Committed run uses seed 42. |
| `03-lexicon-vs-ontology/vsfc_ekman.ipynb` | VSFC-Ekman | **1234** | Committed run uses seed 1234. |
| `03-lexicon-vs-ontology/vsmec.ipynb` | UIT-VSMEC | **42** | Committed run uses seed 42. |
| `04-final-model-comparison/vsfc.ipynb` | UIT-VSFC | **1234** | Committed run uses seed 1234. |
| `04-final-model-comparison/vsfc_ekman.ipynb` | VSFC-Ekman | **42** | Committed run uses seed 42. |
| `04-final-model-comparison/vsmec.ipynb` | UIT-VSMEC | **2025** | Committed run uses seed 2025. |

- **Do committed outputs alone suffice to reconstruct paper mean $\pm$ std?**  
  **No.** Each committed notebook file contains execution outputs from only a **single seed** run. Reconstructing the reported mean $\pm$ sample standard deviation requires running each notebook 5 separate times (across seeds `0, 123, 1234, 2025, 42`) and aggregating test set F1 scores externally.

### A9.4 Early Stopping Patience Audit

| Group / Directory | Dataset Notebook | Early Stopping Patience |
|---|---|:---:|
| `01-phobert-fusion-ablation` | `vsfc.ipynb` | **3** |
| `01-phobert-fusion-ablation` | `vsfc_ekman.ipynb` | **3** (main) / **4** (deep) |
| `01-phobert-fusion-ablation` | `vsmec.ipynb` | **4** |
| `02-visobert-fusion-ablation` | `vsfc.ipynb` | **3** |
| `02-visobert-fusion-ablation` | `vsfc_ekman.ipynb` | **3** |
| `02-visobert-fusion-ablation` | `vsmec.ipynb` | **4** |
| `03-lexicon-vs-ontology` | `vsfc.ipynb` | **3** |
| `03-lexicon-vs-ontology` | `vsfc_ekman.ipynb` | **3** |
| `03-lexicon-vs-ontology` | `vsmec.ipynb` | **4** |
| `04-final-model-comparison` | `vsfc.ipynb` | **3** |
| `04-final-model-comparison` | `vsfc_ekman.ipynb` | **3** |
| `04-final-model-comparison` | `vsmec.ipynb` | **4** |

### A9.5 Gradient Clipping & Weight Decay Audit

| Group / Directory | Notebook | `clip_grad_norm_` used? | `weight_decay` value |
|---|---|:---:|:---:|
| `01-phobert-fusion-ablation` | `vsfc`, `vsfc_ekman`, `vsmec` | No | Default `0.0` |
| `02-visobert-fusion-ablation` | `vsfc`, `vsfc_ekman`, `vsmec` | No | Default `0.0` |
| `03-lexicon-vs-ontology` | `vsfc`, `vsfc_ekman`, `vsmec` | **Yes (`max_norm=1.0`)** | **`0.01`** (AdamW) |
| `04-final-model-comparison` | `vsfc`, `vsfc_ekman`, `vsmec` | No | Default `0.0` |

### A9.6 Audit of `notebooks/05-shap-lime/`

- **Model Variant Explained:** `PhoBERT_Fusion_V2(n_classes, fusion_type="gate", ontology_dim=None)` = **RawGate on PhoBERT** (matches requirement ✅).
- **Seeds Used:**
  - `vsfc-lime-roc.ipynb`: Seed **123**
  - `vsfcekman-roc-lime (2).ipynb`: Seed **0**
  - `vsmec-lime-roc (1).ipynb`: Seed **1234**
- **Datasets Explained:** UIT-VSFC, VSFC-Ekman, and UIT-VSMEC.
- **Explainer Tokenisation:**
  - LIME: `LimeTextExplainer(split_expression=r"\W+")` (splits on whitespace & punctuation → space-separated syllables).
  - SHAP: `shap.maskers.Text(r"\W+")` (splits on whitespace & punctuation → space-separated syllables).
  - Matches paper Section 5.7.2 description ("space-separated syllables") ✅.
- **Class Explained:** Top predicted class (`argmax` of model output probabilities).
- **Paper Example Sentences Verification:**
  - **VSFC-Ekman "nhiệt tình" sentence:** Appears in `vsfcekman-roc-lime (2).ipynb` Cell 11 & 12: `"giảng viên nhiệt tình trong công tác giảng dạy ."` (True: `Happiness`, Model Predict: `Happiness` 1.00) ✅.
  - **UIT-VSMEC "mấy ai được như vậy ??" comment:** Appears in `vsmec-lime-roc (1).ipynb` Cell 9 & 10: `"mấy ai được như vậy ??"` (True: `Other`, Model Predict: `Other` 0.94) ✅.
- **Correspondence with Paper Section 5.7.2 / Fig. 5:**  
  Both example sentences and their generated LIME/SHAP attributions directly match Figure 5 and Section 5.7.2 of the paper.

---

*End of audit report.*
