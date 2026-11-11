# English cross-lingual extension and semantic controls

This extension evaluates whether the transformer–ontology fusion mechanism transfers from Vietnamese to English while keeping the semantic interface and evaluation logic fixed.

## Experimental protocol

- Datasets: DailyDialog, MELD
- Backbone: `roberta-base`
- Models: `baseline`, `nrcflat`, `rawgate_zero`, `rawgate_shuffled`, `rawgate`
- Seeds: `0 42 123 1234 2025`
- Max epochs: 15
- Selection: best development Macro-F1 with early stopping
- Ontology: `ontology/ekman_appraisal_ontology_v2.rdf`
- Interface: 24 dimensions; 18 can be active in the present English grounding

## NRC resource

The NRC Emotion Lexicon is **not redistributed**. Obtain the licensed source separately, then build the local English lexical layer:

```bash
python scripts/build_english_lexicon.py \
  --nrc /path/to/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt \
  --ontology ontology/ekman_appraisal_ontology_v2.rdf
```

This creates the structured ontology-v2 lexical layer and the flat NRC seven-emotion vector used by the confirmatory study.

## Run DailyDialog

Prepare canonical `train.csv`, `dev.csv`, and `test.csv` under your data directory, then run:

```bash
python scripts/run_experiment.py \
  --config configs/dailydialog_roberta.yaml \
  --models baseline nrcflat rawgate_zero rawgate_shuffled rawgate \
  --seeds 0 42 123 1234 2025 \
  --data-dir /path/to/dailydialog \
  --output-root results/english/dailydialog_confirmatory_5seed
```

## Run MELD

```bash
python scripts/run_experiment.py \
  --config configs/meld_roberta.yaml \
  --models baseline nrcflat rawgate_zero rawgate_shuffled rawgate \
  --seeds 0 42 123 1234 2025 \
  --data-dir /path/to/meld \
  --output-root results/english/meld_confirmatory_5seed
```

The runner is resume-safe. A model×seed is skipped only when `metrics.json`, `predictions.csv`, `history.csv`, and `coverage_stratified_metrics.csv` are present and non-empty. The repository runner also fixes the resume aggregation issue by rebuilding dataset-level summaries from **all complete run directories on disk**.

## Kaggle

Use `notebooks/06-english-crosslingual/IJMLC_English_5seed_confirmatory_resume_v10_2.ipynb` for the full resume-safe five-seed workflow. The `...final_completion_v11.ipynb` notebook is retained as provenance for the final seed-2025 completion session and is not the preferred clean-start runner.

## Results

Canonical corrected five-seed tables are in:

`results/english/confirmatory_5seed/analysis/`

A compact public-safe archive containing all per-run predictions, histories, diagnostics, configuration snapshots, and frozen protocol files is available as:

`results/english/confirmatory_5seed/ijmlc_english_confirmatory_5seed_results_public.zip`

Feature-cache arrays and the private NRC source are intentionally excluded from the public package.
