# IJMLC English confirmatory experiment

This directory contains the public-safe outputs for the English external-validation experiment described in the manuscript.

## Protocol

- Datasets: DailyDialog and MELD
- Backbone: RoBERTa-base
- Models: `baseline`, `nrcflat`, `rawgate_zero`, `rawgate_shuffled`, `rawgate`
- Seeds: 0, 42, 123, 1234, 2025
- Total completed runs: **50/50**
- Ontology: `ontology/ekman_appraisal_ontology_v2.rdf`

## Canonical aggregate results

### DailyDialog

| Model | Accuracy (mean ± SD) | Macro-F1 (mean ± SD) |
|---|---:|---:|
| `baseline` | 79.29 ± 1.32 | 47.92 ± 2.03 |
| `nrcflat` | 80.21 ± 1.98 | 49.39 ± 1.86 |
| `rawgate_zero` | 78.95 ± 2.52 | 48.74 ± 1.78 |
| `rawgate_shuffled` | 79.09 ± 2.93 | 48.57 ± 0.30 |
| `rawgate` | 77.75 ± 1.82 | 47.27 ± 2.13 |

### MELD

| Model | Accuracy (mean ± SD) | Macro-F1 (mean ± SD) |
|---|---:|---:|
| `baseline` | 57.84 ± 0.91 | 44.56 ± 0.67 |
| `nrcflat` | 56.54 ± 1.88 | 44.21 ± 0.79 |
| `rawgate_zero` | 56.84 ± 1.62 | 44.13 ± 1.24 |
| `rawgate_shuffled` | 57.33 ± 1.73 | 44.14 ± 0.54 |
| `rawgate` | 56.93 ± 1.33 | 43.41 ± 1.19 |

## Important aggregation note

The final resume session overwrote the dataset-level `summary.csv` and `per_seed_metrics.csv` files with the last requested seed. No run-level data were lost. The canonical five-seed summaries in `analysis/` were reconstructed from all 50 individual `metrics.json` files and verified against the per-run outputs.

The repository version of `scripts/run_experiment.py` has been patched so future resumed executions rebuild aggregate summaries from **all complete run directories on disk**.

## Licensing / redistribution

The private NRC Emotion Lexicon file is **not included**. Reproduction requires users to obtain the NRC resource separately under its applicable terms. The public result archive excludes cached ontology/NRC feature arrays and contains only model outputs, diagnostics, configuration snapshots, and frozen protocol files.
