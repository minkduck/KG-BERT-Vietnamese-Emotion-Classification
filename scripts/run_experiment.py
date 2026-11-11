#!/usr/bin/env python3
"""Run ontology-v2 English experiments for the IJMLC extension.

v2 adds four forms of evidence needed to isolate and explain ontology effects:

1. ``rawgate_zero``: same RawGate architecture with an all-zero ontology vector.
2. ``rawgate_shuffled``: same RawGate architecture with split-wise, seeded sample
   permutation of the ontology vectors, destroying text↔ontology semantic alignment.
3. Gate diagnostics saved per test sample.
4. Coverage-stratified evaluation on ontology-active/inactive and
   appraisal-active/inactive subsets.

The real ontology availability masks are always computed from the unperturbed
ontology-v2 vectors, even for the zero/shuffled controls.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy.stats import t as student_t
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loading import load_dataset_by_name
from src.language_adapters import EnglishAdapter
from src.models.transformer_fusion_models import TransformerOntologyFusion
from src.ontology_engine import OntologyEngine
from src.utils import seed_everything


MODEL_SPECS = {
    "baseline": {
        "fusion_mode": "none",
        "feature_source": "structured",
        "control_mode": "real",
    },
    "nrcflat": {
        "fusion_mode": "flatconcat",
        "feature_source": "flat",
        "control_mode": "real",
    },
    "rawconcat": {
        "fusion_mode": "rawconcat",
        "feature_source": "structured",
        "control_mode": "real",
    },
    "rawgate": {
        "fusion_mode": "rawgate",
        "feature_source": "structured",
        "control_mode": "real",
    },
    "rawgate_zero": {
        "fusion_mode": "rawgate",
        "feature_source": "structured",
        "control_mode": "zero",
    },
    "rawgate_shuffled": {
        "fusion_mode": "rawgate",
        "feature_source": "structured",
        "control_mode": "shuffled",
    },
    "denseconcat": {
        "fusion_mode": "denseconcat",
        "feature_source": "structured",
        "control_mode": "real",
    },
    "densegate": {
        "fusion_mode": "densegate",
        "feature_source": "structured",
        "control_mode": "real",
    },
}

DIMENSION_NAMES = [
    "emotion_Anger",
    "emotion_Disgust",
    "emotion_Fear",
    "emotion_Happiness",
    "emotion_Neutral",
    "emotion_Sadness",
    "emotion_Surprise",
    "appraisal_GoalObstruction",
    "appraisal_Pleasantness",
    "appraisal_Unpleasantness",
    "appraisal_Dangerousness",
    "appraisal_Loss",
    "appraisal_Suddenness",
    "appraisal_LowPredictability",
    "appraisal_Unpredictability",
    "appraisal_Unfairness",
    "intensity_High",
    "intensity_Medium",
    "intensity_Low",
    "polarity_Positive",
    "polarity_Negative",
    "polarity_Neutral",
    "negation_active",
    "context_reserved",
]

FEATURE_GROUPS = {
    "emotion": (0, 7),
    "appraisal": (7, 16),
    "intensity": (16, 19),
    "polarity": (19, 22),
    "negation_context": (22, 24),
}

SPLIT_SHUFFLE_OFFSETS = {
    "train": 1009,
    "dev": 2017,
    "test": 3011,
}


def resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correctness = (predictions == y_true).astype(np.float32)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        ece += float(mask.mean()) * abs(
            float(correctness[mask].mean())
            - float(confidences[mask].mean())
        )
    return float(ece)


class FlatLexiconVectorizer:
    """Transparent 7-d NRC-flat comparison with no ontology structure."""

    def __init__(self, json_path: Path):
        with json_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        self.lookup = {
            str(k).lower(): np.asarray(v, dtype=np.float32)
            for k, v in raw.items()
        }
        self.adapter = EnglishAdapter()
        invalid = [
            k for k, v in self.lookup.items() if v.shape != (7,)
        ]
        if invalid:
            raise ValueError(
                "Flat lexicon vectors must be length 7; invalid example: "
                f"{invalid[0]}"
            )

    def getvector(self, text: str) -> np.ndarray:
        vector = np.zeros(7, dtype=np.float32)
        for token in self.adapter.tokenize(text):
            if token in self.lookup:
                vector += self.lookup[token]
        return np.tanh(vector).astype(np.float32)


class EmotionDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer,
        label_to_id: Dict[str, int],
        structured_features: np.ndarray,
        flat_features: np.ndarray,
        max_len: int,
    ):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.structured = structured_features.astype(np.float32)
        self.flat = flat_features.astype(np.float32)
        self.max_len = int(max_len)
        assert len(self.frame) == len(self.structured) == len(self.flat)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        encoding = self.tokenizer(
            str(row["Sentence"]),
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "sample_id": idx,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(
                self.label_to_id[str(row["Emotion"])],
                dtype=torch.long,
            ),
            "structured": torch.tensor(
                self.structured[idx], dtype=torch.float32
            ),
            "flat": torch.tensor(
                self.flat[idx], dtype=torch.float32
            ),
        }


def compute_feature_matrix(
    frame: pd.DataFrame,
    vectorizer,
    desc: str,
) -> np.ndarray:
    vectors = []
    for text in tqdm(frame["Sentence"].astype(str).tolist(), desc=desc):
        vectors.append(vectorizer.getvector(text))
    return np.stack(vectors).astype(np.float32)


def load_or_build_features(
    frame: pd.DataFrame,
    split: str,
    cache_dir: Path,
    structured_engine,
    flat_engine,
) -> Tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    structured_path = cache_dir / f"{split}_structured24.npy"
    flat_path = cache_dir / f"{split}_nrcflat7.npy"

    if structured_path.exists():
        structured = np.load(structured_path)
    else:
        structured = compute_feature_matrix(
            frame,
            structured_engine,
            f"{split}: ontology-v2 structured24",
        )
        np.save(structured_path, structured)

    if flat_path.exists():
        flat = np.load(flat_path)
    else:
        flat = compute_feature_matrix(
            frame,
            flat_engine,
            f"{split}: NRC-flat7",
        )
        np.save(flat_path, flat)

    if len(structured) != len(frame) or len(flat) != len(frame):
        raise ValueError(
            "Feature cache length differs from dataset. "
            "Delete the cache directory and rebuild."
        )
    if structured.shape[1] != 24:
        raise ValueError(
            f"Structured feature width must be 24, got {structured.shape}"
        )
    return structured, flat


def transform_structured_features(
    structured: np.ndarray,
    control_mode: str,
    seed: int,
    split: str,
) -> np.ndarray:
    """Return the structured vectors seen by a given experimental condition."""
    if control_mode == "real":
        return structured.copy()
    if control_mode == "zero":
        return np.zeros_like(structured, dtype=np.float32)
    if control_mode == "shuffled":
        offset = SPLIT_SHUFFLE_OFFSETS[split]
        rng = np.random.default_rng(int(seed) * 10007 + offset)
        permutation = rng.permutation(len(structured))
        return structured[permutation].copy()
    raise ValueError(f"Unknown control_mode={control_mode}")


def save_feature_audit(
    real_features: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_root: Path,
):
    dimension_rows = []
    group_rows = []

    for split, (structured, _) in real_features.items():
        absolute = np.abs(structured)
        for idx, name in enumerate(DIMENSION_NAMES):
            col = absolute[:, idx]
            dimension_rows.append(
                {
                    "split": split,
                    "dimension_index": idx,
                    "dimension_name": name,
                    "activation_rate": float(np.mean(col > 1e-8)),
                    "mean_abs_value": float(np.mean(col)),
                    "max_abs_value": float(np.max(col)),
                }
            )

        for group_name, (start, end) in FEATURE_GROUPS.items():
            block = absolute[:, start:end]
            active_per_sample = np.sum(block > 1e-8, axis=1)
            group_rows.append(
                {
                    "split": split,
                    "feature_group": group_name,
                    "coverage": float(np.mean(active_per_sample > 0)),
                    "mean_active_dimensions": float(
                        np.mean(active_per_sample)
                    ),
                    "mean_abs_group_value": float(np.mean(block)),
                }
            )

    pd.DataFrame(dimension_rows).to_csv(
        output_root / "feature_dimension_audit.csv",
        index=False,
    )
    pd.DataFrame(group_rows).to_csv(
        output_root / "feature_group_audit.csv",
        index=False,
    )

    appraisal_test = [
        row
        for row in dimension_rows
        if row["split"] == "test"
        and str(row["dimension_name"]).startswith("appraisal_")
    ]
    active_appraisal_dimensions = sum(
        row["activation_rate"] > 0 for row in appraisal_test
    )
    print(
        "Feature audit: active appraisal dimensions on test = "
        f"{active_appraisal_dimensions}/9"
    )


def make_class_weights(
    frame: pd.DataFrame,
    label_to_id: Dict[str, int],
) -> torch.Tensor:
    counts = np.zeros(len(label_to_id), dtype=np.float64)
    for label, count in frame["Emotion"].value_counts().items():
        counts[label_to_id[label]] = count
    if np.any(counts == 0):
        missing = [
            label for label, idx in label_to_id.items()
            if counts[idx] == 0
        ]
        raise ValueError(
            f"Training split contains no examples for classes: {missing}"
        )
    total = counts.sum()
    weights = total / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _safe_nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(values[valid]))


def evaluate(
    model,
    loader,
    device,
    feature_source: str,
    n_classes: int,
    collect_aux: bool = False,
):
    model.eval()
    y_true, y_pred, all_probs, sample_ids = [], [], [], []
    diagnostics = {
        "gate_mean": [],
        "gate_max": [],
        "active_gate_mean": [],
        "ontology_vector_norm": [],
        "gated_ontology_norm": [],
        "model_feature_active": [],
    }

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            features = batch[feature_source].to(device)

            if collect_aux:
                logits, aux = model(
                    input_ids,
                    attention_mask,
                    features,
                    return_aux=True,
                )
            else:
                logits = model(input_ids, attention_mask, features)
                aux = None

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())
            sample_ids.extend(batch["sample_id"].numpy().tolist())

            if collect_aux:
                feature_np = features.detach().cpu().numpy()
                feature_active = np.any(
                    np.abs(feature_np) > 1e-8, axis=1
                )
                diagnostics["model_feature_active"].extend(
                    feature_active.astype(int).tolist()
                )

                if model.fusion_mode == "none":
                    batch_size = len(feature_np)
                    for key in [
                        "gate_mean",
                        "gate_max",
                        "active_gate_mean",
                        "ontology_vector_norm",
                        "gated_ontology_norm",
                    ]:
                        diagnostics[key].extend(
                            [float("nan")] * batch_size
                        )
                    continue

                input_norm = np.linalg.norm(feature_np, axis=1)
                diagnostics["ontology_vector_norm"].extend(
                    input_norm.tolist()
                )

                gate = aux.get("gate") if aux else None
                post = (
                    aux.get("ontology_representation")
                    if aux else None
                )

                if gate is None:
                    batch_size = len(feature_np)
                    diagnostics["gate_mean"].extend(
                        [float("nan")] * batch_size
                    )
                    diagnostics["gate_max"].extend(
                        [float("nan")] * batch_size
                    )
                    diagnostics["active_gate_mean"].extend(
                        [float("nan")] * batch_size
                    )
                    diagnostics["gated_ontology_norm"].extend(
                        [float("nan")] * batch_size
                    )
                else:
                    gate_np = gate.detach().cpu().numpy()
                    diagnostics["gate_mean"].extend(
                        gate_np.mean(axis=1).tolist()
                    )
                    diagnostics["gate_max"].extend(
                        gate_np.max(axis=1).tolist()
                    )

                    if gate_np.shape[1] == feature_np.shape[1]:
                        mask = np.abs(feature_np) > 1e-8
                        active_means = []
                        for row_gate, row_mask in zip(gate_np, mask):
                            if np.any(row_mask):
                                active_means.append(
                                    float(row_gate[row_mask].mean())
                                )
                            else:
                                active_means.append(float("nan"))
                        diagnostics["active_gate_mean"].extend(
                            active_means
                        )
                    else:
                        diagnostics["active_gate_mean"].extend(
                            [float("nan")] * len(feature_np)
                        )

                    post_np = post.detach().cpu().numpy()
                    diagnostics["gated_ontology_norm"].extend(
                        np.linalg.norm(post_np, axis=1).tolist()
                    )

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    probs = np.concatenate(all_probs, axis=0)

    result = {
        "sample_id": np.asarray(sample_ids, dtype=int),
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=list(range(n_classes)),
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=list(range(n_classes)),
            average="weighted",
            zero_division=0,
        ),
        "ece": expected_calibration_error(y_true, probs),
        "confusion": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(n_classes)),
        ),
    }
    if collect_aux:
        result["diagnostics"] = {
            key: np.asarray(values, dtype=float)
            for key, values in diagnostics.items()
        }
    return result


def _amp_context(device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def train_one(
    model,
    train_loader,
    dev_loader,
    device,
    feature_source: str,
    class_weights: torch.Tensor,
    cfg: dict,
    checkpoint_path: Path,
):
    training = cfg["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )
    epochs = int(training["epochs"])
    patience = int(training["patience"])
    grad_clip = float(training.get("gradient_clip", 1.0))
    use_amp = bool(training.get("fp16", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_f1 = -1.0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(
            train_loader,
            desc=f"epoch {epoch}/{epochs}",
            leave=False,
        ):
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            features = batch[feature_source].to(device)

            with _amp_context(device, use_amp):
                logits = model(
                    input_ids,
                    attention_mask,
                    features,
                )
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))

        dev = evaluate(
            model,
            dev_loader,
            device,
            feature_source,
            model.classifier.out_features,
            collect_aux=False,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "dev_accuracy": dev["accuracy"],
            "dev_macro_f1": dev["macro_f1"],
            "dev_weighted_f1": dev["weighted_f1"],
            "dev_ece": dev["ece"],
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} "
            f"loss={row['train_loss']:.4f} "
            f"dev_acc={row['dev_accuracy']:.4f} "
            f"dev_macro_f1={row['dev_macro_f1']:.4f}"
        )

        if dev["macro_f1"] > best_f1 + 1e-8:
            best_f1 = dev["macro_f1"]
            stale = 0
            checkpoint_path.parent.mkdir(
                parents=True, exist_ok=True
            )
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                print(
                    f"Early stopping after epoch {epoch}; "
                    f"best dev Macro-F1={best_f1:.4f}"
                )
                break

    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state)
    return pd.DataFrame(history), best_f1


def subset_metrics(
    test_result: dict,
    mask: np.ndarray,
    n_classes: int,
):
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.sum())
    if n == 0:
        return {
            "n_samples": 0,
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "weighted_f1": float("nan"),
            "ece": float("nan"),
        }

    y_true = test_result["y_true"][mask]
    y_pred = test_result["y_pred"][mask]
    probs = test_result["probs"][mask]
    return {
        "n_samples": n,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(range(n_classes)),
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(range(n_classes)),
                average="weighted",
                zero_division=0,
            )
        ),
        "ece": float(expected_calibration_error(y_true, probs)),
    }


def save_run_outputs(
    run_dir: Path,
    test_result: dict,
    class_names: List[str],
    test_real_structured: np.ndarray,
    test_model_structured: np.ndarray,
    metadata: dict,
):
    run_dir.mkdir(parents=True, exist_ok=True)

    real_abs = np.abs(test_real_structured)
    ontology_active = np.any(real_abs > 1e-8, axis=1)
    ontology_active_dims = np.sum(real_abs > 1e-8, axis=1)
    appraisal_active = np.any(real_abs[:, 7:16] > 1e-8, axis=1)
    appraisal_active_dims = np.sum(
        real_abs[:, 7:16] > 1e-8, axis=1
    )
    polarity_active = np.any(real_abs[:, 19:22] > 1e-8, axis=1)
    negation_active = np.any(real_abs[:, 22:24] > 1e-8, axis=1)

    diagnostics = test_result.get("diagnostics", {})

    metrics = {
        "accuracy": float(test_result["accuracy"]),
        "macro_f1": float(test_result["macro_f1"]),
        "weighted_f1": float(test_result["weighted_f1"]),
        "ece": float(test_result["ece"]),
        "ontology_coverage": float(ontology_active.mean()),
        "mean_active_ontology_dims": float(
            ontology_active_dims.mean()
        ),
        "appraisal_coverage": float(appraisal_active.mean()),
        "mean_active_appraisal_dims": float(
            appraisal_active_dims.mean()
        ),
        "polarity_coverage": float(polarity_active.mean()),
        "negation_coverage": float(negation_active.mean()),
        "gate_mean_mean": _safe_nanmean(
            diagnostics.get("gate_mean", np.array([np.nan]))
        ),
        "gate_max_mean": _safe_nanmean(
            diagnostics.get("gate_max", np.array([np.nan]))
        ),
        "active_gate_mean_mean": _safe_nanmean(
            diagnostics.get(
                "active_gate_mean", np.array([np.nan])
            )
        ),
        "ontology_vector_norm_mean": _safe_nanmean(
            diagnostics.get(
                "ontology_vector_norm", np.array([np.nan])
            )
        ),
        "gated_ontology_norm_mean": _safe_nanmean(
            diagnostics.get(
                "gated_ontology_norm", np.array([np.nan])
            )
        ),
        **metadata,
    }

    probs = test_result["probs"]
    prediction_data = {
        "sample_id": test_result["sample_id"],
        "y_true": test_result["y_true"],
        "y_pred": test_result["y_pred"],
        "true_label": [
            class_names[i] for i in test_result["y_true"]
        ],
        "pred_label": [
            class_names[i] for i in test_result["y_pred"]
        ],
        "confidence": probs.max(axis=1),
        "ontology_active": ontology_active.astype(int),
        "ontology_active_dims": ontology_active_dims,
        "appraisal_active": appraisal_active.astype(int),
        "appraisal_active_dims": appraisal_active_dims,
        "polarity_active": polarity_active.astype(int),
        "negation_active": negation_active.astype(int),
    }

    for key in [
        "model_feature_active",
        "gate_mean",
        "gate_max",
        "active_gate_mean",
        "ontology_vector_norm",
        "gated_ontology_norm",
    ]:
        if key in diagnostics:
            prediction_data[key] = diagnostics[key]

    prediction_df = pd.DataFrame(prediction_data)
    for idx, name in enumerate(class_names):
        prediction_df[f"p_{name}"] = probs[:, idx]
    prediction_df.to_csv(
        run_dir / "predictions.csv", index=False
    )

    report = classification_report(
        test_result["y_true"],
        test_result["y_pred"],
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        [
            {
                "class": name,
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in class_names
        ]
    )
    per_class.to_csv(
        run_dir / "per_class_f1.csv", index=False
    )
    pd.DataFrame(
        test_result["confusion"],
        index=class_names,
        columns=class_names,
    ).to_csv(run_dir / "confusion_matrix.csv")

    strata = {
        "ontology_active": ontology_active,
        "ontology_inactive": ~ontology_active,
        "appraisal_active": appraisal_active,
        "appraisal_inactive": ~appraisal_active,
    }
    stratum_rows = []
    for stratum_name, mask in strata.items():
        values = subset_metrics(
            test_result,
            mask,
            len(class_names),
        )
        values.update(
            {
                "stratum": stratum_name,
                "model": metadata["model"],
                "seed": metadata["seed"],
            }
        )
        stratum_rows.append(values)
        (run_dir / f"metrics_{stratum_name}.json").write_text(
            json.dumps(values, indent=2),
            encoding="utf-8",
        )

    pd.DataFrame(stratum_rows).to_csv(
        run_dir / "coverage_stratified_metrics.csv",
        index=False,
    )

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    if any(key in diagnostics for key in ["gate_mean", "gate_max"]):
        gate_cols = [
            c
            for c in [
                "sample_id",
                "ontology_active",
                "appraisal_active",
                "model_feature_active",
                "gate_mean",
                "gate_max",
                "active_gate_mean",
                "ontology_vector_norm",
                "gated_ontology_norm",
            ]
            if c in prediction_df.columns
        ]
        prediction_df[gate_cols].to_csv(
            run_dir / "gate_diagnostics.csv",
            index=False,
        )

    # Save actual control features only as compact norms/flags in predictions;
    # full feature arrays already exist in the cache and are not duplicated.
    return metrics, stratum_rows


def _collect_all_complete_run_records(output_dir: Path):
    """Rebuild aggregate inputs from every complete run already on disk.

    This makes resumed executions safe even when the current invocation only
    requests one seed. A run is considered complete only when the same four
    artifacts used by the resume logic are present and non-empty.
    """
    disk_records = []
    disk_strata = []

    for metrics_path in sorted(output_dir.glob("*/seed_*/metrics.json")):
        run_dir = metrics_path.parent
        required_outputs = [
            run_dir / "metrics.json",
            run_dir / "predictions.csv",
            run_dir / "history.csv",
            run_dir / "coverage_stratified_metrics.csv",
        ]
        if not all(path.exists() and path.stat().st_size > 0 for path in required_outputs):
            continue

        with metrics_path.open("r", encoding="utf-8") as handle:
            disk_records.append(json.load(handle))

        strata_path = run_dir / "coverage_stratified_metrics.csv"
        disk_strata.extend(pd.read_csv(strata_path).to_dict("records"))

    return disk_records, disk_strata


def aggregate_metrics(
    records: List[dict],
    stratum_records: List[dict],
    output_dir: Path,
):
    # IMPORTANT: on resume, the current command may contain only a subset of
    # seeds. Re-scan all complete run directories so summary.csv and
    # per_seed_metrics.csv always describe the full accumulated experiment.
    disk_records, disk_strata = _collect_all_complete_run_records(output_dir)
    if disk_records:
        records = disk_records
        stratum_records = disk_strata

    frame = pd.DataFrame(records)
    rows = []
    metric_cols = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "ece",
        "ontology_coverage",
        "mean_active_ontology_dims",
        "appraisal_coverage",
        "mean_active_appraisal_dims",
        "polarity_coverage",
        "negation_coverage",
        "gate_mean_mean",
        "gate_max_mean",
        "active_gate_mean_mean",
        "ontology_vector_norm_mean",
        "gated_ontology_norm_mean",
    ]

    for model, group in frame.groupby("model"):
        row = {"model": model, "n_seeds": len(group)}
        for column in metric_cols:
            values = group[column].astype(float).to_numpy()
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                mean = std = lo = hi = float("nan")
            else:
                mean = float(finite.mean())
                std = (
                    float(finite.std(ddof=1))
                    if len(finite) > 1
                    else float("nan")
                )
                if len(finite) > 1:
                    critical = float(
                        student_t.ppf(
                            0.975, df=len(finite) - 1
                        )
                    )
                    half = critical * std / math.sqrt(len(finite))
                    lo, hi = mean - half, mean + half
                else:
                    lo = hi = float("nan")
            row[f"{column}_mean"] = mean
            row[f"{column}_std"] = std
            row[f"{column}_ci95_low"] = lo
            row[f"{column}_ci95_high"] = hi
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("model")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    strata = pd.DataFrame(stratum_records)
    strata.to_csv(
        output_dir / "coverage_stratified_per_seed.csv",
        index=False,
    )

    strata_summary_rows = []
    if not strata.empty:
        for (model, stratum), group in strata.groupby(
            ["model", "stratum"]
        ):
            row = {
                "model": model,
                "stratum": stratum,
                "n_seeds": int(group["seed"].nunique()),
                "n_samples_mean": float(group["n_samples"].mean()),
            }
            for column in [
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "ece",
            ]:
                vals = group[column].astype(float).to_numpy()
                row[f"{column}_mean"] = float(
                    np.nanmean(vals)
                )
                row[f"{column}_std"] = (
                    float(np.nanstd(vals, ddof=1))
                    if len(vals) > 1
                    else float("nan")
                )
            strata_summary_rows.append(row)

    pd.DataFrame(strata_summary_rows).to_csv(
        output_dir / "coverage_stratified_summary.csv",
        index=False,
    )

    print("\n=== Aggregated results ===")
    print(
        summary[
            [
                "model",
                "n_seeds",
                "accuracy_mean",
                "macro_f1_mean",
                "appraisal_coverage_mean",
            ]
        ].to_string(index=False)
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=sorted(MODEL_SPECS),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override config dataset.data_dir",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Override config experiment.output_dir",
    )
    parser.add_argument(
        "--backbone",
        default=None,
        help="Override config model.backbone",
    )
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_yaml(config_path)

    dataset_name = cfg["dataset"]["name"]
    data_dir = resolve_path(
        args.data_dir or cfg["dataset"]["data_dir"]
    )
    class_names = list(cfg["dataset"]["classes"])
    label_to_id = {
        name: idx for idx, name in enumerate(class_names)
    }
    n_classes = len(class_names)

    backbone = args.backbone or cfg["model"]["backbone"]
    models = args.models or list(cfg["experiment"]["models"])
    seeds = args.seeds or list(cfg["experiment"]["seeds"])
    output_root = resolve_path(
        args.output_root or cfg["experiment"]["output_dir"]
    )
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset_name}")
    print(f"Data dir: {data_dir}")
    print(f"Backbone: {backbone}")
    print(f"Models: {models}")
    print(f"Seeds: {seeds}")
    print(f"Output: {output_root}")

    train_df, dev_df, test_df = load_dataset_by_name(
        dataset_name, str(data_dir)
    )
    for split_name, frame in [
        ("train", train_df),
        ("dev", dev_df),
        ("test", test_df),
    ]:
        unknown = sorted(
            set(frame["Emotion"]) - set(class_names)
        )
        if unknown:
            raise ValueError(
                f"{split_name} contains labels missing from config: "
                f"{unknown}"
            )
        print(
            f"{split_name}: n={len(frame):,}; "
            f"labels={frame['Emotion'].value_counts().to_dict()}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        backbone, use_fast=True
    )

    ontology_cfg = cfg["ontology"]
    rdf_path = resolve_path(ontology_cfg["rdf_path"])
    structured_path = resolve_path(
        ontology_cfg["external_lexicon_path"]
    )
    flat_path = resolve_path(ontology_cfg["flat_lexicon_path"])

    if not rdf_path.exists():
        raise FileNotFoundError(f"Ontology v2 missing: {rdf_path}")
    if not structured_path.exists():
        raise FileNotFoundError(
            f"Structured English lexicon missing: {structured_path}\n"
            "Run scripts/build_english_lexicon.py first."
        )
    if not flat_path.exists():
        raise FileNotFoundError(
            f"Flat NRC lexicon missing: {flat_path}\n"
            "Run scripts/build_english_lexicon.py first."
        )

    structured_engine = OntologyEngine(
        rdf_path=str(rdf_path),
        default_conf=float(
            ontology_cfg.get("default_conf", 0.5)
        ),
        manual_boost=None,
        negation_attenuation=float(
            ontology_cfg.get("negation_attenuation", 0.5)
        ),
        alpha_similarity=float(
            ontology_cfg.get("alpha_similarity", 0.2)
        ),
        verbose=True,
        language_adapter=EnglishAdapter(),
        external_lexicon_path=str(structured_path),
    )
    flat_engine = FlatLexiconVectorizer(flat_path)

    cache_dir = output_root / "feature_cache"
    if args.rebuild_features and cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)

    real_features = {}
    for split, frame in [
        ("train", train_df),
        ("dev", dev_df),
        ("test", test_df),
    ]:
        real_features[split] = load_or_build_features(
            frame,
            split,
            cache_dir,
            structured_engine,
            flat_engine,
        )

    save_feature_audit(real_features, output_root)

    batch_size = int(cfg["training"]["batch_size"])
    max_len = int(cfg["training"]["max_len"])
    num_workers = int(
        cfg["training"].get("num_workers", 2)
    )
    class_weights = make_class_weights(
        train_df, label_to_id
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    records = []
    stratum_records = []

    for model_name in models:
        spec = MODEL_SPECS[model_name]
        fusion_mode = spec["fusion_mode"]
        feature_source = spec["feature_source"]
        control_mode = spec["control_mode"]
        feature_dim = 7 if feature_source == "flat" else 24

        for seed in seeds:
            run_dir = output_root / model_name / f"seed_{seed}"
            checkpoint = (
                output_root
                / "_tmp_checkpoints"
                / f"{model_name}_seed_{seed}.pt"
            )

            # Resume-safe behavior: skip only runs with a complete output set.
            required_outputs = [
                run_dir / "metrics.json",
                run_dir / "predictions.csv",
                run_dir / "history.csv",
                run_dir / "coverage_stratified_metrics.csv",
            ]
            if all(path.exists() and path.stat().st_size > 0 for path in required_outputs):
                print("\n" + "=" * 80)
                print(f"SKIP completed model={model_name} seed={seed}")
                with (run_dir / "metrics.json").open("r", encoding="utf-8") as handle:
                    existing_metrics = json.load(handle)
                records.append(existing_metrics)
                strata_path = run_dir / "coverage_stratified_metrics.csv"
                existing_strata = pd.read_csv(strata_path).to_dict("records")
                stratum_records.extend(existing_strata)
                continue

            print("\n" + "=" * 80)
            print(
                f"RUN model={model_name} seed={seed} "
                f"control={control_mode}"
            )
            seed_everything(seed)

            transformed = {}
            for split in ["train", "dev", "test"]:
                real_structured, flat = real_features[split]
                transformed[split] = (
                    transform_structured_features(
                        real_structured,
                        control_mode,
                        seed,
                        split,
                    ),
                    flat,
                )

            train_ds = EmotionDataset(
                train_df,
                tokenizer,
                label_to_id,
                transformed["train"][0],
                transformed["train"][1],
                max_len,
            )
            dev_ds = EmotionDataset(
                dev_df,
                tokenizer,
                label_to_id,
                transformed["dev"][0],
                transformed["dev"][1],
                max_len,
            )
            test_ds = EmotionDataset(
                test_df,
                tokenizer,
                label_to_id,
                transformed["test"][0],
                transformed["test"][1],
                max_len,
            )

            generator = torch.Generator()
            generator.manual_seed(seed)
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
                generator=generator,
            )
            dev_loader = DataLoader(
                dev_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=device.type == "cuda",
            )

            model = TransformerOntologyFusion(
                model_name=backbone,
                n_classes=n_classes,
                fusion_mode=fusion_mode,
                ont_input_dim=feature_dim,
                dense_dim=int(
                    cfg["model"].get("dense_dim", 64)
                ),
                text_dropout=float(
                    cfg["model"].get("text_dropout", 0.3)
                ),
                ontology_dropout=float(
                    cfg["model"].get(
                        "ontology_dropout", 0.1
                    )
                ),
            ).to(device)

            start_time = time.time()
            history, best_dev_f1 = train_one(
                model,
                train_loader,
                dev_loader,
                device,
                feature_source,
                class_weights,
                cfg,
                checkpoint,
            )
            test_result = evaluate(
                model,
                test_loader,
                device,
                feature_source,
                n_classes,
                collect_aux=True,
            )
            elapsed = time.time() - start_time

            metadata = {
                "dataset": dataset_name,
                "backbone": backbone,
                "model": model_name,
                "fusion_mode": fusion_mode,
                "feature_source": feature_source,
                "control_mode": control_mode,
                "seed": int(seed),
                "best_dev_macro_f1": float(best_dev_f1),
                "runtime_seconds": float(elapsed),
                "ontology_rdf": rdf_path.name,
                "structured_lexicon": structured_path.name,
            }

            metrics, run_strata = save_run_outputs(
                run_dir=run_dir,
                test_result=test_result,
                class_names=class_names,
                test_real_structured=real_features["test"][0],
                test_model_structured=transformed["test"][0],
                metadata=metadata,
            )
            history.to_csv(
                run_dir / "history.csv", index=False
            )
            (run_dir / "config_snapshot.yaml").write_text(
                yaml.safe_dump(
                    cfg,
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
            records.append(metrics)
            stratum_records.extend(run_strata)

            print(
                f"TEST acc={metrics['accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f} "
                f"ECE={metrics['ece']:.4f} "
                f"onto_cov={metrics['ontology_coverage']:.4f} "
                f"app_cov={metrics['appraisal_coverage']:.4f}"
            )

            if checkpoint.exists():
                checkpoint.unlink()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    aggregate_metrics(records, stratum_records, output_root)
    print(f"\nCompleted. Results are under: {output_root}")


if __name__ == "__main__":
    main()
