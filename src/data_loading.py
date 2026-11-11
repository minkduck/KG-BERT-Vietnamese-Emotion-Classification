"""Unified dataset loaders for the original Vietnamese experiments and IJMLC English extension."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


# -----------------------------------------------------------------------------
# Existing Vietnamese loaders (kept backward compatible)
# -----------------------------------------------------------------------------

def load_vsfc(data_dir="data/vsfc"):
    """Load UIT-VSFC as (train, dev, test), columns: Sentence, Sentiment."""
    zip_path = os.path.join(data_dir, "vsfc.zip")
    candidates = [os.path.join(data_dir, "UIT-VSFC"), data_dir]
    extracted_base = next(
        (p for p in candidates if os.path.exists(os.path.join(p, "train", "sents.txt"))),
        None,
    )

    if extracted_base:
        def _read_split_folder(folder_name):
            sent_path = os.path.join(extracted_base, folder_name, "sents.txt")
            label_path = os.path.join(extracted_base, folder_name, "sentiments.txt")
            with open(sent_path, "r", encoding="utf-8") as f:
                sentences = [line.strip() for line in f.readlines()]
            with open(label_path, "r", encoding="utf-8") as f:
                labels = [int(line.strip()) for line in f.readlines()]
            min_len = min(len(sentences), len(labels))
            return pd.DataFrame({"Sentence": sentences[:min_len], "Sentiment": labels[:min_len]})

        return (
            _read_split_folder("train"),
            _read_split_folder("dev"),
            _read_split_folder("test"),
        )

    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            def _read_split_zip(folder_name):
                sent_file = f"UIT-VSFC/{folder_name}/sents.txt"
                label_file = f"UIT-VSFC/{folder_name}/sentiments.txt"
                with z.open(sent_file) as f:
                    sentences = [line.decode("utf-8").strip() for line in f.readlines()]
                with z.open(label_file) as f:
                    labels = [int(line.decode("utf-8").strip()) for line in f.readlines()]
                min_len = min(len(sentences), len(labels))
                return pd.DataFrame({"Sentence": sentences[:min_len], "Sentiment": labels[:min_len]})

            return (
                _read_split_zip("train"),
                _read_split_zip("dev"),
                _read_split_zip("test"),
            )

    raise FileNotFoundError(
        f"Could not find UIT-VSFC in '{data_dir}' (extracted folder or vsfc.zip)."
    )


def load_vsfc_ekman(data_dir="data/vsfc-ekman"):
    """Load VSFC-Ekman as (train, dev, test), columns: Sentence, Emotion."""
    zip_path = os.path.join(data_dir, "vsfcekman.zip")

    if os.path.exists(os.path.join(data_dir, "train.csv")):
        df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
        df_valid = pd.read_csv(os.path.join(data_dir, "dev.csv"))
        df_test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    elif os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            df_train = pd.read_csv(z.open("train.csv"))
            df_valid = pd.read_csv(z.open("dev.csv"))
            df_test = pd.read_csv(z.open("test.csv"))
    else:
        raise FileNotFoundError(
            f"Could not find VSFC-Ekman in '{data_dir}' (CSV files or vsfcekman.zip)."
        )

    for df in [df_train, df_valid, df_test]:
        df.rename(columns={"Text": "Sentence", "Ekman_Label": "Emotion"}, inplace=True)
        df.dropna(subset=["Sentence", "Emotion"], inplace=True)
    return df_train, df_valid, df_test


def load_vsmec(data_dir="data/vsmec"):
    """Load UIT-VSMEC as (train, dev, test), columns: Sentence, Emotion."""
    zip_path = os.path.join(data_dir, "uit-vsmec.zip")

    if os.path.exists(os.path.join(data_dir, "train_nor_811.xlsx")):
        df_train = pd.read_excel(os.path.join(data_dir, "train_nor_811.xlsx"))
        df_valid = pd.read_excel(os.path.join(data_dir, "valid_nor_811.xlsx"))
        df_test = pd.read_excel(os.path.join(data_dir, "test_nor_811.xlsx"))
    elif os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            df_train = pd.read_excel(z.open("train_nor_811.xlsx"))
            df_valid = pd.read_excel(z.open("valid_nor_811.xlsx"))
            df_test = pd.read_excel(z.open("test_nor_811.xlsx"))
    else:
        raise FileNotFoundError(
            f"Could not find UIT-VSMEC in '{data_dir}' (Excel files or uit-vsmec.zip)."
        )

    for df in [df_train, df_valid, df_test]:
        df.dropna(subset=["Sentence", "Emotion"], inplace=True)
    return df_train, df_valid, df_test


# -----------------------------------------------------------------------------
# English extension loaders
# -----------------------------------------------------------------------------

ENGLISH_CLASSES = [
    "Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"
]

DAILYDIALOG_ID_TO_EMOTION: Dict[int, str] = {
    0: "Neutral",
    1: "Anger",
    2: "Disgust",
    3: "Fear",
    4: "Happiness",
    5: "Sadness",
    6: "Surprise",
}

MELD_EMOTION_MAP: Dict[str, str] = {
    "anger": "Anger",
    "disgust": "Disgust",
    "fear": "Fear",
    "joy": "Happiness",
    "neutral": "Neutral",
    "sadness": "Sadness",
    "surprise": "Surprise",
}


def _find_recursive(root: str | Path, filename: str) -> Path:
    root = Path(root)
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find '{filename}' below {root}")
    if len(matches) > 1:
        # Prefer the shallowest path, which is normally the canonical dataset copy.
        matches = sorted(matches, key=lambda p: (len(p.parts), str(p)))
    return matches[0]


def _clean_english_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["Sentence", "Emotion"]].copy()
    out["Sentence"] = out["Sentence"].astype(str).str.strip()
    out["Emotion"] = out["Emotion"].astype(str).str.strip()
    out = out[(out["Sentence"] != "") & out["Emotion"].isin(ENGLISH_CLASSES)]
    return out.reset_index(drop=True)


def _read_dailydialog_split(base: Path, split: str) -> pd.DataFrame:
    """Read the official DailyDialog text/emotion files and flatten to utterances."""
    # Official naming uses validation (not dev).
    folder = base / split
    dialogue_path = folder / f"dialogues_{split}.txt"
    emotion_path = folder / f"dialogues_emotion_{split}.txt"

    if not dialogue_path.exists() or not emotion_path.exists():
        # Accept a directory in which the files are directly stored.
        dialogue_path = _find_recursive(base, f"dialogues_{split}.txt")
        emotion_path = _find_recursive(base, f"dialogues_emotion_{split}.txt")

    rows = []
    with open(dialogue_path, "r", encoding="utf-8", errors="replace") as fd, open(
        emotion_path, "r", encoding="utf-8", errors="replace"
    ) as fe:
        for dialogue_id, (dialogue_line, emotion_line) in enumerate(zip(fd, fe)):
            utterances = [x.strip() for x in dialogue_line.strip().split("__eou__") if x.strip()]
            emotion_ids = [int(x) for x in emotion_line.strip().split() if x.strip()]
            if len(utterances) != len(emotion_ids):
                raise ValueError(
                    f"DailyDialog split={split}, dialogue={dialogue_id}: "
                    f"{len(utterances)} utterances != {len(emotion_ids)} emotion labels"
                )
            for utterance_id, (text, emo_id) in enumerate(zip(utterances, emotion_ids)):
                rows.append(
                    {
                        "Sentence": text,
                        "Emotion": DAILYDIALOG_ID_TO_EMOTION[emo_id],
                        "Dialogue_ID": dialogue_id,
                        "Utterance_ID": utterance_id,
                    }
                )
    return pd.DataFrame(rows)


def load_dailydialog(data_dir="data/english/dailydialog") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load DailyDialog train/validation/test and flatten dialogues to utterance rows.

    Supported inputs:
      1. Official extracted dataset with train/, validation/, test/ folders.
      2. Preprocessed train.csv, dev.csv (or validation.csv), test.csv containing
         Sentence and Emotion columns.

    Returns dataframes with at least Sentence and Emotion columns.  Original official
    split boundaries are preserved; no new random split is created.
    """
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"DailyDialog directory not found: {base}")

    # Convenient preprocessed CSV form.
    csv_train = list(base.rglob("train.csv"))
    csv_test = list(base.rglob("test.csv"))
    csv_dev = list(base.rglob("dev.csv")) or list(base.rglob("validation.csv"))
    if csv_train and csv_dev and csv_test:
        return tuple(
            _clean_english_frame(pd.read_csv(p)) for p in (csv_train[0], csv_dev[0], csv_test[0])
        )

    train = _read_dailydialog_split(base, "train")
    dev = _read_dailydialog_split(base, "validation")
    test = _read_dailydialog_split(base, "test")
    return _clean_english_frame(train), _clean_english_frame(dev), _clean_english_frame(test)


def load_meld(data_dir="data/english/meld") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load text-only MELD from the three official *_sent_emo.csv files.

    Only the Utterance and Emotion columns are used.  Dialogue metadata are ignored so
    the experimental unit remains an utterance, matching the sentence-level design of
    the current manuscript.
    """
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"MELD directory not found: {base}")

    paths = {
        "train": _find_recursive(base, "train_sent_emo.csv"),
        "dev": _find_recursive(base, "dev_sent_emo.csv"),
        "test": _find_recursive(base, "test_sent_emo.csv"),
    }

    def _load(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        required = {"Utterance", "Emotion"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"MELD file {path} is missing columns: {sorted(missing)}")
        out = pd.DataFrame(
            {
                "Sentence": df["Utterance"].astype(str),
                "Emotion": df["Emotion"].astype(str).str.lower().map(MELD_EMOTION_MAP),
            }
        )
        if out["Emotion"].isna().any():
            bad = sorted(df.loc[out["Emotion"].isna(), "Emotion"].astype(str).unique())
            raise ValueError(f"Unexpected MELD emotion labels: {bad}")
        return _clean_english_frame(out)

    return _load(paths["train"]), _load(paths["dev"]), _load(paths["test"])


def load_dataset_by_name(name: str, data_dir: str):
    """Small dispatcher used by ``scripts/run_experiment.py``."""
    key = str(name).strip().lower().replace("-", "").replace("_", "")
    if key == "dailydialog":
        return load_dailydialog(data_dir)
    if key == "meld":
        return load_meld(data_dir)
    if key in {"uitvsfc", "vsfc"}:
        return load_vsfc(data_dir)
    if key in {"vsfcekman", "ekman"}:
        return load_vsfc_ekman(data_dir)
    if key in {"uitvsmec", "vsmec"}:
        return load_vsmec(data_dir)
    raise ValueError(f"Unsupported dataset name: {name}")
