#!/usr/bin/env python3
"""Build the English lexical layer for the IJMLC ontology-v2 experiments.

This script does not redistribute the NRC source file. The user supplies a locally
licensed NRC Emotion Lexicon file via ``--nrc``.  Emotion and polarity associations
come from NRC; appraisal semantics are inferred from the supplied ontology v2 by
following stimulus-level semantic relations already encoded in the RDF graph.

The resulting structured entry remains compatible with the 24-dimensional interface:
7 emotion + 9 appraisal + 3 intensity + 3 polarity + 2 negation/context.

Scientific conservatism
-----------------------
Intensity is *not* inferred from emotion alone, because the NRC lexicon does not
provide intensity evidence. Consequently, the three intensity dimensions remain zero
unless a future resource explicitly supplies intensity annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

from rdflib import Graph, Namespace


EKMAN = Namespace("http://example.org/ekman-ontology#")

EMOTION_MAP = {
    "anger": "Anger",
    "disgust": "Disgust",
    "fear": "Fear",
    "joy": "Happiness",
    "sadness": "Sadness",
    "surprise": "Surprise",
}

CLASSES = [
    "Anger",
    "Disgust",
    "Fear",
    "Happiness",
    "Neutral",
    "Sadness",
    "Surprise",
]

APPRAISALS = [
    "GoalObstructionAppraisal",
    "PleasantnessAppraisal",
    "UnpleasantnessAppraisal",
    "DangerousnessAppraisal",
    "LossAppraisal",
    "SuddennessAppraisal",
    "LowPredictabilityAppraisal",
    "UnpredictabilityAppraisal",
    "UnfairnessAppraisal",
]

POLARITIES = [
    "PositivePolarity",
    "NegativePolarity",
    "NeutralPolarity",
]

POLARITY_MAP = {
    "positive": "PositivePolarity",
    "negative": "NegativePolarity",
}

NEGATION_CUES = [
    "not",
    "no",
    "never",
    "without",
    "neither",
    "nor",
    "hardly",
    "scarcely",
    "barely",
]

# Happiness is aligned with the original ontology's Enjoyment concept in v2.
EMOTION_URI_ALIASES = {
    "Anger": ["Anger"],
    "Disgust": ["Disgust"],
    "Fear": ["Fear"],
    "Happiness": ["Happiness", "Enjoyment"],
    "Neutral": ["Neutral", "NeutralEmotion"],
    "Sadness": ["Sadness"],
    "Surprise": ["Surprise"],
}


def local_name(uri) -> str:
    text = str(uri)
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    return text.rstrip("/").rsplit("/", 1)[-1]


def normalize_word(word: str) -> str:
    word = str(word).strip().lower().replace("_", " ")
    return re.sub(r"\s+", " ", word)


def _load_long_format(path: Path):
    emotions = defaultdict(set)
    polarities = defaultdict(set)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 3:
                continue
            word, category, assoc = (
                parts[0],
                parts[1].strip().lower(),
                parts[2].strip(),
            )
            try:
                assoc = int(float(assoc))
            except ValueError:
                continue
            if assoc != 1:
                continue
            word = normalize_word(word)
            if category in EMOTION_MAP:
                emotions[word].add(EMOTION_MAP[category])
            elif category in POLARITY_MAP:
                polarities[word].add(POLARITY_MAP[category])
    return emotions, polarities


def _load_wide_format(path: Path):
    rows = []
    with path.open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        reader = csv.DictReader(f, dialect=dialect)
        headers = [h.strip() for h in (reader.fieldnames or [])]
        for row in reader:
            rows.append({str(k).strip(): v for k, v in row.items()})

    if not rows:
        raise ValueError("No rows found in NRC file")

    word_candidates = [
        h for h in headers if h.lower() in {"word", "english word", "term"}
    ]
    word_col = word_candidates[0] if word_candidates else headers[0]

    emotions = defaultdict(set)
    polarities = defaultdict(set)
    for row in rows:
        word = normalize_word(row.get(word_col, ""))
        if not word:
            continue
        lower_row = {str(k).strip().lower(): v for k, v in row.items()}
        for source, target in EMOTION_MAP.items():
            try:
                if int(float(lower_row.get(source, 0) or 0)) == 1:
                    emotions[word].add(target)
            except ValueError:
                pass
        for source, target in POLARITY_MAP.items():
            try:
                if int(float(lower_row.get(source, 0) or 0)) == 1:
                    polarities[word].add(target)
            except ValueError:
                pass
    return emotions, polarities


def load_nrc(path: Path):
    """Detect common NRC long/wide formats."""
    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        first = f.readline().strip()
    parts = first.split("\t")
    if len(parts) >= 3 and parts[-1].strip() in {
        "0", "1", "0.0", "1.0"
    }:
        return _load_long_format(path)
    return _load_wide_format(path)


def load_overrides(path: Path | None):
    """Optional author-curated emotion corrections.

    Expected columns: ``word,keep_emotions,remove_emotions``.
    Curation should be performed using training vocabulary only.
    """
    if path is None:
        return {}

    overrides = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = normalize_word(row.get("word", ""))
            if not word:
                continue

            def split_list(value):
                return {
                    x.strip()
                    for x in re.split(r"[,;|]", value or "")
                    if x.strip()
                }

            overrides[word] = {
                "keep": split_list(row.get("keep_emotions", "")),
                "remove": split_list(row.get("remove_emotions", "")),
            }
    return overrides


def derive_emotion_semantics(ontology_path: Path):
    """Derive emotion→appraisal/polarity semantics from ontology v2.

    The function reads only relations that are actually present in the RDF graph.
    It does not silently add external psychological knowledge.
    """
    graph = Graph()
    graph.parse(str(ontology_path))

    allowed_apps = set(APPRAISALS)
    allowed_pols = set(POLARITIES)
    mapping = {}

    for emotion in CLASSES:
        apps = set()
        pols = set()
        evidence_stimuli = set()

        for alias in EMOTION_URI_ALIASES[emotion]:
            emotion_uri = EKMAN[alias]
            for stimulus in graph.subjects(
                EKMAN.hasEmotionCategory, emotion_uri
            ):
                stimulus_apps = {
                    local_name(obj)
                    for obj in graph.objects(stimulus, EKMAN.evokesAppraisal)
                    if local_name(obj) in allowed_apps
                }
                stimulus_pols = {
                    local_name(obj)
                    for obj in graph.objects(stimulus, EKMAN.hasPolarity)
                    if local_name(obj) in allowed_pols
                }
                if stimulus_apps or stimulus_pols:
                    evidence_stimuli.add(local_name(stimulus))
                apps |= stimulus_apps
                pols |= stimulus_pols

        mapping[emotion] = {
            "appraisals": sorted(apps),
            "polarities": sorted(pols),
            "evidence_stimuli": sorted(evidence_stimuli),
        }

    return mapping, len(graph)


def build_lexicons(
    nrc_path: Path,
    ontology_path: Path,
    confidence: float,
    overrides_path: Path | None = None,
):
    emotion_map, polarity_map = load_nrc(nrc_path)
    overrides = load_overrides(overrides_path)
    ontology_semantics, triple_count = derive_emotion_semantics(
        ontology_path
    )

    structured: Dict[str, list] = {}
    flat: Dict[str, list] = {}
    all_words = sorted(set(emotion_map) | set(polarity_map))

    for word in all_words:
        if len(word) < 2:
            continue

        emotions: Set[str] = set(emotion_map.get(word, set()))
        if word in overrides:
            keep = overrides[word]["keep"]
            remove = overrides[word]["remove"]
            if keep:
                emotions &= keep
            emotions -= remove

        nrc_polarities = set(polarity_map.get(word, set()))
        if not emotions and not nrc_polarities:
            continue

        inferred_appraisals = set()
        ontology_polarities = set()
        semantic_sources = {}

        for emotion in sorted(emotions):
            info = ontology_semantics.get(emotion, {})
            inferred_appraisals.update(info.get("appraisals", []))
            ontology_polarities.update(info.get("polarities", []))
            semantic_sources[emotion] = {
                "appraisals": info.get("appraisals", []),
                "polarities": info.get("polarities", []),
                "evidence_stimuli": info.get("evidence_stimuli", []),
            }

        combined_polarities = sorted(
            nrc_polarities | ontology_polarities
        )

        structured[word] = [
            {
                "emotions": {e: 1.0 for e in sorted(emotions)},
                "score": float(confidence),
                "appraisals": sorted(inferred_appraisals),
                "intensities": [],
                "polarities": combined_polarities,
                "isnegation": False,
                "evidence_type": "NRC+OntologyV2",
                "semantic_sources": semantic_sources,
            }
        ]

        flat[word] = [
            1.0 if class_name in emotions else 0.0
            for class_name in CLASSES
        ]

    for cue in NEGATION_CUES:
        structured[cue] = [
            {
                "emotions": {},
                "score": 1.0,
                "appraisals": [],
                "intensities": [],
                "polarities": [],
                "isnegation": True,
                "evidence_type": "English_Negation_Cue",
            }
        ]

    appraisal_enriched_keys = sum(
        bool(entries[0].get("appraisals"))
        for entries in structured.values()
        if entries
    )

    summary = {
        "nrc_source_file": nrc_path.name,
        "ontology_file": ontology_path.name,
        "ontology_triples": triple_count,
        "confidence": confidence,
        "classes": CLASSES,
        "structured_keys": len(structured),
        "flat_emotion_keys": sum(any(v) for v in flat.values()),
        "appraisal_enriched_keys": appraisal_enriched_keys,
        "negation_cues": NEGATION_CUES,
        "curation_file": (
            overrides_path.name if overrides_path else None
        ),
        "ontology_semantics": ontology_semantics,
        "note": (
            "Emotion/polarity evidence originates from NRC. Appraisal slots "
            "are inferred only from explicit ontology-v2 relations. Intensity "
            "is deliberately not inferred without lexical intensity evidence."
        ),
    }
    return structured, flat, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nrc",
        required=True,
        help="Path to the locally licensed NRC Emotion Lexicon file",
    )
    parser.add_argument(
        "--ontology",
        default="ontology/ekman_appraisal_ontology_v2.rdf",
        help="Ontology v2 RDF/XML file used for semantic enrichment",
    )
    parser.add_argument(
        "--output",
        default="ontology/english_nrc_ontology_v2_structured.json",
    )
    parser.add_argument(
        "--flat-output",
        default="ontology/english_nrc_flat7.json",
    )
    parser.add_argument(
        "--summary-output",
        default="ontology/english_nrc_ontology_v2_build_summary.json",
    )
    parser.add_argument(
        "--overrides",
        default=None,
        help="Optional training-vocabulary-only curated overrides CSV",
    )
    parser.add_argument("--confidence", type=float, default=0.50)
    args = parser.parse_args()

    nrc_path = Path(args.nrc)
    ontology_path = Path(args.ontology)
    if not nrc_path.exists():
        raise FileNotFoundError(nrc_path)
    if not ontology_path.exists():
        raise FileNotFoundError(ontology_path)

    overrides = Path(args.overrides) if args.overrides else None
    structured, flat, summary = build_lexicons(
        nrc_path=nrc_path,
        ontology_path=ontology_path,
        confidence=args.confidence,
        overrides_path=overrides,
    )

    outputs = [
        Path(args.output),
        Path(args.flat_output),
        Path(args.summary_output),
    ]
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    Path(args.output).write_text(
        json.dumps(structured, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(args.flat_output).write_text(
        json.dumps(flat, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    Path(args.summary_output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Structured ontology-v2 lexicon -> {args.output}")
    print(f"Flat 7-d NRC lexicon           -> {args.flat_output}")


if __name__ == "__main__":
    main()
