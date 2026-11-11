"""Ontology feature extraction engine with pluggable language adapters.

This file is a backward-compatible replacement for the repository's current
``src/ontology_engine.py``.  The Vietnamese default path is intentionally preserved,
while two optional arguments enable the IJMLC English experiments:

``language_adapter``
    Object implementing ``tokenize(text)``.  Defaults to ``VietnameseAdapter``.

``external_lexicon_path``
    JSON lexicon in the same internal entry format used by the RDF compiler.  When
    supplied, the TBox/RDF is still loaded (including emotion-similarity edges), but
    lexical matching uses the external language-specific layer.

The output remains exactly 24 dimensions:
7 emotion + 9 appraisal + 3 intensity + 3 polarity + 2 negation.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, Optional

import numpy as np
from rdflib import Graph, Namespace, RDF
from unidecode import unidecode

from src.language_adapters import BaseLanguageAdapter, VietnameseAdapter


class OntologyEngine:
    """Extract structured 24-d ontology vectors for seven-class emotion tasks."""

    def __init__(
        self,
        rdf_path,
        default_conf=0.85,
        manual_boost=1.2,
        negation_attenuation=0.5,
        alpha_similarity=0.2,
        verbose=True,
        language_adapter: Optional[BaseLanguageAdapter] = None,
        external_lexicon_path: Optional[str] = None,
    ):
        self.EKMAN = Namespace("http://example.org/ekman-ontology#")
        self.default_conf = float(default_conf)
        # Kept for backward compatibility.  The original repository did not apply it.
        self.manual_boost = float(manual_boost) if manual_boost is not None else 1.0
        self.negation_attenuation = float(negation_attenuation)
        self.alpha_similarity = float(alpha_similarity)
        self.verbose = bool(verbose)
        self.language_adapter = language_adapter or VietnameseAdapter()
        self.external_lexicon_path = external_lexicon_path

        self.ALLEMOTIONS = [
            "Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"
        ]
        self.ALLAPPRAISALS = [
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
        self.INTENSITY_CLASSES = ["HighIntensity", "MediumIntensity", "LowIntensity"]
        self.POLARITY_CLASSES = [
            "PositivePolarity", "NegativePolarity", "NeutralPolarity"
        ]

        self.EMOTION_MAPPING = {
            "Enjoyment": "Happiness",
            "Happiness": "Happiness",
            "Anger": "Anger",
            "Disgust": "Disgust",
            "Fear": "Fear",
            "Sadness": "Sadness",
            "Surprise": "Surprise",
            "NeutralEmotion": "Neutral",
            "Neutral": "Neutral",
            "Other": "Neutral",
            "NegationCue": "NegationCue",
        }

        self.IMPLIED_POLARITY = {
            "Anger": "NegativePolarity",
            "Disgust": "NegativePolarity",
            "Fear": "NegativePolarity",
            "Sadness": "NegativePolarity",
            "Happiness": "PositivePolarity",
            "Enjoyment": "PositivePolarity",
            "Surprise": "PositivePolarity",
            "UnpleasantnessAppraisal": "NegativePolarity",
            "GoalObstructionAppraisal": "NegativePolarity",
            "LossAppraisal": "NegativePolarity",
            "UnfairnessAppraisal": "NegativePolarity",
            "DangerousnessAppraisal": "NegativePolarity",
            "PleasantnessAppraisal": "PositivePolarity",
        }

        self.AMBIGUOUS_TERMS = {
            "hay", "chả", "cơ", "ngờ", "ý", "quá", "lại", "tôi", "tao", "tớ",
            "mình", "bạn", "nó", "hắn", "anh", "chị", "em", "là", "thì", "mà",
            "bị", "được", "cái", "con", "người",
        }
        self.VALID_DOMAINS = {"GeneralDomain", "SocialDomain", "EducationDomain"}

        if self.verbose:
            print(f"Loading Knowledge Graph: {rdf_path}")
        self.g = Graph()
        if not os.path.exists(rdf_path):
            raise FileNotFoundError(f"RDF file not found at: {rdf_path}")
        self.g.parse(rdf_path)
        if self.verbose:
            print(f"Loaded {len(self.g)} triples.")

        self.sim_matrix = self._build_similarity_matrix()
        if external_lexicon_path:
            self.lexicon_map = self._load_external_lexicon(external_lexicon_path)
        else:
            self.lexicon_map = self._build_strict_lexicon()

        self.mwe_max_len = max([1] + [len(k.split()) for k in self.lexicon_map.keys()])
        if self.verbose:
            src = f"external lexicon {external_lexicon_path}" if external_lexicon_path else "RDF ABox"
            print(f"Lexicon initialized with {len(self.lexicon_map)} keys from {src}.")

    def _norm(self, s):
        if not s:
            return ""
        return re.sub(r"\s+", " ", str(s).lower().strip().replace("_", " ")).strip()

    def _local(self, uri):
        uri_str = str(uri)
        return uri_str.split("#")[-1] if "#" in uri_str else uri_str.split("/")[-1]

    def _tokens(self, text):
        return [self._norm(t) for t in self.language_adapter.tokenize(str(text)) if self._norm(t)]

    def _load_external_lexicon(self, path: str) -> Dict[str, list]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("External lexicon JSON must be a dictionary keyed by word/phrase.")

        out = {}
        for raw_key, raw_entries in raw.items():
            key = self._norm(raw_key)
            if not key:
                continue
            entries = raw_entries if isinstance(raw_entries, list) else [raw_entries]
            cleaned = []
            for ent in entries:
                if not isinstance(ent, dict):
                    continue
                emotions = {
                    self.EMOTION_MAPPING.get(str(k), str(k)): float(v)
                    for k, v in (ent.get("emotions") or {}).items()
                    if self.EMOTION_MAPPING.get(str(k), str(k)) in self.ALLEMOTIONS
                }
                is_neg = bool(ent.get("isnegation", False))
                if not emotions and not is_neg:
                    continue
                cleaned.append(
                    {
                        "emotions": emotions,
                        "score": float(ent.get("score", self.default_conf)),
                        "appraisals": list(ent.get("appraisals") or []),
                        "intensities": list(ent.get("intensities") or []),
                        "polarities": list(ent.get("polarities") or []),
                        "isnegation": is_neg,
                        "evidence_type": ent.get("evidence_type", "ExternalLexicon"),
                    }
                )
            if cleaned:
                out[key] = cleaned
        return out

    def _build_similarity_matrix(self):
        n = len(self.ALLEMOTIONS)
        mat = np.zeros((n, n), dtype=np.float32)
        np.fill_diagonal(mat, 1.0)
        try:
            q = """
            PREFIX ekman: <http://example.org/ekman-ontology#>
            SELECT ?score ?e1 ?e2 WHERE {
                ?edge a ekman:EmotionSimilarityEdge ;
                      ekman:similarityScore ?score ;
                      ekman:linksEmotion ?e1 ;
                      ekman:linksEmotion ?e2 .
                FILTER(?e1 != ?e2)
            }
            """
            for row in self.g.query(q):
                e1 = self.EMOTION_MAPPING.get(self._local(row.e1))
                e2 = self.EMOTION_MAPPING.get(self._local(row.e2))
                if e1 in self.ALLEMOTIONS and e2 in self.ALLEMOTIONS:
                    i, j = self.ALLEMOTIONS.index(e1), self.ALLEMOTIONS.index(e2)
                    mat[i, j] = mat[j, i] = max(mat[i, j], float(row.score))
        except Exception as exc:
            if self.verbose:
                print(f"Similarity matrix query warning: {exc}")
        row_sum = mat.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return mat / row_sum

    def _infer_attributes(self, resource, emotions: Iterable[str]):
        ek = self.EKMAN
        a, i, p = set(), set(), set()

        def gather(res):
            for o in self.g.objects(res, ek.evokesAppraisal):
                a.add(self._local(o))
            for o in self.g.objects(res, ek.hasIntensity):
                i.add(self._local(o))
            for o in self.g.objects(res, ek.hasPolarity):
                p.add(self._local(o))

        gather(resource)
        for t in self.g.objects(resource, RDF.type):
            gather(t)

        if not p:
            for e in emotions:
                if e in self.IMPLIED_POLARITY:
                    p.add(self.IMPLIED_POLARITY[e])
        if not p:
            for app in a:
                if app in self.IMPLIED_POLARITY:
                    p.add(self.IMPLIED_POLARITY[app])
        return a, i, p

    def _build_strict_lexicon(self):
        ek = self.EKMAN
        lookup = {}
        mix_cache, sim_cache = {}, {}

        query_mix = """
        PREFIX ekman: <http://example.org/ekman-ontology#>
        SELECT ?m ?e ?w WHERE {
            ?m a ekman:EmotionMixture ; ekman:hasComponent ?c .
            ?c ekman:componentEmotion ?e ; ekman:componentWeight ?w .
        }
        """
        for r in self.g.query(query_mix):
            m = r.m
            e = self.EMOTION_MAPPING.get(self._local(r.e))
            w = float(r.w)
            if e in self.ALLEMOTIONS:
                mix_cache.setdefault(m, {})[e] = mix_cache.setdefault(m, {}).get(e, 0.0) + w

        for s, _, o in self.g.triples((None, ek.hasEmotionCategory, None)):
            e = self.EMOTION_MAPPING.get(self._local(o))
            if e in self.ALLEMOTIONS:
                sim_cache[s] = [e]

        for s in self.g.subjects(RDF.type, None):
            if s in mix_cache or s in sim_cache:
                continue
            types = list(self.g.objects(s, RDF.type))
            if ek.NegationCue in types:
                sim_cache[s] = ["NegationCue"]
            else:
                for t in types:
                    mapped = self.EMOTION_MAPPING.get(self._local(t))
                    if mapped in self.ALLEMOTIONS:
                        sim_cache[s] = [mapped]
                        break

        q = """
        PREFIX ekman: <http://example.org/ekman-ontology#>
        SELECT ?word ?stim ?ev ?base ?type ?dom WHERE {
            ?evoc a ekman:Evocation ;
                  ekman:fromLexiconEntry ?lex ;
                  ekman:toStimulus ?stim .
            ?lex ekman:hasContent ?word .
            OPTIONAL { ?evoc ekman:confidenceScore ?ev }
            OPTIONAL { ?evoc ekman:evidenceType ?type }
            OPTIONAL { ?lex ekman:baseIntensityScore ?base }
            OPTIONAL { ?lex ekman:belongsToDomain ?dom }
        }
        """
        for r in self.g.query(q):
            if r.dom and self._local(r.dom) not in self.VALID_DOMAINS:
                continue
            word = self._norm(r.word)
            if len(word) < 2 or word in self.AMBIGUOUS_TERMS:
                continue

            emotions = {}
            if r.stim in mix_cache:
                raw = mix_cache[r.stim]
                total = sum(raw.values())
                if total > 0:
                    emotions = {k: v / total for k, v in raw.items()}
            elif r.stim in sim_cache:
                emotions = {k: 1.0 for k in sim_cache[r.stim] if k != "NegationCue"}

            is_neg = r.stim in sim_cache and "NegationCue" in sim_cache[r.stim]
            if not emotions and not is_neg:
                continue

            apps, intensities, polarities = self._infer_attributes(r.stim, emotions.keys())
            entry_score = float(r.ev or self.default_conf) * float(r.base or 1.0)
            entry = {
                "emotions": emotions,
                "score": entry_score,
                "appraisals": list(apps),
                "intensities": list(intensities),
                "polarities": list(polarities),
                "isnegation": is_neg,
                "evidence_type": str(r.type) if r.type else "OntologyABox",
            }
            lookup.setdefault(word, []).append(entry)
            no_diacritics = self._norm(unidecode(word))
            if no_diacritics != word:
                lookup.setdefault(no_diacritics, []).append(entry)
        return lookup

    def getvector(self, text, debug=False):
        flat = self._tokens(text)
        ve = np.zeros(7, dtype=np.float32)
        va = np.zeros(9, dtype=np.float32)
        vi = np.zeros(3, dtype=np.float32)
        vp = np.zeros(3, dtype=np.float32)
        vn = np.zeros(2, dtype=np.float32)

        idx, n = 0, len(flat)
        hit_trace = []
        negation_active = False

        while idx < n:
            matches, match_len = None, 0
            for span_len in range(min(self.mwe_max_len, n - idx), 0, -1):
                phrase = " ".join(flat[idx : idx + span_len])
                if phrase in self.lexicon_map:
                    matches, match_len = self.lexicon_map[phrase], span_len
                    break

            if not matches:
                idx += 1
                continue

            phrase = " ".join(flat[idx : idx + match_len])
            if debug:
                hit_trace.append({"phrase": phrase, "data": matches[0]})

            for ent in matches:
                if ent.get("isnegation", False):
                    vn[0] = 1.0
                    negation_active = True
                    continue

                score = float(ent.get("score", self.default_conf))
                if negation_active:
                    score *= self.negation_attenuation

                for e, weight in ent.get("emotions", {}).items():
                    if e in self.ALLEMOTIONS:
                        ve[self.ALLEMOTIONS.index(e)] += score * float(weight)
                for x in ent.get("appraisals", []):
                    if x in self.ALLAPPRAISALS:
                        va[self.ALLAPPRAISALS.index(x)] += score
                for x in ent.get("intensities", []):
                    if x in self.INTENSITY_CLASSES:
                        vi[self.INTENSITY_CLASSES.index(x)] = 1.0
                for x in ent.get("polarities", []):
                    if x in self.POLARITY_CLASSES:
                        vp[self.POLARITY_CLASSES.index(x)] += score

            idx += match_len

        if self.alpha_similarity > 0:
            ve = np.clip(ve + self.alpha_similarity * (ve @ self.sim_matrix), 0, None)

        vector = np.concatenate([np.tanh(ve), np.tanh(va), vi, vp, vn]).astype(np.float32)
        return (vector, flat, hit_trace) if debug else vector


class VSFCOntologyEngine(OntologyEngine):
    """Three-class UIT-VSFC engine preserving original polarity injection behaviour."""

    def __init__(
        self,
        rdf_path,
        default_conf=0.9,
        negation_attenuation=0.4,
        alpha_similarity=0.2,
        verbose=True,
        language_adapter: Optional[BaseLanguageAdapter] = None,
        external_lexicon_path: Optional[str] = None,
    ):
        super().__init__(
            rdf_path=rdf_path,
            default_conf=default_conf,
            manual_boost=None,
            negation_attenuation=negation_attenuation,
            alpha_similarity=alpha_similarity,
            verbose=verbose,
            language_adapter=language_adapter,
            external_lexicon_path=external_lexicon_path,
        )
        self.POS_EMOTIONS = {"Happiness", "Enjoyment", "Surprise"}
        self.NEG_EMOTIONS = {"Anger", "Disgust", "Fear", "Sadness"}

    def getvector(self, text, debug=False):
        flat = self._tokens(text)
        ve = np.zeros(7, dtype=np.float32)
        va = np.zeros(9, dtype=np.float32)
        vi = np.zeros(3, dtype=np.float32)
        vp = np.zeros(3, dtype=np.float32)
        vn = np.zeros(2, dtype=np.float32)

        idx, n = 0, len(flat)
        hit_trace = []
        negation_active = False

        while idx < n:
            matches, match_len = None, 0
            for span_len in range(min(self.mwe_max_len, n - idx), 0, -1):
                phrase = " ".join(flat[idx : idx + span_len])
                if phrase in self.lexicon_map:
                    matches, match_len = self.lexicon_map[phrase], span_len
                    break
            if not matches:
                idx += 1
                continue

            phrase = " ".join(flat[idx : idx + match_len])
            if debug:
                hit_trace.append({"phrase": phrase, "data": matches[0]})

            for ent in matches:
                if ent.get("isnegation", False):
                    vn[0] = 1.0
                    negation_active = True
                    continue

                score = float(ent.get("score", self.default_conf))
                if negation_active:
                    score *= self.negation_attenuation

                emotions = ent.get("emotions", {})
                for e, weight in emotions.items():
                    if e in self.ALLEMOTIONS:
                        ve[self.ALLEMOTIONS.index(e)] += score * float(weight)

                pos_score = sum(float(w) for e, w in emotions.items() if e in self.POS_EMOTIONS)
                neg_score = sum(float(w) for e, w in emotions.items() if e in self.NEG_EMOTIONS)
                neu_score = float(emotions.get("Neutral", 0.0))
                if pos_score > 0:
                    vp[0] += score * pos_score
                if neg_score > 0:
                    vp[1] += score * neg_score
                if neu_score > 0:
                    vp[2] += score * neu_score

                for x in ent.get("polarities", []):
                    if x in self.POLARITY_CLASSES:
                        vp[self.POLARITY_CLASSES.index(x)] += score
                for x in ent.get("appraisals", []):
                    if x in self.ALLAPPRAISALS:
                        va[self.ALLAPPRAISALS.index(x)] += score
                for x in ent.get("intensities", []):
                    if x in self.INTENSITY_CLASSES:
                        vi[self.INTENSITY_CLASSES.index(x)] = 1.0

            idx += match_len

        if self.alpha_similarity > 0:
            ve = np.clip(ve + self.alpha_similarity * (ve @ self.sim_matrix), 0, None)
        vp *= 1.5
        vector = np.concatenate([np.tanh(ve), np.tanh(va), vi, np.tanh(vp), vn]).astype(np.float32)
        return (vector, flat, hit_trace) if debug else vector
