import os
import re
import numpy as np
from pyvi import ViTokenizer
from rdflib import Graph, Namespace, RDF
from unidecode import unidecode


class OntologyEngine:
    """
    Extracts structured lexical knowledge vectors (24-dimensional) from an RDF Ekman Knowledge Graph.
    Used for 7-class emotion classification tasks (VSFC-Ekman, UIT-VSMEC).
    """

    def __init__(
        self,
        rdf_path,
        default_conf=0.85,
        manual_boost=1.2,
        negation_attenuation=0.5,
        alpha_similarity=0.2,
        verbose=True,
    ):
        self.EKMAN = Namespace("http://example.org/ekman-ontology#")
        self.default_conf = float(default_conf)
        self.manual_boost = float(manual_boost) if manual_boost is not None else 1.0
        self.negation_attenuation = float(negation_attenuation)
        self.alpha_similarity = float(alpha_similarity)
        self.verbose = bool(verbose)

        # 1. Target Taxonomy Configurations (7 Ekman Emotion Classes)
        self.ALLEMOTIONS = [
            "Anger",
            "Disgust",
            "Fear",
            "Happiness",
            "Neutral",
            "Sadness",
            "Surprise",
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
        self.INTENSITY_CLASSES = [
            "HighIntensity",
            "MediumIntensity",
            "LowIntensity",
        ]
        self.POLARITY_CLASSES = [
            "PositivePolarity",
            "NegativePolarity",
            "NeutralPolarity",
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

        # Semantic polarity derivation rules
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
        self.VALID_DOMAINS = {
            "GeneralDomain",
            "SocialDomain",
            "EducationDomain",
        }

        # Load RDF Knowledge Graph
        if self.verbose:
            print(f"Loading Knowledge Graph: {rdf_path}")
        self.g = Graph()
        if os.path.exists(rdf_path):
            self.g.parse(rdf_path)
            if self.verbose:
                print(f"Loaded {len(self.g)} triples.")
        else:
            raise FileNotFoundError(f"RDF file not found at: {rdf_path}")

        self.sim_matrix = self._build_similarity_matrix()
        self.lexicon_map = self._build_strict_lexicon()

        self.mwe_max_len = 1
        for k in self.lexicon_map.keys():
            self.mwe_max_len = max(self.mwe_max_len, len(k.split()))

        if self.verbose:
            print(f"Lexicon initialized with {len(self.lexicon_map)} entries.")

    def _norm(self, s):
        """Normalize string to lowercase single-spaced format."""
        if not s:
            return ""
        return re.sub(
            r"\s+", " ", str(s).lower().strip().replace("_", " ")
        ).strip()

    def _local(self, uri):
        """Extract local identifier from RDF URI."""
        uri_str = str(uri)
        return uri_str.split("#")[-1] if "#" in uri_str else uri_str.split("/")[-1]

    def _build_similarity_matrix(self):
        """Build normalized inter-emotion similarity matrix."""
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
            }
            """
            for row in self.g.query(q):
                e1 = self.EMOTION_MAPPING.get(self._local(row.e1))
                e2 = self.EMOTION_MAPPING.get(self._local(row.e2))
                if e1 in self.ALLEMOTIONS and e2 in self.ALLEMOTIONS:
                    i, j = self.ALLEMOTIONS.index(e1), self.ALLEMOTIONS.index(e2)
                    mat[i, j] = mat[j, i] = max(mat[i, j], float(row.score))
        except Exception as e:
            if self.verbose:
                print(f"Similarity matrix query warning: {e}")

        s = mat.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return mat / s

    def _build_strict_lexicon(self):
        """Extract lexicon entries, confidence weights, and emotional properties."""
        ek = self.EKMAN
        lookup = {}
        mix_cache, sim_cache = {}, {}

        # Cache emotion mixtures
        query_mix = """
        PREFIX ekman: <http://example.org/ekman-ontology#>
        SELECT ?m ?e ?w WHERE {
            ?m a ekman:EmotionMixture ;
               ekman:hasComponent ?c .
            ?c ekman:componentEmotion ?e ;
               ekman:componentWeight ?w .
        }
        """
        for r in self.g.query(query_mix):
            m, e, w = r.m, self.EMOTION_MAPPING.get(self._local(r.e)), float(r.w)
            if e in self.ALLEMOTIONS:
                mix_cache.setdefault(m, {})[e] = (
                    mix_cache.setdefault(m, {}).get(e, 0.0) + w
                )

        # Cache direct emotion categories
        for s, _, o in self.g.triples((None, ek.hasEmotionCategory, None)):
            e = self.EMOTION_MAPPING.get(self._local(o))
            if e in self.ALLEMOTIONS:
                sim_cache[s] = [e]

        for s in self.g.subjects(RDF.type, None):
            if s in mix_cache or s in sim_cache:
                continue
            ts = list(self.g.objects(s, RDF.type))
            if ek.NegationCue in ts:
                sim_cache[s] = ["NegationCue"]
            else:
                for t in ts:
                    m = self.EMOTION_MAPPING.get(self._local(t))
                    if m in self.ALLEMOTIONS:
                        sim_cache[s] = [m]
                        break

        # Populate lexicon map
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
            w = self._norm(r.word)
            if len(w) < 2 or w in self.AMBIGUOUS_TERMS:
                continue

            emos = {}
            if r.stim in mix_cache:
                raw = mix_cache[r.stim]
                tot = sum(raw.values())
                if tot > 0:
                    emos = {k: v / tot for k, v in raw.items()}
            elif r.stim in sim_cache:
                lst = sim_cache[r.stim]
                emos = {k: 1.0 for k in lst}

            if not emos:
                continue

            apps, ints, pols = self._infer_attributes(r.stim, emos.keys())
            entry_score = float(r.ev or self.default_conf) * float(
                r.base or 1.0
            )

            entry = {
                "emotions": emos,
                "score": entry_score,
                "appraisals": list(apps),
                "intensities": list(ints),
                "polarities": list(pols),
                "isnegation": "NegationCue" in emos,
            }
            lookup.setdefault(w, []).append(entry)
            wu = self._norm(unidecode(w))
            if wu != w:
                lookup.setdefault(wu, []).append(entry)

        return lookup

    def _infer_attributes(self, s, emos):
        """Infer appraisals, intensities, and polarities using ontological relations."""
        ek = self.EKMAN
        a, i, p = set(), set(), set()

        def g(res):
            for o in self.g.objects(res, ek.evokesAppraisal):
                a.add(self._local(o))
            for o in self.g.objects(res, ek.hasIntensity):
                i.add(self._local(o))
            for o in self.g.objects(res, ek.hasPolarity):
                p.add(self._local(o))

        g(s)
        for t in self.g.objects(s, RDF.type):
            g(t)

        if not p:
            for e in emos:
                if e in self.IMPLIED_POLARITY:
                    p.add(self.IMPLIED_POLARITY[e])
        if not p:
            for app in a:
                if app in self.IMPLIED_POLARITY:
                    p.add(self.IMPLIED_POLARITY[app])

        return a, i, p

    def getvector(self, text, debug=False):
        """Convert input text into a 24-dimensional continuous ontology vector."""
        tokens = ViTokenizer.tokenize(str(text).lower()).split()
        flat = [self._norm(t) for t in tokens if self._norm(t)]

        ve = np.zeros(7, dtype=np.float32)
        va = np.zeros(9, dtype=np.float32)
        vi = np.zeros(3, dtype=np.float32)
        vp = np.zeros(3, dtype=np.float32)
        vn = np.zeros(2, dtype=np.float32)

        idx, n = 0, len(flat)
        hit_trace = []

        while idx < n:
            m, ml = None, 0
            for l_span in range(min(self.mwe_max_len, n - idx), 0, -1):
                p = " ".join(flat[idx : idx + l_span])
                if p in self.lexicon_map:
                    m, ml = self.lexicon_map[p], l_span
                    break

            if not m:
                idx += 1
                continue

            if debug:
                hit_trace.append(
                    {"phrase": " ".join(flat[idx : idx + ml]), "data": m[0]}
                )

            for ent in m:
                if ent["isnegation"]:
                    vn[0] = 1.0
                    continue

                sc = ent["score"]
                if vn[0] > 0:
                    sc *= self.negation_attenuation

                for e, w in ent["emotions"].items():
                    if e in self.ALLEMOTIONS:
                        ve[self.ALLEMOTIONS.index(e)] += sc * w
                for x in ent["appraisals"]:
                    if x in self.ALLAPPRAISALS:
                        va[self.ALLAPPRAISALS.index(x)] += sc
                for x in ent["intensities"]:
                    if x in self.INTENSITY_CLASSES:
                        vi[self.INTENSITY_CLASSES.index(x)] = 1.0
                for x in ent["polarities"]:
                    if x in self.POLARITY_CLASSES:
                        vp[self.POLARITY_CLASSES.index(x)] += sc

            idx += ml

        if self.alpha_similarity > 0:
            ve = np.clip(
                ve + self.alpha_similarity * (ve @ self.sim_matrix), 0, None
            )

        fin = np.concatenate([np.tanh(ve), np.tanh(va), vi, vp, vn]).astype(
            np.float32
        )
        return (fin, flat, hit_trace) if debug else fin


# TODO: VSFCOntologyEngine vs OntologyEngine separation
# VSFCOntologyEngine is dedicated to 3-class Sentiment Analysis (UIT-VSFC).
# It performs explicit Polarity Score Injection (mapping emotions -> positive/negative/neutral polarity),
# applies a 1.5x amplification scaling to polarity, and applies tanh() to vp.
# In contrast, OntologyEngine is designed for 7-class Emotion Classification without polarity injection.
class VSFCOntologyEngine(OntologyEngine):
    """
    Specialized Ontology Engine for 3-class Sentiment Analysis (UIT-VSFC).
    Injects polarity scores derived from emotional associations with a 1.5x amplification factor.
    """

    def __init__(
        self,
        rdf_path,
        default_conf=0.9,
        negation_attenuation=0.4,
        alpha_similarity=0.2,
        verbose=True,
    ):
        super().__init__(
            rdf_path=rdf_path,
            default_conf=default_conf,
            manual_boost=None,
            negation_attenuation=negation_attenuation,
            alpha_similarity=alpha_similarity,
            verbose=verbose,
        )
        # Emotion to Polarity mapping for score injection
        self.POS_EMOTIONS = {"Happiness", "Enjoyment", "Surprise"}
        self.NEG_EMOTIONS = {"Anger", "Disgust", "Fear", "Sadness"}

    def getvector(self, text, debug=False):
        """Extract a 24-dimensional feature vector with sentiment polarity injection."""
        tokens = ViTokenizer.tokenize(str(text).lower()).split()
        flat = [self._norm(t) for t in tokens if self._norm(t)]

        ve = np.zeros(7, dtype=np.float32)
        va = np.zeros(9, dtype=np.float32)
        vi = np.zeros(3, dtype=np.float32)
        vp = np.zeros(3, dtype=np.float32)
        vn = np.zeros(2, dtype=np.float32)

        idx, n = 0, len(flat)
        hit_trace = []

        while idx < n:
            m, ml = None, 0
            for l_span in range(min(self.mwe_max_len, n - idx), 0, -1):
                p = " ".join(flat[idx : idx + l_span])
                if p in self.lexicon_map:
                    m, ml = self.lexicon_map[p], l_span
                    break

            if not m:
                idx += 1
                continue

            if debug:
                hit_trace.append(
                    {"phrase": " ".join(flat[idx : idx + ml]), "data": m[0]}
                )

            for ent in m:
                if ent["isnegation"]:
                    vn[0] = 1.0
                    continue

                sc = ent["score"]
                if vn[0] > 0:
                    sc *= self.negation_attenuation

                current_emos = ent["emotions"]
                for e, w in current_emos.items():
                    if e in self.ALLEMOTIONS:
                        ve[self.ALLEMOTIONS.index(e)] += sc * w

                # Polarity score injection
                pos_score = sum(
                    w for e, w in current_emos.items() if e in self.POS_EMOTIONS
                )
                if pos_score > 0:
                    vp[0] += sc * pos_score

                neg_score = sum(
                    w for e, w in current_emos.items() if e in self.NEG_EMOTIONS
                )
                if neg_score > 0:
                    vp[1] += sc * neg_score

                neu_score = current_emos.get("Neutral", 0.0)
                if neu_score > 0:
                    vp[2] += sc * neu_score

                for x in ent["polarities"]:
                    if x in self.POLARITY_CLASSES:
                        vp[self.POLARITY_CLASSES.index(x)] += sc

                for x in ent["appraisals"]:
                    if x in self.ALLAPPRAISALS:
                        va[self.ALLAPPRAISALS.index(x)] += sc

                for x in ent["intensities"]:
                    if x in self.INTENSITY_CLASSES:
                        vi[self.INTENSITY_CLASSES.index(x)] = 1.0

            idx += ml

        if self.alpha_similarity > 0:
            ve = np.clip(
                ve + self.alpha_similarity * (ve @ self.sim_matrix), 0, None
            )

        vp *= 1.5
        fin = np.concatenate(
            [np.tanh(ve), np.tanh(va), vi, np.tanh(vp), vn]
        ).astype(np.float32)
        return (fin, flat, hit_trace) if debug else fin
