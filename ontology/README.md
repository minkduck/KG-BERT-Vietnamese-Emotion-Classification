# Ekman & Cognitive Appraisal Knowledge Graph (OWL/RDF)

This directory contains the formal Ekman & Cognitive Appraisal Knowledge Graph releases used by the `OntologyEngine` vectorizer across the three benchmark datasets.

---

## 1. Archived Ontology Releases

The repository archives two releases of the ontology to ensure complete, bit-exact reproducibility across all experimental setups:

| File Name | Release Target | Total Triples | Named Individuals | OWL Classes | Object Properties | Datatype Properties |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`ekman_appraisal_ontology.rdf`** | **UIT-VSFC** (Revision) | **91,005** | 11,733 | 46 | 14 | 10 |
| **`ekman_appraisal_ontology_v1.rdf`** | **VSFC-Ekman** & **UIT-VSMEC** (Initial Release) | **90,420** | 11,643 | 46 | 14 | 10 |

---

## 2. Dataset Mapping & Usage

- **`ontology/ekman_appraisal_ontology.rdf` (91,005 triples):**  
  Used in the 3-class sentiment analysis experiments on **UIT-VSFC** with the `VSFCOntologyEngine`.
- **`ontology/ekman_appraisal_ontology_v1.rdf` (90,420 triples):**  
  Used in the 7-class emotion classification experiments on **VSFC-Ekman** and **UIT-VSMEC** with the standard `OntologyEngine` (corresponding to Table 2 of the paper).

---

## 3. English Ontology-v2 Release

| File Name | Release Target | Total Triples | Notes |
| :--- | :--- | :---: | :--- |
| **`ekman_appraisal_ontology_v2.rdf`** | **DailyDialog & MELD English confirmatory study** | **92,150** | Adds the ontology-v2 appraisal layer used by the NRC-derived English lexicalisation. |

The English build maps a locally licensed NRC Emotion Lexicon file to the ontology-v2 schema. The NRC source file and generated lexical lookup tables are intentionally not redistributed in this public package. Use `scripts/build_english_lexicon.py` to rebuild them locally.
