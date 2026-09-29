"""
Core modules for Ontology-Guided Knowledge Injection in Vietnamese NLP.
"""

from src.utils import seed_everything, load_config
from src.ontology_engine import OntologyEngine, VSFCOntologyEngine
from src.significance_test import paired_permutation_test, macro_f1
from src.data_loading import load_vsfc, load_vsfc_ekman, load_vsmec

__all__ = [
    "seed_everything",
    "load_config",
    "OntologyEngine",
    "VSFCOntologyEngine",
    "paired_permutation_test",
    "macro_f1",
    "load_vsfc",
    "load_vsfc_ekman",
    "load_vsmec",
]
