"""
VectorAlign - Bilingual Word Alignment using Multilingual Embeddings

A streamlined implementation of SimAlign by CIS, LMU Munich.
"""

from .align import align, get_embeddings_batch
from .projector import ContrastiveProjector, train_projector, load_projector, info_nce_loss
from .dictionary_refinement import refine_dictionary_with_projector, compare_dictionaries

__version__ = "0.3.0"
__all__ = [
    "align",
    "get_embeddings_batch",
    "ContrastiveProjector",
    "train_projector",
    "load_projector",
    "info_nce_loss",
    "refine_dictionary_with_projector",
    "compare_dictionaries",
    "__version__",
]
