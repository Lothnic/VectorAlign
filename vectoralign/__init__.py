"""
VectorAlign - Bilingual Word Alignment using Multilingual Embeddings

A streamlined implementation of SimAlign by CIS, LMU Munich.
"""

from .align import align, get_embeddings_batch
from .cli import cli

__version__ = "0.3.0"
__all__ = ["align", "get_embeddings_batch", "__version__", "cli"]
