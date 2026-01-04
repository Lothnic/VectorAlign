# VectorAlign

Bilingual word alignment using multilingual embeddings — no training required!

For non-nerds: a word matching engine for any language pair using parallel data.

## Installation

```bash
pip install vectoralign
```

## Quick Start

```python
from vectoralign import align

# Your parallel sentences
english = [
    "Hello world",
    "How are you today",
    "The weather is nice"
]
hindi = [
    "नमस्ते दुनिया",
    "आज आप कैसे हैं",
    "मौसम अच्छा है"
]

# Align and build dictionary
dictionary = align(english, hindi)
```

## Features

- **No training required** — Uses pre-trained multilingual embeddings
- **Batch processing** — Batch processing with automatic CUDA detection
- **Multiple models** — Supports LaBSE, mBERT, and other HuggingFace models
- **Bidirectional alignment** — Intersection of forward and backward alignments

## Advanced Usage

```python
from vectoralign import align

# Custom model and batch size
dictionary = align(
    src_sentences,
    tgt_sentences,
    model_name="setu4993/LaBSE",  # or "bert" for mBERT
    batch_size=64,
    threshold=0.6,  # Sentence similarity threshold
    output="my_dictionary.txt"
)
```

## Supported Models

| Model | Name |
|-------|------|
| LaBSE | `setu4993/LaBSE` (default) |
| mBERT | `bert` with `mode='multilingual'` |
| Any HuggingFace model with `pooler_output` | Full model path |

## Output Format

The dictionary is saved as a TSV file:
```
word1    translation1    count
word2    translation2    count
```

## Acknowledgments

This is a spiritual implementation of [SimAlign](https://github.com/cisnlp/simalign) by the Centre for Language and Information Processing, LMU Munich.

- [Paper](https://arxiv.org/pdf/2004.08728)

## License

MIT License
