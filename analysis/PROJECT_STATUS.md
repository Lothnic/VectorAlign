# VectorAlign: Project Status & Next Steps

> Generated: 2025-05-09

## What We Are Doing

VectorAlign builds bilingual dictionaries (Hindi-Kangri, Hindi-Kinnauri) by aligning word-level embeddings from parallel sentences. The current pipeline:

1. **Embed**: Encode source/target sentences with LaBSE (a multilingual sentence embedding model).
2. **Align**: Compute cosine similarity between token embeddings, then find mutual best matches (bidirectional argmax intersection).
3. **Aggregate**: Count co-occurrences of aligned pairs across all sentence batches.
4. **Output**: Write a frequency-weighted dictionary TSV.

This works out of the box but has a fundamental limitation: **LaBSE embeddings from different languages are not directly comparable**. Words with the same meaning live in different regions of the shared embedding space. The contrastive projector is meant to fix this by learning a transformation that aligns the spaces.

---

## What Is Already Built

| Component | File | Status |
|---|---|---|
| Baseline alignment (LaBSE + Bi-Argmax) | `vectoralign/align.py` | Working |
| `ContrastiveProjector` class (2 linear layers, normalized) | `vectoralign/projector.py:16` | Written |
| `train_projector()` (InfoNCE training loop) | `vectoralign/projector.py:97` | Written |
| `align_with_projector()` (alignment using projected space) | `vectoralign/dictionary_refinement.py:20` | Written |
| `load_projector()` (load trained weights) | `vectoralign/projector.py:241` | Written |

**All the building blocks exist. None of them are wired together.**

---

## What Is NOT Working (The Gaps)

### Gap 1: `align()` never uses the projector

`vectoralign/align.py` does raw LaBSE alignment only. The projector code sits in separate files with no connection.

### Gap 2: No orchestration script

There is no single entry point that:
1. Loads seed pairs
2. Trains the projector
3. Runs alignment in projected space
4. Saves the result

### Gap 3: No projector integration in `align.py`

The `align()` function needs a `use_projector` flag and logic to load weights and apply them before computing similarity.

---

## What Needs to Be Done

### Step 1: Wire the projector into align.py (1-2 hours)

In `align.py`, add:

```python
def align(..., use_projector: bool = False, projector_path: str = None, **kwargs):
    ...
    # After getting embeddings (line ~186):
    if use_projector:
        projector = load_projector(projector_path or "checkpoints/contrastive_projector.pth")
        src_emb, tgt_emb = projector(src_emb, tgt_emb)  # project into aligned space
```

Then update `align()` calls in `__init__.py` to pass `use_projector=True`.

### Step 2: Create a pipeline runner (1 hour)

Create `run_pipeline.py`:

```python
# Pseudocode flow:
seed_pairs = load_pairs("data/parallel/seed_pairs.txt")
projector = train_projector(seed_pairs)  # InfoNCE training
torch.save(projector.state_dict(), "checkpoints/contrastive_projector.pth")
dict_hk = align(..., use_projector=True, projector_path=...)  # Hindi-Kangri
dict_hk2 = align(..., use_projector=True, projector_path=...)  # Hindi-Kinnauri
```

### Step 3: Compare baseline vs projector (2-3 hours)

Run both variants and compare:
- `align(..., use_projector=False)` -- current output
- `align(..., use_projector=True)` -- improved output

Metrics: dictionary coverage, overlap with existing output, precision on known seed pairs.

### Step 4: (Future) Iterative refinement

Use the projector-aligned dictionary to bootstrap alignments on monolingual data, retrain with expanded pairs, repeat. This is the high-value Phase 2 from the contrastive projector analysis.

---

## Quick Start After Implementation

```bash
# Train projector on seed pairs
python run_pipeline.py --train-only

# Run alignment with projector
python run_pipeline.py --align

# Run both end-to-end
python run_pipeline.py --all
```

---

## Current Files Summary

```
vectoralign/
  align.py               # Baseline only (raw LaBSE)
  projector.py           # ContrastiveProjector + train/load (unused)
  dictionary_refinement.py # align_with_projector (unused)
  __init__.py            # Exposes align() -- needs projector flag

analysis/
  contrastive_projector_analysis.md   # Background theory (23KB, detailed)
  PROJECT_STATUS.md                      # This file
```
