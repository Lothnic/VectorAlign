# VectorAlign Codebase Analysis & Contrastive Projector Readiness Assessment

## 1. REPOSITORY STRUCTURE

```
VectorAlign/
  pyproject.toml                    # Project config (v0.2.2)
  README.md                         # Docs
  build_dicts.py                    # Top-level script: orchestrates Hindi-Kangri + Hindi-Kinnauri alignment
  data.md                           # Short note pointing to two external corpus repos
  data/
    english.txt                     # 3,440 English sentences (EN-HI parallel)
    hindi.txt                       # 3,440 Hindi sentences (EN-HI parallel)
    parallel/
      kr_hi_src.txt                 # 26,785 Kangri sentences (source for Hindi)
      kr_hi_tgt.txt                 # 26,785 Hindi sentences (target / ground truth)
      kp_hi_src.txt                 # 20,307 Kinnauri-Pahari sentences (source)
      kp_hi_tgt.txt                 # 20,307 Hindi sentences (target)
      seed_pairs.txt                # 2,277 bilingual seed word pairs (Hindi <--> Kangri)
      kangri_src.snt                # Kangri parallel sentences (used by build_dicts.py)
      kangri_tgt.snt                # Hindi parallel sentences
      kinnauri_src.snt              # Kinnauri parallel sentences
      kinnauri_tgt.snt              # Hindi parallel sentences
      corpus_meta.txt               # Stats: 26785 kangri, 20307 kinnauri, 2277 seed pairs
    raw/
      Kangri_corpus/                # Raw Kangri data from https://github.com/chauhanshweta/Kangri_corpus
        train dataset/Kr_1.txt      # Monolingual Kangri (books, stories, novels)
        train dataset/Kr_2.txt      # Hindi-Kangri dictionary words
        train dataset/Kr_3.txt      # Kavitaiyein, Lok-Geet, Kangri Gazals
        train dataset/Kr_4_kangri.txt  # Parallel Kangri
        train dataset/Kr_4_Hindi.txt   # Parallel Hindi
        test dataset/               # Test split of Kr_4
      Kinnauri-Pahari/              # Raw Kinnauri data from https://github.com/phildani7/dlnith
        Parallel/Parallel_data_HP.txt
        Parallel/Parallel_data_Hi.txt
        readme.md
        requirement.txt
  output/
    hindi_kangri_dict.tsv           # 15,056 aligned Hindi-Kangri pairs (from current align run)
    hindi_kinnauri_dict.tsv         # 7,750 aligned Hindi-Kinnauri pairs
    bilingual_dict.txt              # Some merged bilingual dictionary
    dict.txt                        # Another dictionary output
  vectoralign/                      # Package directory
    __init__.py                     # Package init
    align.py                        # Core alignment engine (255 lines)
  vectoralign.egg-info/             # Package metadata
```

---

## 2. CURRENT ALIGNMENT PIPELINE (align.py)

### 2.1 Model: LaBSE ("setu4993/LaBSE")

- **Architecture**: LaBSE (Language-Bank Sentence Embeddings) by Google Research.
- **Base**: mBERT-like architecture with 12 layers, 768-dim hidden size.
- **Training**: Self-supervised bilingual sentence-level objective on 100+ language pairs using parallel corpora (not XLM-RoBERTa-based).
- **Output**: `pooler_output` (sentence-level embeddings) and `last_hidden_state` (token-level/word-level embeddings).
- **Tokenizer**: `AutoTokenizer.from_pretrained("setu4993/LaBSE")` -- WordPiece-based, handles both Hindi (Devanagari) and Kangri/Kinnauri (also Devanagari script) natively.

### 2.2 Pipeline Flow

```
Input: [src_sentences], [tgt_sentences] (parallel, same length)
      |
      v
[1] get_embeddings_batch()     <-- Bilingual sentence-level + token-level embedding extraction
      - Batched inference on mBERT tokenizer + LaBSE model
      - Returns: src_s_emb, tgt_s_emb (sentence-level), src_t_emb, tgt_t_emb (token-level)
      |
      v
[2] _compute_sim_matrix()      <-- Cosine similarity between token embeddings
      - Normalizes token embeddings, computes matmul -> cosine similarity matrix (len(src_tokens) x len(tgt_tokens))
      - Filters out sentence-pairs with sentence-level cosine similarity < threshold (default 0.5)
      |
      v
[3] _bidir_argmax()            <-- Bidirectional intersection alignment
      - Forward: for each src token, take argmax over tgt similarity scores
      - Backward: for each tgt token, take argmax over src similarity scores
      - Intersection: keep only pairs (i,j) that appear in BOTH directions
      |
      v
[4] _indices_to_tokens()       <-- Map aligned indices back to surface tokens
      - Skips [CLS]/[SEP] tokens at positions 0 and -1
      - Guards against PAD tokens that exceed real sentence length
      |
      v
[5] _merge_subwords()          <-- Reconstruct WordPiece subword tokens into surface words
      - Concatenates '##suffix' subword pieces to previous base token
      - Produces (src_word, tgt_word) pairs at surface level
      |
      v
[6] _build_freq_dict()         <-- Aggregates all aligned pairs across all sentence batches
      - defaultdict(int) counting co-occurrences
      - Lowercases Hindi words, preserves Kangri/Kinnauri orthography
      - Strips punctuation (string.punctuation set)
      |
      v
[7] _save_dict()               <-- Writes TSV: src, tgt, freq
      |
      v
Output: bilingual dictionary TSV file
```

### 2.3 Key Implementation Details

**Embedding extraction** (get_embeddings_batch):
- Processes in configurable batch_size (default 32)
- Returns sentence embeddings with pooler_output (dim: [batch, 768])
- Returns token embeddings as list of [1, seq_len, 768] tensors per sentence

**Similarity filtering**:
- Sentence-level cosine similarity threshold (default 0.5) acts as a coarse filter
- Only sentence pairs above threshold get token-level alignment
- This is a simple heuristic; does not use cross-lingual projection

**Bidirectional argmax**:
```python
# Pure dot-product / cosine similarity on normalized embeddings
forward[(i, j)] = if j == argmax_k(S[i, k])   # best tgt for each src
backward[(i, j)] = if i == argmax_k(S[k, j])   # best src for each tgt
alignment = forward intersect backward           # only mutual bests remain
```

**Subword merging**:
- Handles WordPiece convention: '##piece' is a continuation
- Does NOT handle SentencePiece BPE or Unigram subword formats
- The current LaBSE tokenizer uses WordPiece, which is correct

**Frequency weighting**:
- Counts each aligned pair across all sentence batches
- Higher frequency = likely more reliable alignment
- This is the only form of "cross-sentence consistency" currently used

---

## 3. DATA AVAILABILITY

### 3.1 Kangri-Hindi Parallel Corpus

| Dataset                          | Format     | Count     | Source                              |
|----------------------------------|------------|-----------|-------------------------------------|
| kr_hi_src.txt (Kangri)           | .txt       | 26,785    | data/parallel/                     |
| kr_hi_tgt.txt (Hindi)            | .txt       | 26,785    | data/parallel/                     |
| kangri_src.snt (Kangri)          | .snt       | ~26,785   | data/parallel/                     |
| kangri_tgt.snt (Hindi)           | .snt       | ~26,785   | data/parallel/                     |
| Kr_4_kangri.txt + Kr_4_Hindi.txt| parallel   | ~26,862   | train dataset from Kangri_corpus   |
| Kr_2.txt                         | mono/dict  | dictionary| train dataset (Hindi-Kangri words) |
| Kr_1.txt                         | monolingual| 3,376     | books/stories/novels               |
| Kr_3.txt                         | monolingual| poems/ghazals  | Kavitaiyein, Lok-Geet           |
| Hindi-Kangri seed pairs          | .tsv       | 2,277     | data/parallel/seed_pairs.txt       |

**Total useful Kangri-Hindi data**: ~26,785 parallel sentences + 2,277 seed pairs + monolingual Kangri (1.81M tokens from external source).

### 3.2 Kinnauri-Hindi Parallel Corpus

| Dataset                          | Format     | Count     | Source                              |
|----------------------------------|------------|-----------|-------------------------------------|
| kp_hi_src.txt (Kinnauri-Pahari) | .txt       | 20,307    | data/parallel/                     |
| kp_hi_tgt.txt (Hindi)            | .txt       | 20,307    | data/parallel/                     |
| kinnauri_src.snt (Kinnauri)      | .snt       | ~20,307   | data/parallel/                     |
| kinnauri_tgt.snt (Hindi)         | .snt       | ~20,307   | data/parallel/                      |
| Parallel_data_KP.txt             | .txt       | ~20,307   | raw/Kinnauri-Pahari/Parallel/      |
| Parallel_data_Hi.txt             | .txt       | ~20,307   | raw/Kinnauri-Pahari/Parallel/      |
| Monolingual Kinnauri             | external   | 43,367    | from Kinnauri-Pahari repo          |

**Total useful Kinnauri-Hindi data**: ~20,307 parallel sentences + external monolingual (43,367 sentences).

### 3.3 En-HI Test/Reference Corpus

| Dataset     | Count    | Purpose                          |
|-------------|----------|----------------------------------|
| english.txt | 3,440    | EN-HI parallel test/sanity check |
| hindi.txt   | 3,440    | EN-HI parallel test/sanity check |

### 3.4 Existing Dictionary Outputs

| File                        | Pairs | Unique Hindi | Notes                          |
|-----------------------------|-------|--------------|--------------------------------|
| output/hindi_kangri_dict.tsv | 15,056 | ~15,056     | From current align run         |
| output/hindi_kinnauri_dict.tsv| 7,750  | ~7,750      | From current align run         |

---

## 4. CURRENT STATE: WHAT IS AND IS NOT IMPLEMENTED

### 4.1 What IS Implemented

Feature                                    | Status
-------------------------------------------|--------
LaBSE embedding extraction (token + sentence) | Done
WordPiece subword merging                   | Done
Bidirectional argmax intersection           | Done
Sentence-level cosine similarity filtering  | Done
Frequency-based pair aggregation            | Done
Batch processing with memory management     | Done
CUDA auto-detection                         | Done
Configurable threshold and batch size       | Done
TSV dictionary output                       | Done
Hindi-Kangri pipeline                       | Done
Hindi-Kinnauri pipeline                     | Done
English-Hindi test corpus                   | Available

### 4.2 What IS NOT Implemented (Contrastive Projector Gaps)

Feature                                        | Status
-----------------------------------------------|--------
Phase 1: Cross-lingual projection layer        | MISSING - core gap
Phase 2: Iterative dictionary refinement       | MISSING - core gap
Self-training / pseudo-labeling                 | MISSING
Language-specific projector matrices            | MISSING
Contrastive loss function                       | MISSING
Embedding space alignment / projection          | MISSING
Improved sentence-level filtering via projection | MISSING
Language-aware token normalization              | Partial (Hindi only; Kangri/Kinnauri not normalized)

---

## 5. WHAT "CONTRASTIVE PROJECTOR" MEANS (XLM-R + LLINK CONTEXT)

### 5.1 The Contrastive Projector Concept (from the XLM-RoBERTa paper)

The XLM-RoBERTa paper (Conneau et al., 2020) demonstrated that multilingual embeddings, even though trained in a self-supervised manner (masked token prediction across 100+ languages), are **not directly comparable across languages** in the raw embedding space. Words in different languages that share the same meaning live in different regions of the shared embedding space.

The **contrastive projector** is a learned linear transformation layer (typically a single linear layer: y = W*x + b) trained with contrasting objectives to **align the embeddings of the two languages into a common space**, so that:

- Cross-lingual cosine similarity between true translation pairs is **maximized**
- Cross-lingual cosine similarity between contrastive negatives is **minimized**

The core idea from the XLM-R paper's approach:

```
Phase 1 (Projection):
  For source language L1: proj_L1 = Linear(dim_in, dim_out)
  For target language L2: proj_L2 = Linear(dim_in, dim_out)
  
  projected_hl1 = proj_L1(e_hl1)   # normalize
  projected_hl2 = proj_L2(e_hl2)   # normalize
  
  Loss = InfoNCE / NT-Xent:
    For each batch pair (hl1_i, hl2_i) that is a true parallel pair:
      Positives: sim(projected_hl1_i, projected_hl2_i)
      Negatives: all other pairs in the batch
      Loss = -log(exp(pos / tau) / sum(exp(neg_k / tau)))
  
  Optimize proj_L1, proj_L2 with this loss

Phase 2 (Iterative Dictionary Expansion):
  1. Initialize dictionary D from Phase 1 aligned pairs
  2. Use D to improve alignment:
     - Weight similar word pairs in D more heavily
     - Possibly re-extract embeddings conditioned on dictionary
  3. Repeat until convergence
```

### 5.2 How Contrastive Projection Differs from Current approach

| Aspect                   | Current (LaBSE + Bi-Argmax)        | With Contrastive Projector           |
|--------------------------|-------------------------------------|---------------------------------------|
| Embedding space          | Raw LaBSE (cross-lingually misaligned) | Aligned via learned projector         |
| Similarity computation   | Raw cosine on raw embeddings        | Cosine on projected embeddings        |
| Alignment criterion      | Argmax intersection                 | Projected similarity + threshold       |
| Cross-lingual alignment  | Implicit (relies on mBERT pretraining) | Explicit (learned on seed pairs)      |
| Dictionary quality       | Depends on embedding quality        | Improved via iterative retraining      |
| Data efficiency          | Works with any parallel sentences   | Especially benefits from seed pairs   |

### 5.3 Key Insight

LaBSE already encodes cross-lingual information through its pretraining on parallel corpora, but the embedding space is **still not optimally aligned for word-level alignment**. The seed pairs (2,277 Hindi-Kangri and potentially ~2,277 for Kinnauri-Hindi) provide the signal needed to learn a projection that aligns the two languages' embedding subspaces specifically for the alignment task. This is what both the XLM-R contrastive projector approach and the LLINK paper propose.

---

## 6. SPECIFIC RECOMMENDATIONS FOR IMPLEMENTING FULL CONTRASTIVE PROJECTOR ALIGNMENT

### 6.1 Phase 1: Learn Cross-Lingual Projection Layer

**What to do:**
1. Load LaBSE model and freeze (or fine-tune with low LR)
2. Add a single linear projector layer for each language direction
3. Train with InfoNCE/NTXent loss on the seed_pairs.txt bilingual data
4. Use the Kangri-Hindi parallel corpus sentences as additional positive pairs

**Implementation sketch:**
```python
class ContrastiveProjector(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj_src = nn.Linear(dim, dim)   # language-specific projection
        self.proj_tgt = nn.Linear(dim, dim)   # language-specific projection
        self.T = nn.Parameter(torch.tensor(0.07))  # temperature
    
    def forward(self, emb_src, emb_tgt):
        proj_src = F.normalize(self.proj_src(emb_src), dim=-1)
        proj_tgt = F.normalize(self.proj_tgt(emb_tgt), dim=-1)
        return proj_src, proj_tgt

# Training loop with InfoNCE:
# positives = torch.diag(sim_matrix)  # diagonal elements
# negatives = sim_matrix - diag_mask
# loss = -log(exp(pos / T) / exp(negs / T).sum())
```

**Training data sources:**
- seed_pairs.txt (2,277 Hindi-Kangri word pairs) -- main training signal
- kr_hi_parallel corpus (26,785 sentence pairs) -- sentence-level positives
- Seed pair word embeddings computed from LaBSE
- Augment with synonym pairs, root word pairs where available

**Model choice:**
- Option A: Freeze LaBSE weights entirely, train only projector (cheaper, safer)
- Option B: Fine-tune LaBSE projector with low LR (more accurate but needs regularization)
- Option C: Train two separate projectors (one per language pair direction)

### 6.2 Phase 2: Iterative Dictionary Refinement

**What to do:**
1. Use projected-aligned pairs to bootstrap an initial dictionary
2. Use this dictionary to select high-confidence alignment targets for unlabeled monolingual sentences
3. Re-train projector with expanded dictionary
4. Repeat until convergence (typically 3-5 iterations)

**Implementation sketch:**
```python
def iterative_refinement(projector, src_sents, tgt_sents, seed_pairs, max_iters=5):
    dict_D = build_initial_dict(seed_pairs, projector)  # Phase 1 output
    for iteration in range(max_iters):
        # Use D to extract alignment targets from monolingual data
        expanded_pairs = expand_dictionary(dict_D, monolingual_src, monolingual_tgt)
        
        # Re-train projector with expanded pairs
        dict_D = train_projector(projector, expanded_pairs)
        
        # Evaluate alignment quality
        score = alignment_quality(projector, seed_pairs)
        if score < prev_score * (1 - delta):
            break
        prev_score = score
    return dict_D
```

**Key refinement strategies:**
- Use projected sentence similarity to bootstrap sentence alignment
- Translate monolingual sentences using the current dictionary as a hint
- Re-extract and align with the projector
- Expand the dictionary with newly aligned pairs
- Apply filtering (frequency thresholds, consistency checks)

### 6.3 Specific Code Additions Needed

**New files/modules:**
```
vectoralign/
  __init__.py
  align.py                    # Existing (keep as baseline)
  projector.py                # NEW: ContrastiveProjector class + training
  alignment.py                # NEW: Contrastive alignment using projector
  dictionaries.py              # NEW: Dictionary building/refinement utilities
  data/                       # NEW: Seed pair data loaders
```

**Changes to existing align.py:**
- Add `use_projector: bool = False` parameter
- When True, load and apply ContrastiveProjector before computing similarity
- Store projector weights alongside dictionary output

### 6.4 Kangri-Hindi Specific Recommendations

1. The 2,277 seed_pairs.txt is **sufficient for Phase 1** -- this is the minimum viable set
2. The 26,785 parallel sentences provide **strong sentence-level positive pairs** for training
3. Use the **monolingual Kangri data** (1.81M tokens) in Phase 2 for unsupervised expansion
4. LaBSE handles Kangri (Devanagari script) well because Kangri shares the script with Hindi

### 6.5 Kinnauri-Hindi Specific Recommendations

1. Similar data situation as Kangri-Hindi but with **fewer parallel sentences** (20,307 vs 26,785)
2. Need seed pairs for Kinnauri-Hindi if not already available -- the seed_pairs.txt currently has Kangri pairs only
3. **Generate Kinnauri-Hindi seed pairs** by:
   - Aligning kp_hi parallel corpus with current baseline
   - Extracting top-confidence pairs
   - Using these as pseudo-seed pairs for the projector
4. Use the 43,367 monolingual Kinnauri sentences in Phase 2 expansion

### 6.6 Benchmarking Strategy

To properly evaluate improvements, implement a **three-baseline comparison**:

| Variant                  | Projector | Seed Pairs | Data Used         | Eval Metric       |
|--------------------------|-----------|------------|--------------------|--------------------|
| V1 (baseline)            | None      | None       | Parallel corpus    | Alignment accuracy |
| V2 (Phase 1 only)        | LaBSE +   | Seed pairs | Parallel corpus    | Alignment accuracy |
| V3 (Phase 1 + 2)         | LaBSE +   | Seed +     | Parallel + Mono    | Alignment accuracy |
| V4 (LaBSE + InfoNCE)     | Learned   | Seed pairs | Parallel + Mono    | Alignment accuracy |

**Evaluation metrics:**
- Precision@K (top-k most similar words)
- Exact word translation accuracy (% correctly aligned pairs)
- Dictionary coverage (% of words that have at least one translation)
- Mean reciprocal rank of correct alignment

---

## 7. RISK ASSESSMENT AND POTENTIAL ISSUES

| Risk                          | Likelihood | Impact | Mitigation                        |
|-------------------------------|------------|--------|------------------------------------|
| seed_pairs.txt only Kangri    | High       | Medium | Generate Kinnauri seed pairs via baseline alignment |
| LaBSE tokenizer limitation   | Low        | Low    | Tested and working with Devanagari |
| CUDA memory for full corpus   | Medium     | Low    | Already managed via batched inference |
| Kangri subword fragmentation  | Medium     | Low    | Existing _merge_subwords handles this |
| Cross-lingual vocabulary gap  | High       | High   | Projector addresses this            |
| Low-resource language drift   | Medium     | Medium | Use language-aware projector        |
| Overfitting on seed pairs     | Medium     | Medium | Use parallel corpus augmentation   |
| No labeled test set for eval  | High       | High   | Use baseline alignment as pseudo-ground truth |

---

## 8. EXECUTION PLAN (STEP BY STEP)

```
Phase 1 (4-6 hours of work):
  [ ] 1. Create vectoralign/projector.py with ContrastiveProjector class
  [ ] 2. Implement InfoNCE loss function
  [ ] 3. Implement training loop optimized for seed pairs
  [ ] 4. Integrate projector into align() function
  [ ] 5. Test Phase 1 with seed_pairs.txt

Phase 2 (6-8 hours of work):
  [ ] 6. Implement dictionary expansion from projected similarities
  [ ] 7. Implement iterative refinement loop
  [ ] 8. Add filtering for high-confidence new pairs
  [ ] 9. Test full Phase 1+2 with Kangri-Hindi
  [ ] 10. Test full Phase 1+2 with Kinnauri-Hindi

Benchmarking (2-3 hours of work):
  [ ] 11. Implement evaluation metrics
  [ ] 12. Run V1-V3 variants for both language pairs
  [ ] 13. Compile results and report improvements
```

---

## 9. SUMMARY OF FINDINGS

1. **The codebase is clean and well-structured.** The core align.py (255 lines) is a solid foundation implementing SimAlign-style alignment with LaBSE embeddings.

2. **No contrastive projector is currently implemented.** The alignment relies entirely on raw LaBSE cosine similarity and bidirectional argmax intersection -- this is the main gap.

3. **Data is rich and sufficient.** With 26,785 Kangri-Hindi and 20,307 Kinnauri-Hindi parallel sentences plus 2,277 seed pairs, we have enough data to train a meaningful projector.

4. **The model choice (LaBSE) is good but suboptimal for word-level alignment.** LaBSE is optimized for sentence-level tasks; cross-lingual word embedding alignment via a projector would significantly improve word-level alignment.

5. **The contrastive projector approach is straightforward to implement.** A single linear projection layer per language, trained with InfoNCE on seed pairs, should yield measurable improvements.

6. **Phase 2 (iterative refinement) is the high-value addition.** Using the projector to bootstrap alignments on monolingual data can dramatically increase dictionary coverage, especially for low-frequency words.

7. **The most practical immediate step** is to implement the ContrastiveProjector with training on seed_pairs.txt and evaluate on the existing parallel corpus as pseudo-ground truth.
