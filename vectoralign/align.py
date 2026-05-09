"""
Core alignment functions for VectorAlign.

Portuguese notes (as written by the author during development):
- 'alinhamento' = alignment
- 'palavra' = word
- 'embedding' stays 'embedding'
- 'frase' = sentence, 'sentença' also = sentence
- 'modelo' = model
- 'dicionário' = dictionary
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import gc
import numpy as np
from tqdm import tqdm
import string
from .projector import load_projector


def get_embeddings_batch(src_sentences: list[str], tgt_sentences: list[str], tokenizer, model, batch_size: int = 32, device: str = 'cpu') -> tuple:
    """
    Get sentence and word embeddings for source and target sentences in batches.

    Args:
        src_sentences: List of source language sentences
        tgt_sentences: List of target language sentences
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        batch_size: Batch size for processing (default: 32)
        device: Device to use ('cuda' or 'cpu') (default: 'cpu')

    Returns:
        Tuple of (src_sentence_emb, tgt_sentence_emb, src_token_emb, tgt_token_emb)
        Each word embedding list contains individual [1, seq_len, hidden] tensors.
    """
    src_s_emb = []
    tgt_s_emb = []
    src_t_emb = []
    tgt_t_emb = []
    
    length = min(len(src_sentences), len(tgt_sentences))
    with torch.no_grad():
        for i in range(0, length, batch_size):
            batch_src = src_sentences[i:i + batch_size]
            batch_tgt = tgt_sentences[i:i + batch_size]

            src_tokens = tokenizer(batch_src, return_tensors='pt', padding=True, truncation=True).to(device)
            tgt_tokens = tokenizer(batch_tgt, return_tensors='pt', padding=True, truncation=True).to(device)

            src_out = model(**src_tokens)
            tgt_out = model(**tgt_tokens)

            # pooler_output -> sentence-level embeddings
            src_s_emb.append(src_out.pooler_output.detach().cpu())
            tgt_s_emb.append(tgt_out.pooler_output.detach().cpu())
            # last_hidden_state -> token-level embeddings
            src_t_emb.append(src_out.last_hidden_state.detach().cpu())
            tgt_t_emb.append(tgt_out.last_hidden_state.detach().cpu())

    # Concatenate the per-batch tensors into single tensors
    src_s_emb = torch.cat(src_s_emb, dim=0)
    tgt_s_emb = torch.cat(tgt_s_emb, dim=0)

    # Wrap each sentence embedding in [1, hidden] so caller indexing is uniform
    src_s_emb = [e.unsqueeze(0) for e in src_s_emb]
    tgt_s_emb = [e.unsqueeze(0) for e in tgt_s_emb]

    # Split word-embedding batches per sentence (seq_len varies across batches)
    src_t_flat = []
    tgt_t_flat = []
    
    for batch in src_t_emb:
        for j in range(batch.shape[0]):
            src_t_flat.append(batch[j:j + 1])  # [1, seq_len, hidden]

    for batch in tgt_t_emb:
        for j in range(batch.shape[0]):
            tgt_t_flat.append(batch[j:j + 1])

    return src_s_emb, tgt_s_emb, src_t_flat, tgt_t_flat


def _compute_sim_matrix(src_tokens, tgt_tokens, src_sent_emb, tgt_sent_emb, threshold=0.5):
    """Compute token-level similarity matrix and filter by sentence similarity."""
    sent_sim = F.cosine_similarity(src_sent_emb, tgt_sent_emb, dim=-1).item()
    
    if sent_sim < threshold:
        return None

    src_norm = F.normalize(src_tokens.squeeze(0), dim=-1)
    tgt_norm = F.normalize(tgt_tokens.squeeze(0), dim=-1)
    
    # Normalized vectors -> matmul gives cosine similarity
    sim_matrix = torch.matmul(src_norm, tgt_norm.T)
    return sim_matrix


def _bidir_argmax(sim_matrix):
    """Get bidirectional alignment by intersecting forward and backward argmax."""
    forward = []
    for i in range(sim_matrix.shape[0]):
        best = int(torch.argmax(sim_matrix[i]))
        forward.append((i, best))

    backward = []
    for j in range(sim_matrix.shape[1]):
        best = int(torch.argmax(sim_matrix[:, j]))
        backward.append((best, j))

    return list(set(forward) & set(backward))


def _indices_to_tokens(alignments, src_tokens_list, tgt_tokens_list, tokenizer, offset=0):
    """Convert alignment indices back to surface tokens, skipping [CLS]/[SEP]."""
    aligned = []
    for src_idx, tgt_idx in alignments:
        # Guard against PAD tokens that appear in the tensor but not in the real sentence
        if src_idx >= len(src_tokens_list) or tgt_idx >= len(tgt_tokens_list):
            continue
        if src_idx == 0 or src_idx == len(src_tokens_list) - 1:
            continue
        if tgt_idx == 0 or tgt_idx == len(tgt_tokens_list) - 1:
            continue
        aligned.append((src_tokens_list[src_idx], tgt_tokens_list[tgt_idx]))
    return aligned


def _merge_subwords(pairs):
    """Merge WordPiece subwords (tokens starting with '##') into complete words."""
    merged = []
    cur_src = ""
    cur_tgt = ""
    
    for src_tok, tgt_tok in pairs:
        if src_tok.startswith("##"):
            cur_src += src_tok[2:]
        else:
            if cur_src:
                merged.append((cur_src, cur_tgt))
            cur_src = src_tok
            cur_tgt = tgt_tok
    
    if cur_src:
        merged.append((cur_src, cur_tgt))
    
    return merged


def _build_freq_dict(all_pairs):
    """Build a frequency dictionary from a list of (src, tgt) word pairs."""
    from collections import defaultdict
    freq = defaultdict(int)
    for src, tgt in all_pairs:
        src_clean = src.lower().strip(string.punctuation)
        tgt_clean = tgt.strip()
        if src_clean and tgt_clean:
            freq[(src_clean, tgt_clean)] += 1
    return dict(freq)


def _save_dict(freq_dict, path):
    """Save frequency dictionary as TSV to disk."""
    import os
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write("src\ttgt\tfreq\n")
        for (src, tgt), count in sorted(freq_dict.items(), key=lambda x: x[1], reverse=True):
            fh.write(f"{src}\t{tgt}\t{count}\n")
    
    print(f"Dictionary saved to {path}, {len(freq_dict)} unique pairs.")


def align(
    src_sentences: list[str],
    tgt_sentences: list[str],
    batch_size: int = 32,
    model_name: str = "setu4993/LaBSE",
    device: str = "auto",
    output: str = "output/dict.txt",
    threshold: float = 0.5,
    use_projector: bool = False,
    projector_path: str = None,
):
    """
    Align words between source and target sentences and build a bilingual dictionary.

    Args:
        src_sentences: List of source-language sentences.
        tgt_sentences: List of target-language sentences (same length as src).
        batch_size: Batch size for embedding computation (default 32).
        model_name: HuggingFace model id or path. (default "setu4993/LaBSE")
        device: Device to run on – 'auto' detects CUDA, 'cuda', or 'cpu'.
        output: Path where the resulting TSV dictionary is written.
        threshold: Minimum sentence-level cosine similarity to keep a sentence pair.
        use_projector: If True, project embeddings through a trained ContrastiveProjector
            before computing similarity. This aligns the cross-lingual embedding spaces.
        projector_path: Path to a trained projector checkpoint (.pth/.pt). Required when
            use_projector=True. Defaults to 'checkpoints/contrastive_projector.pth'.

    Returns:
        dict: { (src_word, tgt_word): count } mapping all aligned word pairs.
    """
    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model (always in eval mode, no gradients)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    print(f"Model loaded: {model_name}")

    # Load projector if requested
    projector = None
    if use_projector:
        if projector_path is None:
            projector_path = "checkpoints/contrastive_projector.pth"
        projector = load_projector(projector_path, device=device)
        print(f"Projector loaded from {projector_path}")

    n = min(len(src_sentences), len(tgt_sentences))
    all_pairs = []

    # Process one batch of sentences at a time
    for i in tqdm(range(0, n, batch_size), desc="Aligning"):
        # 1) embeddings
        s_sent, t_sent, s_tok, t_tok = get_embeddings_batch(
            src_sentences[i:i + batch_size],
            tgt_sentences[i:i + batch_size],
            tokenizer, model, batch_size, device,
        )

        batch_count = len(s_sent)
        for j in range(batch_count):
            # Optionally project embeddings into aligned cross-lingual space
            s_tok_j = s_tok[j]
            t_tok_j = t_tok[j]
            s_sent_j = s_sent[j]
            t_sent_j = t_sent[j]

            if projector is not None:
                with torch.no_grad():
                    s_tok_j = projector.project_tokens(
                        s_tok_j.to(device), side="src"
                    ).cpu()
                    t_tok_j = projector.project_tokens(
                        t_tok_j.to(device), side="tgt"
                    ).cpu()
                    s_sent_j, t_sent_j = projector.align_embeddings(
                        s_sent_j.to(device), t_sent_j.to(device)
                    )
                    s_sent_j = s_sent_j.cpu()
                    t_sent_j = t_sent_j.cpu()

            # 2) skip if sentence similarity too low
            matrix = _compute_sim_matrix(
                s_tok_j, t_tok_j, s_sent_j, t_sent_j, threshold,
            )
            if matrix is None:
                continue

            # 3) bidirectional argmax intersection -> token indices
            aln = _bidir_argmax(matrix)

            # 4) convert indices to actual tokens
            src_raw = ["[CLS]"] + tokenizer.tokenize(src_sentences[i + j]) + ["[SEP]"]
            tgt_raw = ["[CLS]"] + tokenizer.tokenize(tgt_sentences[i + j]) + ["[SEP]"]
            pairs = _indices_to_tokens(aln, src_raw, tgt_raw, tokenizer)

            # 5) merge subwords --> surface-level (src, tgt) pairs
            merged = _merge_subwords(pairs)
            all_pairs.extend(merged)

        # Clean up per-batch to avoid memory bloat
        del s_sent, t_sent, s_tok, t_tok
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Build and persist dictionary
    freq = _build_freq_dict(all_pairs)
    _save_dict(freq, output)
    return freq
