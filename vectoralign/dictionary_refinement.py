"""
Dictionary refinement using contrastive projector alignment.

This module provides functions to:
1. Re-align sentences using the projected embedding space
2. Refine dictionaries by re-computing alignments with projector
3. Export dictionaries in the same format as baseline
"""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import string
from collections import defaultdict
import os



def align_with_projector(
    src_sentences: list[str],
    tgt_sentences: list[str],
    projector,
    tokenizer,
    model,
    batch_size: int = 32,
    threshold: float = 0.4,
    device: str = "cpu",
) -> list[tuple[str, str]]:
    """
    Align sentences using the contrastive projector instead of raw cosine similarity.
    
    Args:
        src_sentences: Source language sentences
        tgt_sentences: Target language sentences
        projector: Trained ContrastiveProjector
        tokenizer: Tokenizer
        model: Embedding model
        batch_size: Batch size for processing
        threshold: Minimum sentence similarity threshold
        device: Device for computation
        
    Returns:
        List of aligned (src_word, tgt_word) pairs
    """
    model = model.to(device)
    projector = projector.to(device)
    model.eval()
    projector.eval()
    
    n = min(len(src_sentences), len(tgt_sentences))
    all_pairs = []
    
    src_embs_batch = []
    tgt_embs_batch = []
    src_proj_batch = []
    tgt_proj_batch = []
    
    for i in range(0, n, batch_size):
        batch_src = src_sentences[i:i + batch_size]
        batch_tgt = tgt_sentences[i:i + batch_size]
        batch_len = len(batch_src)
        
        with torch.no_grad():
            src_tokens = tokenizer(batch_src, return_tensors="pt", padding=True, truncation=True).to(device)
            tgt_tokens = tokenizer(batch_tgt, return_tensors="pt", padding=True, truncation=True).to(device)
            
            src_out = model(**src_tokens)
            tgt_out = model(**tgt_tokens)
            
            src_s_emb = src_out.pooler_output.detach().cpu()
            tgt_s_emb = tgt_out.pooler_output.detach().cpu()
            src_t_emb = src_out.last_hidden_state.detach().cpu()
            tgt_t_emb = tgt_out.last_hidden_state.detach().cpu()
            
            # Project embeddings
            proj_src, proj_tgt = projector(src_s_emb, tgt_s_emb)
            
            # Re-compute similarity matrix with projected embeddings
            src_proj_batch = proj_src
            tgt_proj_batch = proj_tgt
            
            for j in range(batch_len):
                # Check sentence similarity in projected space
                sent_sim = F.cosine_similarity(proj_src[j:j+1], proj_tgt[j:j+1], dim=-1).item()
                
                if sent_sim < threshold:
                    continue
                
                # Compute token-level similarity in projected space
                src_norm = F.normalize(src_t_emb[j].squeeze(0), dim=-1)
                tgt_norm = F.normalize(tgt_t_emb[j].squeeze(0), dim=-1)
                
                sim_matrix = torch.matmul(src_norm, tgt_norm.T)
                
                # Bidirectional argmax
                forward = []
                for k in range(sim_matrix.shape[0]):
                    best = int(torch.argmax(sim_matrix[k]))
                    forward.append((k, best))
                
                backward = []
                for k in range(sim_matrix.shape[1]):
                    best = int(torch.argmax(sim_matrix[:, k]))
                    backward.append((best, k))
                
                alignments = list(set(forward) & set(backward))
                
                # Reconstruct token lists (without PAD that might be added by tokenizer)
                # We use tokenizer to get the actual token count
                src_raw = tokenizer.tokenize(batch_src[j])
                tgt_raw = tokenizer.tokenize(batch_tgt[j])
                src_raw = ["[CLS]"] + src_raw + ["[SEP]"]
                tgt_raw = ["[CLS]"] + tgt_raw + ["[SEP]"]
                
                # Convert alignments to tokens
                aligned = []
                for src_idx, tgt_idx in alignments:
                    if src_idx >= len(src_raw) or src_idx == 0:
                        continue
                    if tgt_idx >= len(tgt_raw) or tgt_idx == 0:
                        continue
                    aligned.append((src_raw[src_idx], tgt_raw[tgt_idx]))
                
                # Merge subwords
                merged = _merge_subwords(aligned)
                all_pairs.extend(merged)
    
    projector.cpu()
    model.cpu()
    
    return all_pairs


def _merge_subwords(pairs):
    """Merge WordPiece subwords (tokens starting with '##') into complete words."""
    merged = []
    cur_src = ""
    cur_tgt = ""
    
    for src_tok, tgt_tok in pairs:
        if src_tok.startswith("##"):
            cur_src += src_tok[2:]
            cur_tgt += tgt_tok[2:]
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
    freq = defaultdict(int)
    for src, tgt in all_pairs:
        src_clean = src.lower().strip(string.punctuation)
        tgt_clean = tgt.strip()
        if src_clean and tgt_clean:
            freq[(src_clean, tgt_clean)] += 1
    return dict(freq)


def _save_dict(freq_dict, path):
    """Save frequency dictionary as TSV to disk."""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write("src\ttgt\tfreq\n")
        for (src, tgt), count in sorted(freq_dict.items(), key=lambda x: x[1], reverse=True):
            fh.write(f"{src}\t{tgt}\t{count}\n")
    
    print(f"Dictionary saved to {path}, {len(freq_dict)} unique pairs.")


def refine_dictionary_with_projector(
    src_sentences: list[str],
    tgt_sentences: list[str],
    projector,
    tokenizer,
    model,
    batch_size: int = 32,
    threshold: float = 0.4,
    base_threshold: float = 0.5,
    device: str = "cpu",
    output_path: str = None,
) -> dict:
    """
    Align and build dictionary using contrastive projector.
    
    Args:
        src_sentences: Source language sentences
        tgt_sentences: Target language sentences
        projector: Trained ContrastiveProjector
        tokenizer: Tokenizer
        model: Embedding model
        batch_size: Batch size
        threshold: Sentence similarity threshold in projected space
        base_threshold: Fallback to raw threshold if needed
        device: Device
        output_path: Path to save dictionary
        
    Returns:
        Frequency dictionary
    """
    
    all_pairs = align_with_projector(
        src_sentences, tgt_sentences, projector, tokenizer, model,
        batch_size=batch_size, threshold=threshold, device=device
    )
    
    freq = _build_freq_dict(all_pairs)
    print(f"Alignments with projector: {len(all_pairs)} total pairs")
    print(f"Dictionary: {len(freq)} unique pairs")
    
    if output_path:
        _save_dict(freq, output_path)
    
    return freq


def compare_dictionaries(dict1_path, dict2_path):
    """Compare two dictionaries: baseline vs. projector."""
    
    def load_dict(path):
        freq = defaultdict(int)
        with open(path, 'r', encoding='utf-8') as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    freq[(parts[0], parts[1])] = int(parts[2])
        return freq
    
    d1 = load_dict(dict1_path)
    d2 = load_dict(dict2_path)
    
    print(f"Baseline ({dict1_path}): {len(d1)} unique pairs, {sum(d1.values())} total pairs")
    print(f"Projector ({dict2_path}): {len(d2)} unique pairs, {sum(d2.values())} total pairs")
    
    # Overlap
    overlap = set(d1.keys()) & set(d2.keys())
    print(f"Overlap: {len(overlap)} pairs ({len(overlap)/max(len(d1), len(d2))*100:.1f}% of baseline)")
    
    # Frequency-weighted overlap
    overlap_freq_1 = sum(d1[k] for k in overlap)
    overlap_freq_2 = sum(d2[k] for k in overlap)
    print(f"Frequency-weighted overlap:\n"
          f"  Baseline: {overlap_freq_1}/{sum(d1.values())} ({overlap_freq_1/max(sum(d1.values()),1)*100:.1f}%)\n"
          f"  Projector: {overlap_freq_2}/{sum(d2.values())} ({overlap_freq_2/max(sum(d2.values()),1)*100:.1f}%)")
    
    # New pairs (not in baseline)
    new_pairs = set(d2.keys()) - set(d1.keys())
    print(f"New unique pairs added: {len(new_pairs)}")
    
    return {
        'baseline_unique': len(d1),
        'projector_unique': len(d2),
        'overlap': len(overlap),
        'new_pairs': len(new_pairs),
        'baseline_total': sum(d1.values()),
        'projector_total': sum(d2.values()),
        'overlap_freq_1': overlap_freq_1,
        'overlap_freq_2': overlap_freq_2
    }
