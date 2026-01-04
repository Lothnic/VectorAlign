"""
Core alignment functions for VectorAlign.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import gc
import numpy as np
from tqdm import tqdm
import string


def get_embeddings_batch(src_sentences: list[str], tgt_sentences: list[str], tokenizer, model, batch_size=32, device='cpu'):
    """
    Get sentence and word embeddings for source and target sentences in batches.
    
    Args:
        src_sentences: List of source language sentences
        tgt_sentences: List of target language sentences
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        batch_size: Batch size for processing
        device: Device to use ('cuda' or 'cpu')
    
    Returns:
        Tuple of (src_embeddings, tgt_embeddings, src_word_embeddings, tgt_word_embeddings)
    """
    src_embeddings = []
    tgt_embeddings = []
    src_word_embeddings = []
    tgt_word_embeddings = []
    
    length = min(len(src_sentences), len(tgt_sentences))
    with torch.no_grad():
        for i in range(0, length, batch_size):
            batch_src = src_sentences[i:i+batch_size]
            batch_tgt = tgt_sentences[i:i+batch_size]

            src_tokens = tokenizer(batch_src, return_tensors='pt', padding=True, truncation=True).to(device)
            tgt_tokens = tokenizer(batch_tgt, return_tensors='pt', padding=True, truncation=True).to(device)

            src_out = model(**src_tokens)
            tgt_out = model(**tgt_tokens)

            src_embeddings.append(src_out.pooler_output.detach().cpu())
            tgt_embeddings.append(tgt_out.pooler_output.detach().cpu())
            src_word_embeddings.append(src_out.last_hidden_state.detach().cpu())
            tgt_word_embeddings.append(tgt_out.last_hidden_state.detach().cpu())

    src_embeddings = torch.cat(src_embeddings, dim=0)
    tgt_embeddings = torch.cat(tgt_embeddings, dim=0)

    src_embeddings = [e.unsqueeze(0) for e in src_embeddings]
    tgt_embeddings = [e.unsqueeze(0) for e in tgt_embeddings]

    # Flatten word embeddings: each batch has different seq_len, so we split individually
    src_word_flat = []
    tgt_word_flat = []
    
    for batch in src_word_embeddings:
        for j in range(batch.shape[0]):
            src_word_flat.append(batch[j:j+1])

    for batch in tgt_word_embeddings:
        for j in range(batch.shape[0]):
            tgt_word_flat.append(batch[j:j+1])

    return src_embeddings, tgt_embeddings, src_word_flat, tgt_word_flat


def _compute_sim_matrix_batch(src_word_emb, tgt_word_emb, src_sent_emb, tgt_sent_emb, threshold=0.5):
    """Compute similarity matrix using batched PyTorch operations."""
    sent_sim = F.cosine_similarity(src_sent_emb, tgt_sent_emb, dim=-1).item()
    
    if sent_sim < threshold:
        return None

    src_tokens = src_word_emb.squeeze(0)
    tgt_tokens = tgt_word_emb.squeeze(0)
    
    src_norm = F.normalize(src_tokens, dim=-1)
    tgt_norm = F.normalize(tgt_tokens, dim=-1)
    
    sim_matrix = torch.matmul(src_norm, tgt_norm.T)
    return sim_matrix


def _bidir_argmax(sim_matrix):
    """Get bidirectional alignment using argmax intersection."""
    forward = []
    for i in range(sim_matrix.shape[0]):
        best_match = torch.argmax(sim_matrix[i])
        forward.append((i, int(best_match)))

    backward = []
    for j in range(sim_matrix.shape[1]):
        best_match = torch.argmax(sim_matrix[:, j])
        backward.append((int(best_match), j))

    final_alignment = set(forward) & set(backward)
    return list(final_alignment)


def _convert_id_to_token(alignments, src_sentence, tgt_sentence, tokenizer):
    """Convert alignment indices to actual tokens."""
    src_tokens = ["[CLS]"] + tokenizer.tokenize(src_sentence) + ["[SEP]"]
    tgt_tokens = ["[CLS]"] + tokenizer.tokenize(tgt_sentence) + ["[SEP]"]
    
    aligned_pairs = []
    
    for (src_idx, tgt_idx) in alignments:
        if src_idx >= len(src_tokens) or tgt_idx >= len(tgt_tokens):
            continue
        if src_idx == 0 or src_idx == len(src_tokens) - 1:
            continue
        if tgt_idx == 0 or tgt_idx == len(tgt_tokens) - 1:
            continue
        
        aligned_pairs.append((src_tokens[src_idx], tgt_tokens[tgt_idx]))
    
    return aligned_pairs


def _merge_subwords(aligned_pairs):
    """Merge WordPiece subwords into complete words."""
    merged = []
    current_src = ""
    current_tgt = ""
    
    for (src_tok, tgt_tok) in aligned_pairs:
        if src_tok.startswith("##"):
            current_src += src_tok[2:]
        else:
            if current_src:
                merged.append((current_src, current_tgt))
            current_src = src_tok
            current_tgt = tgt_tok
    
    if current_src:
        merged.append((current_src, current_tgt))
    
    return merged


def _build_dictionary(all_alignments):
    """Build frequency dictionary from alignments."""
    from collections import defaultdict
    
    dictionary = defaultdict(int)
    
    for (src, tgt) in all_alignments:
        src_clean = src.lower().strip(string.punctuation)
        tgt_clean = tgt.strip()
        
        if src_clean and tgt_clean:
            dictionary[(src_clean, tgt_clean)] += 1
    
    return dict(dictionary)


def _save_dictionary(dictionary, output_path="output/dict.txt"):
    """Save dictionary to file."""
    import os
    dir_name = os.path.dirname(output_path)
    if dir_name:  # Only makedirs if there's a directory component
        os.makedirs(dir_name, exist_ok=True)
    
    sorted_dict = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for (src, tgt), count in sorted_dict:
            if count > 1:
                f.write(f"{src}\t{tgt}\t{count}\n")
    
    print(f"Dictionary saved to {output_path}")


def align(
    src_sentences: list[str],
    tgt_sentences: list[str],
    batch_size: int = 32,
    model_name: str = "setu4993/LaBSE",
    mode: str = 'multilingual',
    output: str = 'output/dict.txt',
    threshold: float = 0.5
):
    """
    Align words between source and target sentences and build a bilingual dictionary.
    
    Args:
        src_sentences: List of source language sentences
        tgt_sentences: List of target language sentences  
        batch_size: Batch size for embedding computation (default: 32)
        model_name: HuggingFace model name. Use "bert" for BERT models, 
                   or a full model path like "setu4993/LaBSE" (default)
        mode: For BERT models, 'multilingual' or 'uncased' (default: 'multilingual')
        output: Output path for the dictionary file (default: 'output/dict.txt')
        threshold: Minimum sentence similarity threshold (default: 0.5)
    
    Returns:
        dict: Dictionary mapping (src_word, tgt_word) -> count
    
    Example:
        >>> from lexialign import align
        >>> english = ["Hello world", "How are you"]
        >>> hindi = ["नमस्ते दुनिया", "आप कैसे हैं"]
        >>> dictionary = align(english, hindi, model_name="setu4993/LaBSE")
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if model_name == "bert":
        if mode == 'multilingual':
            tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            model = AutoModel.from_pretrained("bert-base-multilingual-cased").to(device)
        else:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
    
    model.eval()
    print(f"Model loaded: {model_name}")

    NUM_SENTENCES = min(len(src_sentences), len(tgt_sentences))

    all_alignments = []

    for i in tqdm(range(0, NUM_SENTENCES, batch_size), desc="Processing sentences"):

        src_embeddings, tgt_embeddings, src_word_embeddings, tgt_word_embeddings = get_embeddings_batch(
            src_sentences[i:i+batch_size], tgt_sentences[i:i+batch_size], tokenizer, model, batch_size, device
        )
    
        for j in range(len(src_embeddings)):
            matrix = _compute_sim_matrix_batch(
                src_word_embeddings[j],
                tgt_word_embeddings[j],
                src_embeddings[j],
                tgt_embeddings[j],
                threshold
            )
            
            if matrix is not None:
                alignments = _bidir_argmax(matrix)
                token_pairs = _convert_id_to_token(
                    alignments,
                    src_sentences[i+j],
                    tgt_sentences[i+j],
                    tokenizer
                )
                merged_pairs = _merge_subwords(token_pairs)
                all_alignments.extend(merged_pairs)
        
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    dictionary = _build_dictionary(all_alignments)
    _save_dictionary(dictionary, output)
    
    return dictionary
