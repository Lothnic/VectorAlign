#!/usr/bin/env python3
"""
Full VectorAlign Pipeline:
Phase 1: Train contrastive projector on parallel corpora
Phase 2: Re-align using projector, compare against baseline

Usage:
    python run_pipeline.py --type kangri   # Align Kangri-Hindi
    python run_pipeline.py --type kinnauri # Align Kinnauri-Hindi
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict

# Add parent dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModel
from vectoralign import (
    align,
    ContrastiveProjector,
    train_projector,
    refine_dictionary_with_projector,
    compare_dictionaries,
)
from vectoralign.dictionary_refinement import _build_freq_dict


def load_parallel_data(src_file, tgt_file, n=None):
    """Load parallel source/target file, one sentence per line."""
    with open(src_file, 'r', encoding='utf-8') as f:
        src = [line.strip() for line in f if line.strip()]
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt = [line.strip() for line in f if line.strip()]
    
    n = n if n else min(len(src), len(tgt))
    return src[:n], tgt[:n]


def load_seed_pairs(path):
    """Load seed word pairs (tab-separated)."""
    pairs = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0:  # skip header
                continue
            parts = line.strip().split('\t')
            if len(parts) == 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def summarize_dictionary(freq_dict, name=""):
    """Print basic dictionary statistics."""
    print(f"\n{'='*60}")
    print(f"  {name} Dictionary Summary")
    print(f"{'='*60}")
    print(f"  Total aligned pairs: {sum(freq_dict.values())}")
    print(f"  Unique word pairs:   {len(freq_dict)}")
    
    # Top 15 by frequency
    top = sorted(freq_dict.items(), key=lambda x: -x[1])[:15]
    print(f"  Top 15 pairs:")
    for (src, tgt), count in top:
        print(f"    {src:15s} -> {tgt:20s} ({count}x)")


def main():
    parser = argparse.ArgumentParser(description="VectorAlign Pipeline")
    parser.add_argument("--type", type=str, default="kangri",
                       choices=["kangri", "kinnauri"],
                       help="Language pair to align")
    parser.add_argument("--projector-path", type=str, default=None,
                       help="Path to pre-trained projector (if resuming)")
    parser.add_argument("--epochs", type=int, default=8,
                       help="Number of training epochs for projector")
    parser.add_argument("--eval-size", type=int, default=100,
                       help="Number of word pairs for evaluation")
    parser.add_argument("--projector-hidden", type=int, default=512,
                       help="Hidden size of projector (default: 512)")
    parser.add_argument("--temp", type=float, default=0.07,
                       help="Temperature for InfoNCE loss")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate for projector training")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use")
    parser.add_argument("--threshold", type=float, default=0.4,
                       help="Similarity threshold for projector-based alignment")
    parser.add_argument("--no-train", action="store_true",
                       help="Skip projector training, use existing projector")
    parser.add_argument("--output-dir", type=str, default="output")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() or args.device == "cuda" else "cpu"
    
    # Determine paths
    if args.type == "kangri":
        src_file = "data/parallel/kr_hi_src.txt"
        tgt_file = "data/parallel/kr_hi_tgt.txt"
        baseline_dict = os.path.join(args.output_dir, "hindi_kangri_dict.tsv")
        projector_dict = os.path.join(args.output_dir, "hindi_kangri_dict_projector.tsv")
        projector_path = "checkpoints/projector_kangri.pt"
    elif args.type == "kinnauri":
        src_file = "data/parallel/kp_hi_src.txt"
        tgt_file = "data/parallel/kp_hi_tgt.txt"
        baseline_dict = os.path.join(args.output_dir, "hindi_kinnauri_dict.tsv")
        projector_dict = os.path.join(args.output_dir, "hindi_kinnauri_dict_projector.tsv")
        projector_path = "checkpoints/projector_kinnauri.pt"
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  VectorAlign Pipeline: {args.type.upper()}")
    print(f"  Phase 1: Train contrastive projector")
    print(f"  Phase 2: Align with projector & compare")
    print(f"{'='*60}\n")
    
    # Load parallel data for projectoer training
    n_train = None  # Use all data for training
    src_par, tgt_par = load_parallel_data(src_file, tgt_file, n=n_train)
    print(f"Loaded {len(src_par)} parallel sentences for projector training")
    
    # ========== PHASE 1: Train Projector ==========
    if args.no_train:
        print("\n[Phase 1] Skipping projector training!")
        projector = ContrastiveProjector(dim=512)
        try:
            state = torch.load(projector_path, map_location=device)
            projector.load_state_dict(state)
            print(f"Loaded projector from {projector_path}")
        except FileNotFoundError:
            print(f"ERROR: No projector at {projector_path}")
            return
    else:
        print(f"\n[Phase 1] Training contrastive projector...")
        print(f"  Training with {len(src_par)} parallel sentences")
        print(f"  Epochs: {args.epochs}, LR: {args.lr}, Temp: {args.temp}")
        
        # Initialize projector (LaBSE outputs 768-dim vectors)
        projector = ContrastiveProjector(dim=768)
        
        # Load model (LaBSE)
        tokenizer = AutoTokenizer.from_pretrained("setu4993/LaBSE")
        model = AutoModel.from_pretrained("setu4993/LaBSE").to(device)
        model.eval()
        
        # Train projector
        print("Training projector...")
        projector = train_projector(
            src_sentences=src_par,
            tgt_sentences=tgt_par,
            projector=projector,
            tokenizer=tokenizer,
            model=model,
            max_epochs=args.epochs,
            lr=args.lr,
            batch_size=64,
            temperature=args.temp,
            device=device,
            output_path=projector_path,
        )
    
    # ========== PHASE 2: Align with Projector ==========
    print(f"\n[Phase 2] Aligning with contrastive projector...")
    
    # Align with projector
    freq_projector = refine_dictionary_with_projector(
        src_sentences=src_par,
        tgt_sentences=tgt_par,
        projector=projector,
        tokenizer=tokenizer,
        model=model,
        batch_size=32,
        threshold=args.threshold,
        device=device,
        output_path=projector_dict,
    )
    
    # ========== PHASE 3: Compare ==========
    print(f"\n[Phase 3] Comparing dictionaries...")
    
    # Check if baseline exists
    if os.path.exists(baseline_dict):
        comparison = compare_dictionaries(baseline_dict, projector_dict)
        print("\nComparison results:")
        for k, v in comparison.items():
            print(f"  {k}: {v}")
        
        # Calculate improvement
        overlap_pct = comparison['overlap'] / max(comparison['baseline_unique'], 1)
        new_pct = comparison['new_pairs'] / max(comparison['projector_unique'], 1)
        print(f"\n  Overlap rate: {overlap_pct*100:.1f}%")
        print(f"  New unique pairs: {new_pct*100:.1f}%")
    else:
        print(f"  Baseline dictionary not found: {baseline_dict}")
        print(f"  Run baseline alignment first (python align.py)")
    
    # ========== PHASE 4: Evaluation (if seed pairs available) ==========
    print(f"\n[Phase 4] Evaluation...")
    
    try:
        seed_pairs = load_seed_pairs("data/parallel/seed_pairs.txt")
        evaluator = Evaluator(seed_pairs)
        
        # Evaluate baseline
        if os.path.exists(baseline_dict):
            eval_baseline = evaluator.evaluate_dict(baseline_dict)
            print(f"\n  Baseline evaluation:")
            evaluator.print_results(eval_baseline)
        
        # Evaluate projector
        eval_projector = evaluator.evaluate_dict(projector_dict)
        print(f"\n  Projector evaluation:")
        evaluator.print_results(eval_projector)
        
        # Show improvement
        if os.path.exists(baseline_dict):
            print(f"\n  Improvement:")
            for metric in ['precision_1', 'precision_3', 'precision_5', 'recall']:
                if eval_baseline[metric] > 0:
                    change = eval_projector[metric] - eval_baseline[metric]
                    pct_change = change/max(eval_baseline[metric], 0.001)*100
                    sign = "+" if change >= 0 else "-"
                    print(f"    {metric}: {sign}{change*100:.2f}% ({sign}{pct_change:.1f}%)")
    
    except Exception as e:
        print(f"  Evaluation skipped: {e}")
    
    print(f"\n{'='*60}")
    print(f"  Pipeline complete!")
    print(f"  Projector dictionary: {projector_dict}")
    print(f"{'='*60}\n")


class Evaluator:
    """Evaluate alignment dictionaries against gold seed pairs."""
    
    def __init__(self, gold_pairs):
        self.gold_pairs = gold_pairs
        self.gold_src = {p[0] for p in gold_pairs}
        self.gold_tgt = {p[1] for p in gold_pairs}
        self.gold_src_tgt = {p[0]: p[1] for p in gold_pairs}
        self.gold_tgt_src = {p[1]: p[0] for p in gold_pairs}
    
    def evaluate_dict(self, dict_path):
        """Evaluate a dictionary file against gold pairs."""
        freq = defaultdict(int)
        with open(dict_path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    freq[(parts[0], parts[1])] = int(parts[2])
        
        # Convert to simple set of pairs
        pairs = set(freq.keys())
        
        # Precision at k
        prec_1 = self._precision_at_k(pairs, k=1)
        prec_3 = self._precision_at_k(pairs, k=3)
        prec_5 = self._precision_at_k(pairs, k=5)
        
        # Recall (fraction of gold pairs recovered)
        recovered = len(pairs & set(self.gold_pairs))
        recall = recovered / max(len(self.gold_pairs), 1)
        
        # Frequency-weighted precision
        wp = self._weighted_precision(freq)
        
        return {
            'precision_1': prec_1,
            'precision_3': prec_3,
            'precision_5': prec_5,
            'recall': recall,
            'weighted_precision': wp,
            'num_pairs': len(pairs),
        }
    
    def _precision_at_k(self, pairs, k=1):
        """Precision at k: for each source word, what fraction of top-k targets are correct."""
        correct = 0
        total = 0
        
        src_of_pairs = defaultdict(set)
        for s, t in pairs:
            src_of_pairs[s].add(t)
        
        for s, gold_t in self.gold_src_tgt.items():
            if s in src_of_pairs:
                targets = src_of_pairs[s]
                if len(targets) == 1 and list(targets)[0] == gold_t:
                    correct += 1
                else:
                    correct += min(len(targets) / k, 1.0)
                total += 1
        
        return correct / max(total, 1)
    
    def _weighted_precision(self, freq):
        """Frequency-weighted precision."""
        pairs_by_src = defaultdict(lambda: defaultdict(int))
        for (s, t), c in freq.items():
            pairs_by_src[s][t] += c
        
        total_weight = 0
        correct_weight = 0
        
        for s, t_dict in pairs_by_src.items():
            if s in self.gold_src_tgt:
                gold_t = self.gold_src_tgt[s]
                total_weight += sum(t_dict.values())
                correct_weight += t_dict.get(gold_t, 0)
        
        return correct_weight / max(total_weight, 1)
    
    def print_results(self, results):
        """Pretty print evaluation results."""
        print(f"    Precision@1:  {results['precision_1']*100:.2f}%")
        print(f"    Precision@3:  {results['precision_3']*100:.2f}%")
        print(f"    Precision@5:  {results['precision_5']*100:.2f}%")
        print(f"    Recall:       {results['recall']*100:.2f}%")
        print(f"    Weighted Prec: {results['weighted_precision']*100:.2f}%")
        print(f"    Num pairs:    {results['num_pairs']}")


if __name__ == "__main__":
    main()
