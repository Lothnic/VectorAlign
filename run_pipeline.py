#!/usr/bin/env python3
"""
Full VectorAlign Pipeline:
Phase 1: Train contrastive projector on parallel corpora
Phase 2: Re-align using projector, compare against baseline

Usage:
    python run_pipeline.py --type kangri   # Align Kangri-Hindi
    python run_pipeline.py --type kinnauri # Align Kinnauri-Hindi
    python run_pipeline.py --type kangri --train-only   # Train projector only
    python run_pipeline.py --type kangri --align-only   # Align only (requires trained projector)
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
    compare_dictionaries,
)


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
        
        # Group by source word, sorted by frequency (descending)
        src_to_targets = defaultdict(list)
        for (s, t), count in sorted(freq.items(), key=lambda x: -x[1]):
            src_to_targets[s].append((t, count))
        
        # Recall@1, Recall@5, MRR, Coverage
        recall_1 = 0
        recall_5 = 0
        mrr = 0.0
        coverage = 0
        total = 0
        
        for src_gold, tgt_gold in self.gold_pairs:
            total += 1
            candidates = src_to_targets.get(src_gold, [])
            if not candidates:
                continue
            coverage += 1
            target_list = [t for t, _ in candidates]
            if tgt_gold in target_list:
                rank = target_list.index(tgt_gold) + 1
                mrr += 1.0 / rank
                if rank == 1:
                    recall_1 += 1
                if rank <= 5:
                    recall_5 += 1
        
        n = max(total, 1)
        return {
            'recall_1': recall_1 / n,
            'recall_5': recall_5 / n,
            'mrr': mrr / n,
            'coverage': coverage / n,
            'num_pairs': len(freq),
        }
    
    def print_results(self, results):
        """Pretty print evaluation results."""
        print(f"    Recall@1:   {results['recall_1']*100:.2f}%")
        print(f"    Recall@5:   {results['recall_5']*100:.2f}%")
        print(f"    MRR:        {results['mrr']:.4f}")
        print(f"    Coverage:   {results['coverage']*100:.2f}%")
        print(f"    Num pairs:  {results['num_pairs']}")


def main():
    parser = argparse.ArgumentParser(description="VectorAlign Pipeline")
    parser.add_argument("--type", type=str, default="kangri",
                       choices=["kangri", "kinnauri"],
                       help="Language pair to align")
    parser.add_argument("--projector-path", type=str, default=None,
                       help="Path to pre-trained projector (overrides default)")
    parser.add_argument("--epochs", type=int, default=8,
                       help="Number of training epochs for projector")
    parser.add_argument("--temp", type=float, default=0.07,
                       help="Initial temperature for InfoNCE loss")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate for projector training")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use")
    parser.add_argument("--threshold", type=float, default=0.4,
                       help="Similarity threshold for projector-based alignment")
    parser.add_argument("--train-only", action="store_true",
                       help="Train projector only, skip alignment")
    parser.add_argument("--align-only", action="store_true",
                       help="Align only using existing projector, skip training")
    parser.add_argument("--output-dir", type=str, default="output")
    
    args = parser.parse_args()
    
    # Resolve device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Determine paths
    if args.type == "kangri":
        src_file = "data/parallel/kr_hi_src.txt"
        tgt_file = "data/parallel/kr_hi_tgt.txt"
        baseline_dict = os.path.join(args.output_dir, "hindi_kangri_dict.tsv")
        projector_dict = os.path.join(args.output_dir, "hindi_kangri_dict_projector.tsv")
        projector_path = args.projector_path or "checkpoints/projector_kangri.pt"
    elif args.type == "kinnauri":
        src_file = "data/parallel/kp_hi_src.txt"
        tgt_file = "data/parallel/kp_hi_tgt.txt"
        baseline_dict = os.path.join(args.output_dir, "hindi_kinnauri_dict.tsv")
        projector_dict = os.path.join(args.output_dir, "hindi_kinnauri_dict_projector.tsv")
        projector_path = args.projector_path or "checkpoints/projector_kinnauri.pt"
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  VectorAlign Pipeline: {args.type.upper()}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")
    
    # Load parallel data
    src_par, tgt_par = load_parallel_data(src_file, tgt_file)
    print(f"Loaded {len(src_par)} parallel sentences")
    
    # ========== PHASE 1: Train Projector ==========
    if not args.align_only:
        print(f"\n[Phase 1] Training contrastive projector...")
        print(f"  Epochs: {args.epochs}, LR: {args.lr}, Temp: {args.temp}")
        
        # Initialize projector (LaBSE outputs 768-dim vectors)
        projector = ContrastiveProjector(dim=768)
        
        # Load model (LaBSE)
        tokenizer = AutoTokenizer.from_pretrained("setu4993/LaBSE")
        model = AutoModel.from_pretrained("setu4993/LaBSE").to(device)
        model.eval()
        
        # Train projector
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
        print(f"  Projector saved to {projector_path}")
        
        if args.train_only:
            print(f"\n{'='*60}")
            print(f"  Training complete! (--train-only)")
            print(f"{'='*60}\n")
            return
    else:
        # Verify projector exists
        if not os.path.exists(projector_path):
            print(f"ERROR: No projector at {projector_path}")
            print(f"  Run without --align-only first to train one.")
            return
        print(f"[Phase 1] Skipping training (--align-only)")
        print(f"  Using projector: {projector_path}")
    
    # ========== PHASE 2: Align with Projector ==========
    print(f"\n[Phase 2] Aligning with contrastive projector...")
    
    # Use the unified align() function with projector support
    freq_projector = align(
        src_sentences=src_par,
        tgt_sentences=tgt_par,
        batch_size=32,
        device=device,
        output=projector_dict,
        threshold=args.threshold,
        use_projector=True,
        projector_path=projector_path,
    )
    
    summarize_dictionary(freq_projector, f"Projector ({args.type})")
    
    # ========== PHASE 3: Compare ==========
    print(f"\n[Phase 3] Comparing dictionaries...")
    
    if os.path.exists(baseline_dict):
        comparison = compare_dictionaries(baseline_dict, projector_dict)
        print("\nComparison results:")
        for k, v in comparison.items():
            print(f"  {k}: {v}")
        
        overlap_pct = comparison['overlap'] / max(comparison['baseline_unique'], 1)
        new_pct = comparison['new_pairs'] / max(comparison['projector_unique'], 1)
        print(f"\n  Overlap rate: {overlap_pct*100:.1f}%")
        print(f"  New unique pairs: {new_pct*100:.1f}%")
    else:
        print(f"  Baseline dictionary not found: {baseline_dict}")
        print(f"  Run baseline first: python build_dicts.py")
    
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
            print(f"\n  Improvement (Projector vs Baseline):")
            for metric in ['recall_1', 'recall_5', 'mrr', 'coverage']:
                base_val = eval_baseline[metric]
                proj_val = eval_projector[metric]
                change = proj_val - base_val
                sign = "+" if change >= 0 else ""
                print(f"    {metric:12s}: {sign}{change*100:.2f}pp "
                      f"({base_val*100:.2f}% → {proj_val*100:.2f}%)")
    
    except Exception as e:
        print(f"  Evaluation skipped: {e}")
    
    print(f"\n{'='*60}")
    print(f"  Pipeline complete!")
    print(f"  Projector dictionary: {projector_dict}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
