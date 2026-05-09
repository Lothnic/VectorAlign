#!/usr/bin/env python3
"""
VectorAlign Benchmark: Compare baseline vs contrastive projector alignment.

Implements the 4-layer evaluation framework:
  Layer 1: Train/test split on seed pairs (gold-standard metrics)
  Layer 2: Dictionary-level structural comparison
  Layer 3: Embedding space visualization (t-SNE)
  Layer 4: Per-word case studies

Usage:
    python scripts/benchmark.py --type kangri --device auto
    python scripts/benchmark.py --type kangri --skip-viz  # skip t-SNE plots
"""

import os
import sys
import argparse
import random
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


# ============================================================
# Layer 1: Gold-Standard Evaluation on Held-Out Seed Pairs
# ============================================================

def load_dict_as_freq(path):
    """Load a TSV dictionary into a {(src, tgt): freq} dict."""
    freq = {}
    with open(path, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                freq[(parts[0], parts[1])] = int(parts[2])
    return freq


def evaluate_on_held_out(dict_path, test_pairs):
    """
    Evaluate a dictionary against held-out gold pairs.
    
    Metrics:
        recall@1: Is the gold translation the top-ranked candidate?
        recall@5: Is the gold translation in the top-5 candidates?
        mrr: Mean Reciprocal Rank of the gold translation
        coverage: Fraction of test source words found in the dictionary
    """
    freq = load_dict_as_freq(dict_path)
    
    # Group by source word, sorted by frequency (descending)
    src_to_targets = defaultdict(list)
    for (s, t), count in sorted(freq.items(), key=lambda x: -x[1]):
        src_to_targets[s].append((t, count))
    
    recall_1, recall_5, mrr, coverage = 0, 0, 0.0, 0
    
    for src_gold, tgt_gold in test_pairs:
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
    
    n = max(len(test_pairs), 1)
    return {
        "recall@1": recall_1 / n,
        "recall@5": recall_5 / n,
        "mrr": mrr / n,
        "coverage": coverage / n,
        "recall_1_count": recall_1,
        "recall_5_count": recall_5,
        "coverage_count": coverage,
        "total_test": len(test_pairs),
        "num_dict_pairs": len(freq),
    }


# ============================================================
# Layer 2: Dictionary Structural Comparison
# ============================================================

def structural_comparison(baseline_path, projector_path):
    """Compare two dictionaries structurally."""
    d1 = load_dict_as_freq(baseline_path)
    d2 = load_dict_as_freq(projector_path)
    
    set1 = set(d1.keys())
    set2 = set(d2.keys())
    
    overlap = set1 & set2
    new_pairs = set2 - set1
    lost_pairs = set1 - set2
    
    # Source word coverage
    src1 = {s for s, t in d1}
    src2 = {s for s, t in d2}
    
    # Frequency statistics of new/lost pairs
    new_freq = [d2[p] for p in new_pairs] if new_pairs else [0]
    lost_freq = [d1[p] for p in lost_pairs] if lost_pairs else [0]
    
    return {
        "baseline_unique": len(d1),
        "projector_unique": len(d2),
        "overlap": len(overlap),
        "new_pairs": len(new_pairs),
        "lost_pairs": len(lost_pairs),
        "baseline_src_words": len(src1),
        "projector_src_words": len(src2),
        "baseline_total_freq": sum(d1.values()),
        "projector_total_freq": sum(d2.values()),
        "overlap_freq_baseline": sum(d1[p] for p in overlap),
        "overlap_freq_projector": sum(d2[p] for p in overlap),
        "new_pairs_avg_freq": np.mean(new_freq),
        "lost_pairs_avg_freq": np.mean(lost_freq),
    }


# ============================================================
# Layer 3: Embedding Space Visualization
# ============================================================

def visualize_embedding_space(seed_pairs, model, tokenizer, projector, output_dir, device="cpu"):
    """Create t-SNE plots of embedding space before and after projection."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Skipping visualization: install scikit-learn and matplotlib")
        print("  pip install scikit-learn matplotlib")
        return
    
    n_pairs = min(100, len(seed_pairs))
    pairs = seed_pairs[:n_pairs]
    hindi_words = [p[0] for p in pairs]
    kangri_words = [p[1] for p in pairs]
    
    # Encode words
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        hi_tok = tokenizer(hindi_words, return_tensors="pt", padding=True, truncation=True).to(device)
        hi_emb = model(**hi_tok).pooler_output.cpu().numpy()
        
        kr_tok = tokenizer(kangri_words, return_tensors="pt", padding=True, truncation=True).to(device)
        kr_emb = model(**kr_tok).pooler_output.cpu().numpy()
    
    # --- BEFORE projection ---
    all_emb = np.concatenate([hi_emb, kr_emb])
    tsne_raw = TSNE(n_components=2, random_state=42, perplexity=min(30, n_pairs-1)).fit_transform(all_emb)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    ax = axes[0]
    ax.scatter(tsne_raw[:n_pairs, 0], tsne_raw[:n_pairs, 1], 
               c='#e74c3c', alpha=0.7, s=30, label='Hindi')
    ax.scatter(tsne_raw[n_pairs:, 0], tsne_raw[n_pairs:, 1], 
               c='#3498db', alpha=0.7, s=30, label='Kangri/Kinnauri')
    # Draw lines between translation pairs
    for i in range(n_pairs):
        ax.plot([tsne_raw[i, 0], tsne_raw[n_pairs+i, 0]], 
                [tsne_raw[i, 1], tsne_raw[n_pairs+i, 1]], 
                'gray', alpha=0.15, linewidth=0.5)
    ax.set_title('BEFORE Projection (Raw LaBSE)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    
    # --- AFTER projection ---
    projector = projector.to(device)
    projector.eval()
    
    with torch.no_grad():
        hi_proj = projector.project_tokens(
            torch.tensor(hi_emb).unsqueeze(1).to(device), side="src"
        ).squeeze(1).cpu().numpy()
        kr_proj = projector.project_tokens(
            torch.tensor(kr_emb).unsqueeze(1).to(device), side="tgt"
        ).squeeze(1).cpu().numpy()
    
    all_proj = np.concatenate([hi_proj, kr_proj])
    tsne_proj = TSNE(n_components=2, random_state=42, perplexity=min(30, n_pairs-1)).fit_transform(all_proj)
    
    ax = axes[1]
    ax.scatter(tsne_proj[:n_pairs, 0], tsne_proj[:n_pairs, 1], 
               c='#e74c3c', alpha=0.7, s=30, label='Hindi')
    ax.scatter(tsne_proj[n_pairs:, 0], tsne_proj[n_pairs:, 1], 
               c='#3498db', alpha=0.7, s=30, label='Kangri/Kinnauri')
    for i in range(n_pairs):
        ax.plot([tsne_proj[i, 0], tsne_proj[n_pairs+i, 0]], 
                [tsne_proj[i, 1], tsne_proj[n_pairs+i, 1]], 
                'gray', alpha=0.15, linewidth=0.5)
    ax.set_title('AFTER Projection (Contrastive Projector)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.suptitle('Embedding Space Alignment: Hindi ↔ Kangri/Kinnauri', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "embedding_space_comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to {plot_path}")
    
    # Compute mean pairwise distance change
    raw_dists = np.linalg.norm(hi_emb - kr_emb, axis=1)
    proj_dists = np.linalg.norm(hi_proj - kr_proj, axis=1)
    print(f"  Mean translation pair distance: {raw_dists.mean():.4f} (raw) → {proj_dists.mean():.4f} (projected)")
    print(f"  Distance reduction: {(1 - proj_dists.mean()/raw_dists.mean())*100:.1f}%")


# ============================================================
# Layer 4: Per-Word Case Studies
# ============================================================

def per_word_case_studies(baseline_path, projector_path, test_pairs, n=30):
    """Generate side-by-side comparison for specific words."""
    d1 = load_dict_as_freq(baseline_path)
    d2 = load_dict_as_freq(projector_path)
    
    # Group by source word
    base_src = defaultdict(list)
    proj_src = defaultdict(list)
    for (s, t), c in sorted(d1.items(), key=lambda x: -x[1]):
        base_src[s].append((t, c))
    for (s, t), c in sorted(d2.items(), key=lambda x: -x[1]):
        proj_src[s].append((t, c))
    
    # Select test words that appear in at least one dictionary
    study_pairs = []
    for src, tgt in test_pairs:
        if src in base_src or src in proj_src:
            study_pairs.append((src, tgt))
        if len(study_pairs) >= n:
            break
    
    results = []
    for src_gold, tgt_gold in study_pairs:
        base_top3 = base_src.get(src_gold, [])[:3]
        proj_top3 = proj_src.get(src_gold, [])[:3]
        
        base_correct = any(t == tgt_gold for t, _ in base_top3)
        proj_correct = any(t == tgt_gold for t, _ in proj_top3)
        
        results.append({
            "hindi": src_gold,
            "gold": tgt_gold,
            "baseline_top3": base_top3,
            "projector_top3": proj_top3,
            "baseline_correct": base_correct,
            "projector_correct": proj_correct,
        })
    
    return results


# ============================================================
# Report Generation
# ============================================================

def generate_report(results, output_dir, lang_type):
    """Generate a markdown report with all evaluation results."""
    report_path = os.path.join(output_dir, "benchmark_results.md")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# VectorAlign Benchmark Results: {lang_type.title()}\n\n")
        f.write(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        # Layer 1: Gold-Standard Metrics
        if "layer1" in results:
            f.write("## Layer 1: Gold-Standard Evaluation (Held-Out Seed Pairs)\n\n")
            l1 = results["layer1"]
            f.write(f"Test set: {l1['baseline']['total_test']} held-out pairs "
                    f"(20% of seed pairs, never seen during training)\n\n")
            
            f.write("| Metric | Baseline | Projector | Δ Change |\n")
            f.write("|---|---|---|---|\n")
            for m in ["recall@1", "recall@5", "mrr", "coverage"]:
                b = l1["baseline"][m]
                p = l1["projector"][m]
                delta = p - b
                sign = "+" if delta >= 0 else ""
                if m == "mrr":
                    f.write(f"| {m} | {b:.4f} | {p:.4f} | {sign}{delta:.4f} |\n")
                else:
                    f.write(f"| {m} | {b*100:.2f}% | {p*100:.2f}% | {sign}{delta*100:.2f}pp |\n")
            f.write("\n")
        
        # Layer 2: Structural Comparison
        if "layer2" in results:
            f.write("## Layer 2: Dictionary Structural Comparison\n\n")
            l2 = results["layer2"]
            f.write("| Metric | Baseline | Projector |\n")
            f.write("|---|---|---|\n")
            f.write(f"| Unique word pairs | {l2['baseline_unique']:,} | {l2['projector_unique']:,} |\n")
            f.write(f"| Unique source words | {l2['baseline_src_words']:,} | {l2['projector_src_words']:,} |\n")
            f.write(f"| Total frequency | {l2['baseline_total_freq']:,} | {l2['projector_total_freq']:,} |\n")
            f.write(f"\n")
            f.write(f"| Overlap | New pairs (projector only) | Lost pairs (baseline only) |\n")
            f.write(f"|---|---|---|\n")
            f.write(f"| {l2['overlap']:,} | {l2['new_pairs']:,} (avg freq: {l2['new_pairs_avg_freq']:.1f}) "
                    f"| {l2['lost_pairs']:,} (avg freq: {l2['lost_pairs_avg_freq']:.1f}) |\n\n")
        
        # Layer 4: Case Studies
        if "layer4" in results:
            f.write("## Layer 4: Per-Word Case Studies\n\n")
            f.write("| Hindi | Gold | Baseline Top-3 | ✓ | Projector Top-3 | ✓ |\n")
            f.write("|---|---|---|---|---|---|\n")
            for cs in results["layer4"]:
                base_str = ", ".join(f"{t}({c}x)" for t, c in cs["baseline_top3"]) or "—"
                proj_str = ", ".join(f"{t}({c}x)" for t, c in cs["projector_top3"]) or "—"
                b_mark = "✓" if cs["baseline_correct"] else "✗"
                p_mark = "✓" if cs["projector_correct"] else "✗"
                f.write(f"| {cs['hindi']} | {cs['gold']} | {base_str} | {b_mark} | {proj_str} | {p_mark} |\n")
            
            # Summary
            base_correct = sum(1 for cs in results["layer4"] if cs["baseline_correct"])
            proj_correct = sum(1 for cs in results["layer4"] if cs["projector_correct"])
            total = len(results["layer4"])
            f.write(f"\n**Summary**: Baseline {base_correct}/{total} correct, "
                    f"Projector {proj_correct}/{total} correct\n\n")
    
    print(f"  Report saved to {report_path}")
    return report_path


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="VectorAlign Benchmark")
    parser.add_argument("--type", type=str, default="kangri",
                       choices=["kangri", "kinnauri"])
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"])
    parser.add_argument("--skip-viz", action="store_true",
                       help="Skip t-SNE visualization")
    parser.add_argument("--test-split", type=float, default=0.2,
                       help="Fraction of seed pairs for test set (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for train/test split")
    parser.add_argument("--output-dir", type=str, default="output")
    args = parser.parse_args()
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Paths
    if args.type == "kangri":
        baseline_dict = os.path.join(args.output_dir, "hindi_kangri_dict.tsv")
        projector_dict = os.path.join(args.output_dir, "hindi_kangri_dict_projector.tsv")
        projector_path = "checkpoints/projector_kangri.pt"
    else:
        baseline_dict = os.path.join(args.output_dir, "hindi_kinnauri_dict.tsv")
        projector_dict = os.path.join(args.output_dir, "hindi_kinnauri_dict_projector.tsv")
        projector_path = "checkpoints/projector_kinnauri.pt"
    
    # Validate files exist
    for p, label in [(baseline_dict, "Baseline dict"), (projector_dict, "Projector dict")]:
        if not os.path.exists(p):
            print(f"ERROR: {label} not found: {p}")
            print(f"  Run the pipeline first: python run_pipeline.py --type {args.type}")
            return
    
    print(f"\n{'='*60}")
    print(f"  VectorAlign Benchmark: {args.type.upper()}")
    print(f"  Baseline:  {baseline_dict}")
    print(f"  Projector: {projector_dict}")
    print(f"{'='*60}\n")
    
    results = {}
    
    # --- Layer 1: Gold-Standard Evaluation ---
    print("[Layer 1] Gold-standard evaluation on held-out seed pairs...")
    try:
        # Load and split seed pairs
        seed_pairs = []
        with open("data/parallel/seed_pairs.txt", 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    seed_pairs.append((parts[0], parts[1]))
        
        random.seed(args.seed)
        random.shuffle(seed_pairs)
        split_idx = int(len(seed_pairs) * (1 - args.test_split))
        train_pairs = seed_pairs[:split_idx]
        test_pairs = seed_pairs[split_idx:]
        
        print(f"  Seed pairs: {len(seed_pairs)} total")
        print(f"  Train set: {len(train_pairs)} (for projector training)")
        print(f"  Test set: {len(test_pairs)} (held out for evaluation)")
        
        eval_baseline = evaluate_on_held_out(baseline_dict, test_pairs)
        eval_projector = evaluate_on_held_out(projector_dict, test_pairs)
        
        results["layer1"] = {"baseline": eval_baseline, "projector": eval_projector}
        
        print(f"\n  {'Metric':<12s}  {'Baseline':>10s}  {'Projector':>10s}  {'Δ':>10s}")
        print(f"  {'-'*46}")
        for m in ["recall@1", "recall@5", "mrr", "coverage"]:
            b, p = eval_baseline[m], eval_projector[m]
            d = p - b
            sign = "+" if d >= 0 else ""
            if m == "mrr":
                print(f"  {m:<12s}  {b:>10.4f}  {p:>10.4f}  {sign}{d:>9.4f}")
            else:
                print(f"  {m:<12s}  {b*100:>9.2f}%  {p*100:>9.2f}%  {sign}{d*100:>8.2f}pp")
        print()
    except FileNotFoundError:
        print("  Skipping: seed_pairs.txt not found")
        test_pairs = []
    
    # --- Layer 2: Structural Comparison ---
    print("[Layer 2] Dictionary structural comparison...")
    struct = structural_comparison(baseline_dict, projector_dict)
    results["layer2"] = struct
    
    print(f"  Baseline:  {struct['baseline_unique']:,} unique pairs, "
          f"{struct['baseline_src_words']:,} source words")
    print(f"  Projector: {struct['projector_unique']:,} unique pairs, "
          f"{struct['projector_src_words']:,} source words")
    print(f"  Overlap: {struct['overlap']:,} | "
          f"New: {struct['new_pairs']:,} (avg freq {struct['new_pairs_avg_freq']:.1f}) | "
          f"Lost: {struct['lost_pairs']:,} (avg freq {struct['lost_pairs_avg_freq']:.1f})")
    print()
    
    # --- Layer 3: Visualization ---
    if not args.skip_viz:
        print("[Layer 3] Embedding space visualization...")
        try:
            from transformers import AutoTokenizer, AutoModel
            from vectoralign import ContrastiveProjector
            from vectoralign.projector import load_projector
            
            tokenizer = AutoTokenizer.from_pretrained("setu4993/LaBSE")
            model = AutoModel.from_pretrained("setu4993/LaBSE")
            projector = load_projector(projector_path, device=device)
            
            visualize_embedding_space(
                seed_pairs, model, tokenizer, projector, args.output_dir, device
            )
        except Exception as e:
            print(f"  Visualization failed: {e}")
        print()
    else:
        print("[Layer 3] Skipping visualization (--skip-viz)\n")
    
    # --- Layer 4: Case Studies ---
    if test_pairs:
        print("[Layer 4] Per-word case studies...")
        case_studies = per_word_case_studies(baseline_dict, projector_dict, test_pairs, n=30)
        results["layer4"] = case_studies
        
        base_correct = sum(1 for cs in case_studies if cs["baseline_correct"])
        proj_correct = sum(1 for cs in case_studies if cs["projector_correct"])
        print(f"  Checked {len(case_studies)} words")
        print(f"  Baseline correct in top-3: {base_correct}/{len(case_studies)}")
        print(f"  Projector correct in top-3: {proj_correct}/{len(case_studies)}")
        print()
    
    # --- Generate Report ---
    print("Generating report...")
    report_path = generate_report(results, args.output_dir, args.type)
    
    print(f"\n{'='*60}")
    print(f"  Benchmark complete!")
    print(f"  Report: {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
