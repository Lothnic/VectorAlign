#!/usr/bin/env python3
"""
Generate pseudo-seed pairs for Kinnauri-Hindi from baseline alignment.

Since seed_pairs.txt only contains Kangri-Hindi pairs, this script generates
Kinnauri-Hindi seed pairs by running baseline alignment on the Kinnauri-Hindi
parallel corpus and extracting top-confidence pairs.

Usage:
    python scripts/generate_kinnauri_seeds.py
    python scripts/generate_kinnauri_seeds.py --top-n 2000
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectoralign import align


def main():
    parser = argparse.ArgumentParser(description="Generate Kinnauri-Hindi seed pairs")
    parser.add_argument("--top-n", type=int, default=2000,
                       help="Number of top-frequency pairs to extract as seeds")
    parser.add_argument("--min-freq", type=int, default=3,
                       help="Minimum frequency to include a pair")
    parser.add_argument("--output", type=str, 
                       default="data/parallel/kinnauri_seed_pairs.txt",
                       help="Output path for generated seed pairs")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    
    print("Generating Kinnauri-Hindi seed pairs via baseline alignment...")
    
    # Load Kinnauri-Hindi parallel corpus
    with open("data/parallel/kp_hi_src.txt", 'r', encoding='utf-8') as f:
        src = [line.strip() for line in f if line.strip()]
    with open("data/parallel/kp_hi_tgt.txt", 'r', encoding='utf-8') as f:
        tgt = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {min(len(src), len(tgt))} parallel sentences")
    
    # Run baseline alignment
    freq = align(
        src_sentences=src,
        tgt_sentences=tgt,
        batch_size=32,
        device=args.device,
        output="output/hindi_kinnauri_baseline_for_seeds.tsv",
        threshold=0.5,
    )
    
    # Filter and sort by frequency
    sorted_pairs = sorted(freq.items(), key=lambda x: -x[1])
    filtered = [(pair, count) for pair, count in sorted_pairs if count >= args.min_freq]
    top_pairs = filtered[:args.top_n]
    
    # Save as seed pairs
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("src\ttgt\n")
        for (src_word, tgt_word), count in top_pairs:
            f.write(f"{src_word}\t{tgt_word}\n")
    
    print(f"\nGenerated {len(top_pairs)} Kinnauri-Hindi seed pairs")
    print(f"  Min frequency: {args.min_freq}")
    print(f"  Saved to: {args.output}")
    
    # Show top 20
    print(f"\nTop 20 pairs:")
    for (s, t), c in top_pairs[:20]:
        print(f"  {s:20s} -> {t:20s} ({c}x)")


if __name__ == "__main__":
    main()
