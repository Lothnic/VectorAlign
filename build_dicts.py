"""Build Hindi-Kangri and Hindi-Kinnauri dictionaries using VectorAlign."""

import sys
sys.path.insert(0, '.')

from vectoralign.align import align


def load_sents(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def run_pair(hindi_path, other_path, output_path, label):
    print(f"\n{'='*60}")
    print(f"=== Aligning {label} ===")
    print(f"{'='*60}")
    hindi_sents = load_sents(hindi_path)
    other_sents = load_sents(other_path)
    print(f"Hindi sentences: {len(hindi_sents)}")
    print(f"{label} sentences: {len(other_sents)}")
    freq = align(
        src_sentences=hindi_sents,
        tgt_sentences=other_sents,
        batch_size=32,
        model_name="setu4993/LaBSE",
        device="auto",
        output=output_path,
        threshold=0.5,
    )
    return freq


if __name__ == "__main__":
    # 1. Hindi-Kangri dictionary (src=Hindi, tgt=Kangri)
    freq_kangri = run_pair(
        "data/parallel/kangri_tgt.snt",
        "data/parallel/kangri_src.snt",
        "output/hindi_kangri_dict.tsv",
        "Hindi-Kangri",
    )

    # 2. Hindi-Kinnauri dictionary (src=Hindi, tgt=Kinnauri)
    freq_kinnauri = run_pair(
        "data/parallel/kinnauri_tgt.snt",
        "data/parallel/kinnauri_src.snt",
        "output/hindi_kinnauri_dict.tsv",
        "Hindi-Kinnauri",
    )

    # Dump stats
    from collections import Counter
    def summarise(d, name):
        print(f"\n--- {name} summary ---")
        print(f"Total unique pairs: {len(d)}")
        # Words per direction
        src_words = set()
        tgt_words = set()
        for (s, t), c in d.items():
            src_words.add(s)
            tgt_words.add(t)
        print(f"Unique Hindi words: {len(src_words)}")
        print(f"Unique {name} words: {len(tgt_words)}")
        
        # Multi-target words
        tgt_of_src = {}
        for (s, t), c in d.items():
            tgt_of_src.setdefault(s, set()).add(t)
        multi = {k: v for k, v in tgt_of_src.items() if len(v) > 1}
        print(f"Hindi words with multiple {name} translations: {len(multi)}")
        
        # Top 20 by freq
        top = sorted(d.items(), key=lambda x: -x[1])[:20]
        print("\nTop 20 pairs:")
        for (s, t), c in top:
            print(f"  {s:20s} <-> {t:20s}  ({c}x)")
    
    summarise(freq_kangri, "Hindi-Kangri")
    summarise(freq_kinnauri, "Hindi-Kinnauri")
