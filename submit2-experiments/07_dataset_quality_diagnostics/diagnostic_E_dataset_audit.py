"""
Diagnosis E: Dataset self-check. Diagnosis A-D, with a fixed dataset and varying model parameters,
all show that mismatched categories are difficult to improve through model tuning, and the intra-class
cohesion of this category is very high (0.5106). This script checks the dataset: whether the samples deceiving
the model are carefully designed lexical traps or boundary/mislabeled samples. Three checks are performed: word
length distribution for each category, TF-IDF word overlap between mismatched and aligned samples, and manual
verification of the exported highly confusing mismatched sample text
"""
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items/dataset_items.csv"
DIAGNOSTIC_D_RESULTS = "diagnostic_D_bge_prefixed_results.csv"

LABELS = ["Aligned", "Weakly Aligned", "Mismatched"]


def main():
    print("*" * 70)
    print("  Diagnostic E: Dataset self-audit")
    print("*" * 70)

    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f"OK, Loaded {len(df)} items.")

    # word-count distribution per class
    df["_word_count"] = df["item_text"].apply(lambda t: len(str(t).split()))
    print(f"\n{'Class':<20}{'mean words':>12}{'median':>10}{'min':>8}{'max':>8}{'n':>8}")
    print("-" * 60)
    for label in LABELS:
        sub = df.loc[df["gold_label"] == label, "_word_count"]
        print(f"{label:<20}{sub.mean():>12.1f}{sub.median():>10.1f}{sub.min():>8}{sub.max():>8}{len(sub):>8}")
    print("\n If one class is systematically much longer/shorter than the others,"
          "\n that is a spurious length cue models may be (mis)using")

    #TF-IDF vocabulary overlap
    print("\n" + "*" * 70)
    print("  TF-IDF top terms per class + Aligned Mismatched vocabulary overlap")
    print("*" * 70)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
    tfidf = vec.fit_transform(df["item_text"].astype(str).tolist())
    terms = np.array(vec.get_feature_names_out())

    top_term_sets = {}
    for label in LABELS:
        idx = df.index[df["gold_label"] == label].tolist()
        # Calculate the average TF-IDF weight of each term in this class
        mean_weights = np.asarray(tfidf[idx].mean(axis=0)).ravel()
        top_idx = mean_weights.argsort()[::-1][:20]
        top_terms = terms[top_idx].tolist()
        top_term_sets[label] = set(top_terms)
        print(f"\n{label} top-20 TF-IDF terms:")
        print(", ".join(top_terms))

    for a, b in [("Aligned", "Mismatched"), ("Weakly Aligned", "Mismatched"), ("Aligned", "Weakly Aligned")]:
        overlap = top_term_sets[a] & top_term_sets[b]
        jaccard = len(overlap) / len(top_term_sets[a] | top_term_sets[b])
        print(f"\n{a} <-> {b} top-20 term overlap: {len(overlap)} shared terms, Jaccard={jaccard:.3f}")
        if overlap:
            print(f"  shared terms: {sorted(overlap)}")

    # Export the most messy mismatches
    print("\n" + "*" * 70)
    print("  Manual-read export: highest-scoring Mismatched items (from Diagnostic D)")
    print("*" * 70)

    if os.path.exists(DIAGNOSTIC_D_RESULTS):
        res = pd.read_csv(DIAGNOSTIC_D_RESULTS)
        merged = res.merge(df[["item_id", "item_text"]], on="item_id", how="left")
        mism = merged[merged["gold_label"] == "Mismatched"].sort_values("bge_prefixed_score", ascending=False)
        print(f"\n Top 10 Mismatched items BGE was MOST confident were Aligned "
              f"(highest bge_prefixed_score, predicted='{mism['predicted_label'].mode().iloc[0] if len(mism) else 'n/a'}'):")
        for i, row in mism.head(10).iterrows():
            print(f"\n  item_id={row['item_id']}  score={row['bge_prefixed_score']:.4f}  predicted={row['predicted_label']}")
            print(f"  target_role={row.get('target_role', 'n/a')}")
            print(f"  text: {row['item_text']}")
        mism.head(20).to_csv("diagnostic_E_worst_mismatched_items.csv", index=False)
        print("\n OK, Saved top-20 worst-confused Mismatched items to diagnostic_E_worst_mismatched_items.csv")
    else:
        print(f"\n SKIP, {DIAGNOSTIC_D_RESULTS} not found in working directory  "
              f"run diagnostic_D_bge_with_instruction.py first if you want this export")

if __name__ == "__main__":
    main()
