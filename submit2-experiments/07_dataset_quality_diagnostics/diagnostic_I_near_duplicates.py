"""
Diagnosis I: Near-Duplicate Item Detection.

Check a dataset containing 200 entries for unexpected near-duplicate items—pairs of entries
that are essentially the same item but with slightly different wording. Near-duplicate items are a
real risk in handcrafted datasets, as these datasets may be written across multiple entries/sessions
(it's easy to unconsciously reuse wording templates). If near-duplicate items are present,
they can overstate similarity-based metrics (such as intra-class cohesion in Diagnosis C)
and cause bias in cross-validation, as a pair of near-duplicate items may end up scattered across
the training/test fold, thus leaking information.
"""
import difflib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items/dataset_items.csv"
MODEL_PATH = "/kaggle/input/models/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2"

COSINE_THRESHOLD = 0.93
TEXT_THRESHOLD = 0.75


def main():
    print("*" * 70)
    print("  Diagnostic I: Near-duplicate item detection")
    print("*" * 70)

    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f"OK, Loaded {len(df)} items")

    print(f"\n Loading {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH, local_files_only=True) if MODEL_PATH.startswith(
        "/kaggle") else SentenceTransformer(MODEL_PATH)

    print(" Encoding all item texts")
    vectors = model.encode(df["item_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    sim_matrix = vectors @ vectors.T

    texts = df["item_text"].astype(str).tolist()
    ids = df["item_id"].tolist()
    labels = df["gold_label"].tolist()

    n = len(df)
    candidates = []
    print(f"\n Screening {n * (n - 1) // 2} item pairs for cosine similarity >= {COSINE_THRESHOLD}")
    for i in range(n):
        for j in range(i + 1, n):
            cos = sim_matrix[i, j]
            if cos >= COSINE_THRESHOLD:
                candidates.append((i, j, cos))
    print(f"OK, {len(candidates)} candidate pairs passed the embedding-similarity screen")

    print(f"\n Checking literal text similarity (difflib) for candidate pairs, threshold >= {TEXT_THRESHOLD} ")

    flagged = []
    for i, j, cos in candidates:
        ratio = difflib.SequenceMatcher(None, texts[i], texts[j]).ratio()
        if ratio >= TEXT_THRESHOLD:
            flagged.append({
                "item_id_a": ids[i], "item_id_b": ids[j],
                "label_a": labels[i], "label_b": labels[j],
                "cosine_similarity": round(float(cos), 4),
                "text_similarity": round(float(ratio), 4),
                "text_a": texts[i], "text_b": texts[j],
            })

    print("\n" + "*" * 70)
    print(f"  {len(flagged)} pairs flagged as likely near-duplicates (both signals high)")
    print("*" * 70)
    if not flagged:
        print("  OK, No near-duplicate pairs detected. The dataset does not contain")
        print("  accidental repeat items above the similarity thresholds used")
    else:
        flagged_sorted = sorted(flagged, key=lambda r: r["text_similarity"], reverse=True)
        for r in flagged_sorted:
            same_label = "SAME class" if r["label_a"] == r["label_b"] else "DIFFERENT classes"
            print(f"\n  {r['item_id_a']} ({r['label_a']})  <->  {r['item_id_b']} ({r['label_b']})  [{same_label}]")
            print(f"  cosine={r['cosine_similarity']}  text_similarity={r['text_similarity']}")
            print(f"  A: {r['text_a']}")
            print(f"  B: {r['text_b']}")
        print(f"\n  FLAG, Review these {len(flagged)} pairs , if genuinely near-duplicate,"
              "\n  consider whether they inflate intra-class cohesion metrics or risk"
              "\n  train/test leakage across CV folds, and whether one item in each pair"
              "\n  should be revised or removed.")

    out = pd.DataFrame(flagged)
    out.to_csv("diagnostic_I_near_duplicates_results.csv", index=False)
    print("\nOK, Saved diagnostic_I_near_duplicates_results.csv")

if __name__ == "__main__":
    main()
