"""
Diagnostic C: Dataset-coherence test (intra-class embedding cohesion)
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items/dataset_items.csv"
MODEL_PATH = "/kaggle/input/models/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2"

LABELS = ["Aligned", "Weakly Aligned", "Mismatched"]


def main():
    print("*" * 70)
    print("  Diagnostic C: Intra-class embedding cohesion (dataset-coherence test)")
    print("*" * 70)

    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f"OK, Loaded {len(df)} items")

    print(f"\n Loading {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH, local_files_only=True) if MODEL_PATH.startswith("/kaggle") else SentenceTransformer(MODEL_PATH)

    print(" Encoding all item texts ")
    vectors = model.encode(df["item_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    df["_idx"] = range(len(df))

    sim_matrix = vectors @ vectors.T  # full pairwise cosine similarity, since normalised

    print(f"\n {'Class':<20}{'Intra-class cohesion':>24}{'n items':>12}")
    print("-" * 50)
    intra = {}
    for label in LABELS:
        idx = df.loc[df["gold_label"] == label, "_idx"].to_numpy()
        sub = sim_matrix[np.ix_(idx, idx)]
        n = len(idx)

        # exclude diagonal
        off_diag_sum = sub.sum() - np.trace(sub)
        mean_intra = off_diag_sum / (n * (n - 1))
        intra[label] = mean_intra
        print(f"{label:<20}{mean_intra:>24.4f}{n:>12}")

    print(f"\n {'Class pair':<32}{'Inter-class similarity':>24}")
    print("-" * 60)
    inter = {}
    for i, a in enumerate(LABELS):
        for b in LABELS[i + 1:]:
            idx_a = df.loc[df["gold_label"] == a, "_idx"].to_numpy()
            idx_b = df.loc[df["gold_label"] == b, "_idx"].to_numpy()
            sub = sim_matrix[np.ix_(idx_a, idx_b)]
            mean_inter = sub.mean()
            inter[(a, b)] = mean_inter
            print(f"{a} <-> {b:<20}{mean_inter:>24.4f}")

    summary = pd.DataFrame({
        "class": LABELS,
        "intra_class_cohesion": [intra[l] for l in LABELS],
    })

    summary.to_csv("diagnostic_C_intraclass_cohesion_results.csv", index=False)
    print("\nOK, Saved diagnostic_C_intraclass_cohesion_results.csv")


if __name__ == "__main__":
    main()
