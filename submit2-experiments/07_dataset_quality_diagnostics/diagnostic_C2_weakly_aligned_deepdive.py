"""
    Diagnostic C2: Weakly Aligned deep-dive cohesion audit.
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items/dataset_items.csv"
MODEL_PATH = "/kaggle/input/models/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2"
# MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

LABELS = ["Aligned", "Weakly Aligned", "Mismatched"]
RANDOM_STATE = 42


def pairwise_stats(sim_matrix, idx):
    sub = sim_matrix[np.ix_(idx, idx)]
    n = len(idx)
    mask = ~np.eye(n, dtype=bool)
    vals = sub[mask]
    return {
        "mean": vals.mean(),
        "std": vals.std(),
        "min": vals.min(),
        "max": vals.max(),
        "median": np.median(vals),
    }


def main():
    print("*" * 70)
    print("  Diagnostic C2: Weakly Aligned deep-dive cohesion audit")
    print("*" * 70)

    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f"OK, Loaded {len(df)} items")

    print(f"\n Loading {MODEL_PATH} ")
    model = SentenceTransformer(MODEL_PATH, local_files_only=True) if MODEL_PATH.startswith(
        "/kaggle") else SentenceTransformer(MODEL_PATH)

    print(" Encoding all item texts")
    vectors = model.encode(df["item_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    df["_idx"] = range(len(df))
    sim_matrix = vectors @ vectors.T

    print(f"\n{'Class':<20}{'mean':>8}{'std':>8}{'min':>8}{'max':>8}{'median':>8}{'n':>6}")
    print("-" * 70)
    for label in LABELS:
        idx = df.loc[df["gold_label"] == label, "_idx"].to_numpy()
        stats = pairwise_stats(sim_matrix, idx)
        flag = "  <-- full distribution, not just the mean" if label == "Weakly Aligned" else ""
        print(f"{label:<20}{stats['mean']:>8.4f}{stats['std']:>8.4f}{stats['min']:>8.4f}"
              f"{stats['max']:>8.4f}{stats['median']:>8.4f}{len(idx):>6}{flag}")

    print("\n" + "*" * 70)
    print("  Sub-clustering check: does Weakly Aligned split into 2 tight,")
    print("  mutually dissimilar sub-groups?")
    print("*" * 70)

    for label in LABELS:
        idx = df.loc[df["gold_label"] == label, "_idx"].to_numpy()
        sub_vectors = vectors[idx]
        km = KMeans(n_clusters=2, n_init=10, random_state=RANDOM_STATE).fit(sub_vectors)
        c0 = idx[km.labels_ == 0]
        c1 = idx[km.labels_ == 1]
        within0 = pairwise_stats(sim_matrix, c0)["mean"] if len(c0) > 1 else float("nan")
        within1 = pairwise_stats(sim_matrix, c1)["mean"] if len(c1) > 1 else float("nan")
        between = sim_matrix[np.ix_(c0, c1)].mean() if len(c0) and len(c1) else float("nan")
        print(f"\n{label}: cluster sizes = {len(c0)} / {len(c1)}")
        print(f"  within-cluster-0 similarity = {within0:.4f}")
        print(f"  within-cluster-1 similarity = {within1:.4f}")
        print(f"  between-cluster similarity  = {between:.4f}")
        gap = min(within0, within1) - between
        if label == "Weakly Aligned":
            if gap > 0.10:
                print("  FLAG, Both sub-clusters are noticeably tighter internally than they are")
                print("  to each other  Weakly Aligned likely spans two distinguishable")
                print("  sub-patterns rather than being one uniform category. Worth reviewing")
                print("  which items fall in each sub-cluster.")
            else:
                print("  OK, No strong sub-cluster split detected  Weakly Aligned behaves as")
                print("  one reasonably uniform category, consistent with a well-defined label.")

if __name__ == "__main__":
    main()
