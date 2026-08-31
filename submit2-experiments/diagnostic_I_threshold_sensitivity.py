import difflib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items-final/dataset_items_final.csv"
MODEL_PATH = "/kaggle/input/models/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2"

ORIGINAL_COSINE = 0.93
ORIGINAL_TEXT = 0.75

COSINE_GRID = [0.80, 0.85, 0.88, 0.90, 0.93]
TEXT_GRID = [0.55, 0.60, 0.65, 0.70, 0.75]

TOP_K = 10

def main():
    print("*" * 70)
    print("  Diagnostic I supplement: threshold sensitivity sweep")
    print("*" * 70)

    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    texts = df["item_text"].astype(str).tolist()
    ids = df["item_id"].tolist()
    labels = df["gold_label"].tolist()
    n = len(df)
    print(f"OK, Loaded {n} items -> {n * (n - 1) // 2} pairs.")

    print(f"\n Loading {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH, local_files_only=True) if MODEL_PATH.startswith("/kaggle") else SentenceTransformer(MODEL_PATH)

    print(" Encoding all item texts")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    sim_matrix = vectors @ vectors.T

    # Print the complete pairwise cosine distribution
    iu = np.triu_indices(n, k=1)
    all_cos = sim_matrix[iu]
    print("\n" + "-" * 75)
    print("  Empirical distribution: MiniLM pairwise cosine similarity")
    print("-" * 75)
    print(f"  max    = {all_cos.max():.4f}")
    print(f"  p99.9  = {np.percentile(all_cos, 99.9):.4f}")
    print(f"  p99    = {np.percentile(all_cos, 99):.4f}")
    print(f"  median = {np.median(all_cos):.4f}")
    print(f"  Margin below original cosine threshold {ORIGINAL_COSINE}: "
          f"{ORIGINAL_COSINE - all_cos.max():.4f}")

    order = np.argsort(all_cos)[::-1][:TOP_K]
    print(f"\n  Top-{TOP_K} most semantically similar pairs:")
    for rank, k in enumerate(order, 1):
        i, j = iu[0][k], iu[1][k]
        print(f"   {rank:2d}. {ids[i]} ({labels[i]}) <-> {ids[j]} ({labels[j]})  cosine={all_cos[k]:.4f}")

    loosest_cos = min(COSINE_GRID)
    pool_mask = all_cos >= loosest_cos
    pool_idx = set(np.where(pool_mask)[0].tolist())
    pool_idx.update(np.argsort(all_cos)[::-1][:200].tolist())
    pool_idx = sorted(pool_idx)
    print(f"\n Computing difflib ratio for {len(pool_idx)} candidate pairs "
          f"(cosine >= {loosest_cos} or global top-200)")

    pool = []
    for k in pool_idx:
        i, j = iu[0][k], iu[1][k]
        ratio = difflib.SequenceMatcher(None, texts[i], texts[j]).ratio()
        pool.append({"k": k, "i": i, "j": j, "cos": float(all_cos[k]), "ratio": ratio})

    ratios = np.array([p["ratio"] for p in pool])
    print("\n" + "-" * 75)
    print("  Empirical distribution: difflib ratio (within candidate pool)")
    print("-" * 75)
    print(f"  max    = {ratios.max():.4f}")
    print(f"  median = {np.median(ratios):.4f}")
    print(f"  Margin below original text threshold {ORIGINAL_TEXT}: "
          f"{ORIGINAL_TEXT - ratios.max():.4f}")
    print(f"  (Reference: difflib.get_close_matches default cutoff = 0.6)")

    top_ratio = sorted(pool, key=lambda p: p["ratio"], reverse=True)[:TOP_K]
    print(f"\n  Top-{TOP_K} most literally similar pairs in pool:")
    for rank, p in enumerate(top_ratio, 1):
        i, j = p["i"], p["j"]
        print(f"   {rank:2d}. {ids[i]} ({labels[i]}) <-> {ids[j]} ({labels[j]})  "
              f"difflib={p['ratio']:.4f}  cosine={p['cos']:.4f}")

    # threshold grid sweep
    print("\n" + "-" * 75)
    print("  Flagged-pair count across the threshold grid (BOTH signals >= cutoffs)")
    print("-" * 75)
    header = "  cosine\\text |" + "".join(f"  >={t:.2f}" for t in TEXT_GRID)
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows = []
    for c_th in COSINE_GRID:
        cells = []
        for t_th in TEXT_GRID:
            count = sum(1 for p in pool if p["cos"] >= c_th and p["ratio"] >= t_th)
            cells.append(count)
            rows.append({"cosine_threshold": c_th, "text_threshold": t_th, "flagged_pairs": count})
        marker = "  <-- original row" if abs(c_th - ORIGINAL_COSINE) < 1e-9 else ""
        print(f"    >={c_th:.2f}    |" + "".join(f"  {c:5d}" for c in cells) + marker)

    pd.DataFrame(rows).to_csv("diagnostic_I_threshold_sensitivity.csv", index=False)
    print("\nOK， Saved diagnostic_I_threshold_sensitivity.csv")

if __name__ == "__main__":
    main()