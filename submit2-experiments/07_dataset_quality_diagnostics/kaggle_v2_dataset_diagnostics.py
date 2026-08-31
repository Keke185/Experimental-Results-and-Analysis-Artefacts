
import difflib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sentence_transformers import SentenceTransformer

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items-final-v2/dataset_items_final_v2.csv"
MODEL_PATH = "/kaggle/input/models/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2"

LABELS = ["Aligned", "Weakly Aligned", "Mismatched"]
COSINE_THRESHOLD = 0.93
TEXT_THRESHOLD = 0.75

WA_D_IDS = {
    "Q_104", "Q_113", "Q_114", "Q_115", "Q_116", "Q_119", "Q_120", "Q_121", "Q_122",
    "Q_123", "Q_124", "Q_125", "Q_126", "Q_127", "Q_128", "Q_130", "Q_136", "Q_138",
    "Q_139", "Q_143", "Q_146", "Q_147", "Q_148", "Q_149", "Q_150",
}
WA_F_IDS = {
    "Q_101", "Q_102", "Q_103", "Q_105", "Q_106", "Q_107", "Q_108", "Q_109", "Q_110",
    "Q_111", "Q_112", "Q_117", "Q_118", "Q_129", "Q_131", "Q_132", "Q_133", "Q_134",
    "Q_135", "Q_137", "Q_140", "Q_141", "Q_142", "Q_144", "Q_145",
}


def near_duplicate_screen(df, vectors):
    print("\n" + "*" * 70)
    print(f"  1. Near-duplicate screen (cosine >= {COSINE_THRESHOLD} AND text_sim >= {TEXT_THRESHOLD})")
    print("*" * 70)
    texts = df["item_text"].astype(str).tolist()
    ids = df["item_id"].tolist()
    labels = df["gold_label"].tolist()
    sim_matrix = vectors @ vectors.T
    n = len(df)

    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= COSINE_THRESHOLD:
                candidates.append((i, j, sim_matrix[i, j]))
    print(f"{len(candidates)} candidate pairs passed the embedding-similarity screen.")

    flagged = []
    for i, j, cos in candidates:
        ratio = difflib.SequenceMatcher(None, texts[i], texts[j]).ratio()
        if ratio >= TEXT_THRESHOLD:
            flagged.append({
                "item_id_a": ids[i], "item_id_b": ids[j],
                "label_a": labels[i], "label_b": labels[j],
                "cosine_similarity": round(float(cos), 4),
                "text_similarity": round(float(ratio), 4),
            })
    if not flagged:
        print(f"OK, No near-duplicate pairs detected (v2 dataset).")
    else:
        for r in flagged:
            print(f"  {r['item_id_a']} <-> {r['item_id_b']}  cosine={r['cosine_similarity']}  "
                  f"text_sim={r['text_similarity']}")
    pd.DataFrame(flagged).to_csv("diagnostic_I_near_duplicates_v2_results.csv", index=False)
    print("OK, Saved diagnostic_I_near_duplicates_v2_results.csv")


def cohesion(df, vectors):
    print("\n" + "*" * 70)
    print("  2.Intra-class cohesion / inter-class similarity")
    print("*" * 70)
    df = df.copy()
    df["_idx"] = range(len(df))
    sim_matrix = vectors @ vectors.T

    print(f"{'Class':<20}{'Intra-class cohesion':>24}{'n items':>12}")
    intra = {}
    for label in LABELS:
        idx = df.loc[df["gold_label"] == label, "_idx"].to_numpy()
        sub = sim_matrix[np.ix_(idx, idx)]
        m = len(idx)
        mean_intra = (sub.sum() - np.trace(sub)) / (m * (m - 1))
        intra[label] = mean_intra
        print(f"{label:<20}{mean_intra:>24.4f}{m:>12}")

    print(f"\n{'Class pair':<32}{'Inter-class similarity':>24}")
    for i, a in enumerate(LABELS):
        for b in LABELS[i + 1:]:
            idx_a = df.loc[df["gold_label"] == a, "_idx"].to_numpy()
            idx_b = df.loc[df["gold_label"] == b, "_idx"].to_numpy()
            sub = sim_matrix[np.ix_(idx_a, idx_b)]
            print(f"{a} <-> {b:<20}{sub.mean():>24.4f}")

    pd.DataFrame({"class": LABELS, "intra_class_cohesion": [intra[l] for l in LABELS]}
                 ).to_csv("diagnostic_C_intraclass_cohesion_v2_results.csv", index=False)
    print("\nOK, Saved diagnostic_C_intraclass_cohesion_v2_results.csv")


def wa_subclustering(df, vectors):
    print("\n" + "*" * 70)
    print("  3. Weakly Aligned sub-clustering (KMeans, LIVE computation)")
    print("*" * 70)
    wa_mask = (df["gold_label"] == "Weakly Aligned").to_numpy()
    wa_ids = df.loc[wa_mask, "item_id"].to_numpy()
    wa_vecs = vectors[wa_mask]
    design_subtype = np.array(["WA-D" if i in WA_D_IDS else "WA-F" for i in wa_ids])

    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    cluster_labels = km.fit_predict(wa_vecs)

    # orient cluster 0/1 to whichever majority-aligns-with WA-D/WA-F for a readable crosstab
    crosstab = pd.crosstab(pd.Series(design_subtype, name="design_subtype"),
                           pd.Series(cluster_labels, name="cluster"))
    print("\nCrosstab (design subtype x KMeans cluster):")
    print(crosstab)

    # for each cluster, take majority design_subtype, sum majority counts / n
    n = len(wa_ids)
    purity = sum(crosstab.max(axis=0)) / n
    ari = adjusted_rand_score(design_subtype, cluster_labels)
    nmi = normalized_mutual_info_score(design_subtype, cluster_labels)
    print(f"\nPurity: {purity:.4f}")
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")

    for st in ["WA-D", "WA-F"]:
        st_mask = design_subtype == st
        st_clusters = cluster_labels[st_mask]
        majority_cluster = pd.Series(st_clusters).mode()[0]
        crossover_ids = wa_ids[st_mask][st_clusters != majority_cluster]
        print(f"{st} items landing in the OTHER majority cluster: {list(crossover_ids)}")

    # per-cluster intra-cluster cohesion
    sim_matrix = wa_vecs @ wa_vecs.T
    for c in sorted(set(cluster_labels)):
        idx = np.where(cluster_labels == c)[0]
        m = len(idx)
        sub = sim_matrix[np.ix_(idx, idx)]
        mean_intra = (sub.sum() - np.trace(sub)) / (m * (m - 1)) if m > 1 else float("nan")
        dominant_subtype = pd.Series(design_subtype[idx]).mode()[0]
        print(f"Cluster {c} (n={m}, majority={dominant_subtype}): intra-cluster cohesion = {mean_intra:.4f}")

    detail = pd.DataFrame({
        "item_id": wa_ids, "design_subtype": design_subtype, "kmeans_cluster": cluster_labels,
    })
    detail.to_csv("wa_subclustering_v2_detail.csv", index=False)
    print("\n OK, Saved wa_subclustering_v2_detail.csv")


def main():
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f"OK, Loaded {len(df)} rows from {DATASET_PATH}")
    assert df["item_text"].str.contains("Explain:").sum() == 0, "v2 dataset should have no 'Explain:' left!"

    print(f" Loading {MODEL_PATH}")
    model = SentenceTransformer(MODEL_PATH, local_files_only=True)
    print(" Encoding all item texts")
    vectors = model.encode(df["item_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)

    near_duplicate_screen(df, vectors)
    cohesion(df, vectors)
    wa_subclustering(df, vectors)


if __name__ == "__main__":
    main()
