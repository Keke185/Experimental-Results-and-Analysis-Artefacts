"""
Hybrid semantic + keyword matching.
"""
import os
import pickle
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

LABELS_ORDER = ["Aligned", "Weakly Aligned", "Mismatched"]
LABEL_TO_CODE = {lbl: i for i, lbl in enumerate(LABELS_ORDER)}


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def min_max_normalise(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def macro_f1_and_acc(pred_code: np.ndarray, true_code: np.ndarray):
    idx = true_code * 3 + pred_code
    cm = np.bincount(idx, minlength=9).reshape(3, 3)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred_totals = cm.sum(axis=0)
    precision = np.where(pred_totals > 0, tp / np.maximum(pred_totals, 1), 0.0)
    recall = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    acc = tp.sum() / true_code.shape[0]
    return f1.mean(), acc


def best_thresholds_for(scores: np.ndarray, true_code: np.ndarray, th_candidates: np.ndarray):
    best = None
    for weak_th in th_candidates:
        below_weak = scores < weak_th
        for aligned_th in th_candidates:
            if aligned_th <= weak_th:
                continue
            above_aligned = scores >= aligned_th
            pred_code = np.where(above_aligned, 0, np.where(~below_weak, 1, 2))
            macro_f1, acc = macro_f1_and_acc(pred_code, true_code)
            if best is None or macro_f1 > best[0]:
                best = (macro_f1, acc, weak_th, aligned_th)
    return best


def classify_score(score: float, aligned_th: float, weak_th: float) -> str:
    if score >= aligned_th:
        return "Aligned"
    elif score >= weak_th:
        return "Weakly Aligned"
    else:
        return "Mismatched"


def report(name: str, scores: np.ndarray, y_true, true_code: np.ndarray, th_candidates: np.ndarray):
    macro_f1, acc, weak_th, aligned_th = best_thresholds_for(scores, true_code, th_candidates)
    y_pred = [classify_score(s, aligned_th, weak_th) for s in scores]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS_ORDER, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS_ORDER)
    print(f"\n--- {name} ---")
    print(f"  best thresholds: weakly_aligned={weak_th:.3f}, aligned={aligned_th:.3f}")
    print(f"  macro-F1={macro_f1:.4f}, accuracy={acc:.4f}")
    for lbl, p, r, f, s in zip(LABELS_ORDER, precision, recall, f1, support):
        print(f"    {lbl:15s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  n={s}")
    print(f"  confusion matrix (rows=true, cols=pred), order={LABELS_ORDER}")
    print(" ", str(cm).replace("\n", "\n  "))
    return macro_f1, acc, weak_th, aligned_th


def main():
    print("*" * 50)
    print("  Hybrid Semantic + Keyword Matching")
    print("*" * 50)

    artifact_path = "/kaggle/working/distribution.pkl"
    dataset_path = "/kaggle/input/datasets/kehuang5/dataset-items/dataset_items.csv"
    model_path = '/kaggle/input/models/srg9000/all-minilm-l6-v2/transformers/default/1/all-MiniLM-L6-v2'

    print(f"\n Loading cloud-generated artifact: {artifact_path}")
    with open(artifact_path, "rb") as f:
        artifact = pickle.load(f)
    capability_vectors = np.asarray(artifact["capability_vectors"])
    capability_chunks = artifact["capability_chunks"]
    print(f"OK, Artifact loaded. Capability blocks: {capability_vectors.shape[0]}")

    print(f"\n Loading semantic model: {artifact['metadata']['model_identifier']}")
    model = SentenceTransformer(model_path, local_files_only=True)
    print("OK, Model loaded.")

    print(f"\n Loading evaluation dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    df.columns = [c.strip() for c in df.columns]
    print(f"OK, Loaded {len(df)} evaluation items")

    # Semantic score
    print("\n Encoding evaluation items and computing semantic max-similarity")
    item_vectors = np.asarray(model.encode(df["item_text"].tolist(), show_progress_bar=True))
    sem_sim = cosine_similarity_matrix(item_vectors, capability_vectors)
    semantic_score = sem_sim.max(axis=1)

    # Keyword score
    print(" Computing TF-IDF keyword max-similarity")
    vectorizer = TfidfVectorizer(stop_words="english")
    corpus = list(capability_chunks) + df["item_text"].tolist()
    X = vectorizer.fit_transform(corpus)
    chunk_tfidf = X[: len(capability_chunks)]
    item_tfidf = X[len(capability_chunks):]
    kw_sim = sk_cosine_similarity(item_tfidf, chunk_tfidf)
    keyword_score = kw_sim.max(axis=1)

    df["semantic_score"] = semantic_score
    df["keyword_score"] = keyword_score

    semantic_norm = min_max_normalise(semantic_score)
    keyword_norm = min_max_normalise(keyword_score)

    y_true = df["gold_label"]
    true_code = y_true.map(LABEL_TO_CODE).to_numpy()
    th_candidates = np.round(np.arange(0.0, 1.01, 0.02), 3)

    report("Semantic only (normalised)", semantic_norm, y_true, true_code, th_candidates)
    report("Keyword only (normalised)", keyword_norm, y_true, true_code, th_candidates)

    print("\n Grid-searching fusion weight alpha (hybrid = alpha*semantic + (1-alpha)*keyword) "
          "and thresholds to maximise macro-F1")
    t0 = time.time()
    alphas = np.round(np.arange(0.0, 1.01, 0.05), 2)
    best_overall = None
    for alpha in alphas:
        hybrid_score = alpha * semantic_norm + (1 - alpha) * keyword_norm
        macro_f1, acc, weak_th, aligned_th = best_thresholds_for(hybrid_score, true_code, th_candidates)
        print(f"    alpha={alpha:.2f}  macro-F1={macro_f1:.4f}  accuracy={acc:.4f}")
        if best_overall is None or macro_f1 > best_overall[0]:
            best_overall = (macro_f1, acc, weak_th, aligned_th, alpha)
    print(f" Grid search finished in {time.time() - t0:.1f}s")

    macro_f1, acc, weak_th, aligned_th, alpha = best_overall
    hybrid_score = alpha * semantic_norm + (1 - alpha) * keyword_norm
    df["hybrid_score"] = hybrid_score
    df["predicted_label"] = [classify_score(s, aligned_th, weak_th) for s in hybrid_score]

    print(f"\n OK, Best hybrid configuration: alpha={alpha} "
          f"(semantic weight={alpha}, keyword weight={1 - alpha:.2f})")
    print(f"     thresholds: weakly_aligned={weak_th:.3f}, aligned={aligned_th:.3f}")
    print(f"     macro-F1={macro_f1:.4f}, accuracy={acc:.4f}")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, df["predicted_label"], labels=LABELS_ORDER, zero_division=0
    )
    cm = confusion_matrix(y_true, df["predicted_label"], labels=LABELS_ORDER)
    print("\n Per-class metrics at best hybrid configuration:")
    for lbl, p, r, f, s in zip(LABELS_ORDER, precision, recall, f1, support):
        print(f"  {lbl:15s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  n={s}")
    print(f"\n Confusion Matrix (rows=true, cols=pred), label order = {LABELS_ORDER}")
    print(cm)

    out_path = "hybrid_matching_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\n OK, Full per-item results saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
