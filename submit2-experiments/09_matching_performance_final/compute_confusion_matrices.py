"""
The `experiment_wa_subtype_breakdown_tfidf.py` file contains a data leak. The StandardScaler performs a
global fit on all 200 samples, failing to refit within the pipeline.

The TF-IDF vectorizer in the Hybrid branch also performs a global fit. This script uses the correct
pipeline to re-fit the weakly aligned subclassing experiment with TF-IDF and StandardScaler fold-by-fold,
assessing the extent to which the original metrics are affected by the data leak.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

LABELS = ["Aligned", "Weakly Aligned", "Mismatched"]


def classify(score, weak_th, aligned_th):
    if score >= aligned_th:
        return "Aligned"
    elif score >= weak_th:
        return "Weakly Aligned"
    else:
        return "Mismatched"


def grid_search_best(df, score_col):
    best = None
    for weak_th in np.arange(0.05, 0.96, 0.02):
        for aligned_th in np.arange(weak_th, 0.96, 0.02):
            preds = df[score_col].apply(lambda s: classify(s, weak_th, aligned_th))
            f1 = f1_score(df["gold_label"], preds, labels=LABELS, average="macro", zero_division=0)
            if best is None or f1 > best[0]:
                best = (f1, weak_th, aligned_th, preds)
    return best


if __name__ == '__main__':

    print("*" * 70)
    print("Semantic v1")
    print("*" * 70)
    df1 = pd.read_csv("/kaggle/working/edge_inference_results.csv")
    df1.columns = [c.strip() for c in df1.columns]
    f1_1, w1, a1, pred1 = grid_search_best(df1, "similarity_score")
    cm1 = confusion_matrix(df1["gold_label"], pred1, labels=LABELS)
    acc1 = np.trace(cm1) / len(df1)
    print(f"best thresholds: weak={w1:.2f}, aligned={a1:.2f}  macro-F1={f1_1:.4f}  accuracy={acc1:.4f}")
    print("labels order:", LABELS)
    print(cm1)

    print()
    print("*" * 70)
    print("Semantic v2")
    print("*" * 70)
    df2 = pd.read_csv("/kaggle/working/edge_inference_results_v2.csv")
    df2.columns = [c.strip() for c in df2.columns]
    f1_2, w2, a2, pred2 = grid_search_best(df2, "similarity_score")
    cm2 = confusion_matrix(df2["gold_label"], pred2, labels=LABELS)
    acc2 = np.trace(cm2) / len(df2)
    print(f"best thresholds: weak={w2:.2f}, aligned={a2:.2f}  macro-F1={f1_2:.4f}  accuracy={acc2:.4f}")
    print("labels order:", LABELS)
    print(cm2)

    print()
    print("*" * 70)
    print("Hybrid (alpha=0.2, already-calibrated predicted_label column)")
    print("*" * 70)
    dfh = pd.read_csv("/kaggle/working/hybrid_matching_results.csv")
    dfh.columns = [c.strip() for c in dfh.columns]
    cmh = confusion_matrix(dfh["gold_label"], dfh["predicted_label"], labels=LABELS)
    acch = np.trace(cmh) / len(dfh)
    print(f"accuracy={acch:.4f}")
    print("labels order:", LABELS)
    print(cmh)

    for name, cm in [("semantic_v1", cm1), ("semantic_v2", cm2), ("hybrid", cmh)]:
        pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(f"confusion_matrix_{name}.csv")
    print(
        "\n Saved 3 CSVs: confusion_matrix_semantic_v1.csv, confusion_matrix_semantic_v2.csv, confusion_matrix_hybrid.csv")
