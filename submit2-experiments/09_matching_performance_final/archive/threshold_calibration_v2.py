import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

LABELS_ORDER = ["Aligned", "Weakly Aligned", "Mismatched"]


def classify_score(score: float, aligned_th: float, weak_th: float) -> str:
    if score >= aligned_th:
        return "Aligned"
    elif score >= weak_th:
        return "Weakly Aligned"
    else:
        return "Mismatched"


def main():
    results_path = "/kaggle/working/edge_inference_results_v2.csv"

    print("*" * 50)
    print("  Threshold Calibration v2 - Empirical Analysis")
    print("  (specialised + general technical foundations profile)")
    print("*" * 50)

    df = pd.read_csv(results_path)
    y_true = df["gold_label"]
    scores = df["similarity_score"].values

    print("\n Similarity score distribution by gold_label:")
    stats = df.groupby("gold_label")["similarity_score"].describe()
    print(stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]])

    print("\n Grid-searching threshold pairs (weakly_aligned_threshold < aligned_threshold) "
          "to maximise macro-F1")
    candidates = np.round(np.arange(0.05, 0.95, 0.01), 2)

    best = None
    for weak_th, aligned_th in product(candidates, candidates):
        if aligned_th <= weak_th:
            continue
        y_pred = [classify_score(s, aligned_th, weak_th) for s in scores]
        acc = accuracy_score(y_true, y_pred)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS_ORDER, zero_division=0
        )
        macro_f1 = f1.mean()
        if best is None or macro_f1 > best[0]:
            best = (macro_f1, acc, weak_th, aligned_th)

    macro_f1, acc, weak_th, aligned_th = best
    print(f"\nOK, Best thresholds found on this labelled set:")
    print(f"     weakly_aligned_threshold = {weak_th}")
    print(f"     aligned_threshold        = {aligned_th}")
    print(f"     -> macro-F1 = {macro_f1:.4f}, accuracy = {acc:.4f}")
    print("     (this is an oracle upper bound for the current max-similarity strategy;")
    print("      treat it as a calibration target, not a final held-out test result)")

    y_pred = [classify_score(s, aligned_th, weak_th) for s in scores]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS_ORDER, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS_ORDER)

    print("\n Per-class metrics at best thresholds:")
    for lbl, p, r, f, s in zip(LABELS_ORDER, precision, recall, f1, support):
        print(f"  {lbl:15s}  P={p:.3f}  R={r:.3f}  F1={f:.3f}  n={s}")

    print(f"\n Confusion Matrix (rows=true, cols=pred), label order = {LABELS_ORDER}")
    print(cm)

    y_pred_orig = [classify_score(s, 0.50, 0.30) for s in scores]
    acc_orig = accuracy_score(y_true, y_pred_orig)
    _, _, f1_orig, _ = precision_recall_fscore_support(
        y_true, y_pred_orig, labels=LABELS_ORDER, zero_division=0
    )
    print(f"\n Reference, Original placeholder thresholds (0.30 / 0.50): "
          f"accuracy = {acc_orig:.4f}, macro-F1 = {f1_orig.mean():.4f}")


if __name__ == "__main__":
    main()
