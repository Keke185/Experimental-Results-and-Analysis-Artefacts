import numpy as np

LABELS = ["Aligned", "Weakly Aligned", "Mismatched"]

MATRICES = {
    ("TF-IDF", "threshold (fair CV, leak-fixed)"): [[38, 7, 5], [3, 4, 43], [9, 16, 75]],
    ("TF-IDF", "classifier (CV, leak-fixed)"):      [[46, 1, 3], [5, 34, 11], [9, 33, 58]],
    ("MiniLM", "threshold (fair CV)"): [[30, 1, 19], [0, 0, 50], [4, 8, 88]],
    ("MiniLM", "classifier (CV)"):     [[41, 4, 5], [1, 41, 8], [5, 10, 85]],
    ("BGE-large-v1.5", "threshold (fair CV)"): [[47, 3, 0], [16, 34, 0], [95, 5, 0]],
    ("BGE-large-v1.5", "classifier (CV)"):     [[42, 2, 6], [2, 42, 6], [4, 7, 89]],
    ("Hybrid", "threshold (fair CV) [PENDING leak-fix rerun]"): [[31, 6, 13], [2, 0, 48], [8, 5, 87]],
    ("Hybrid", "classifier (CV) [PENDING leak-fix rerun]"):     [[45, 1, 4], [2, 41, 7], [2, 13, 85]],
}

def per_class_metrics(cm):
    cm = np.array(cm, dtype=float)
    tp = np.diag(cm)
    support = cm.sum(axis=1)
    pred_totals = cm.sum(axis=0)
    precision = np.where(pred_totals > 0, tp / np.maximum(pred_totals, 1), 0.0)
    recall = np.where(support > 0, tp / np.maximum(support, 1), 0.0)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    acc = tp.sum() / cm.sum()
    macro_f1 = f1.mean()
    return precision, recall, f1, support, acc, macro_f1


def main():
    print(f"{'Method':<16}{'Logic':<38}{'Class':<16}{'Prec':>7}{'Rec':>7}{'F1':>7}{'Support':>9}")
    print("-" * 100)
    rows = []
    for (method, logic), cm in MATRICES.items():
        precision, recall, f1, support, acc, macro_f1 = per_class_metrics(cm)
        for i, cls in enumerate(LABELS):
            print(f"{method:<16}{logic:<38}{cls:<16}{precision[i]:>7.3f}{recall[i]:>7.3f}{f1[i]:>7.3f}{int(support[i]):>9}")
            rows.append((method, logic, cls, precision[i], recall[i], f1[i], int(support[i])))
        print(f"{'':<16}{'':<38}{'ACC / macro-F1':<16}{acc:>7.3f}{'':>7}{macro_f1:>7.3f}")
        print("-" * 100)

    print("\n    High-risk trap leakage: Mismatched items misclassified as Aligned    ")
    for (method, logic), cm in MATRICES.items():
        cm = np.array(cm)
        mismatched_to_aligned = cm[2, 0]
        mismatched_total = cm[2].sum()
        print(f"  {method:<16}{logic:<38} {mismatched_to_aligned}/{mismatched_total} "
              f"({mismatched_to_aligned/mismatched_total*100:.1f}%)")


if __name__ == "__main__":
    main()
