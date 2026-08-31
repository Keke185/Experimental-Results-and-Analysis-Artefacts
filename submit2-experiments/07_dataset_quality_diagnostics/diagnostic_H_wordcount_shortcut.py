"""
Diagnosis H: Word count shortcut detection. Diagnosis E: Statistical analysis revealed systematic
differences in the average word count across different sample types (weak alignment ~45.1,
aligned ~38.2, mismatch ~34.0). This script trains a classifier using only word count as a feature
to quantitatively verify whether word count constitutes a spurious shortcut in the model. If the macro
F1 score of the word count classifier is significantly higher than most baselines, then the dataset
contains spurious word count signals; if the performance is close to the baseline, then the length difference
is merely noise. A 10-fold hierarchical KFold + logistic regression method, consistent with the decision logic
experiment, is used, and the results can be directly compared with the 9-feature classifier metrics.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.dummy import DummyClassifier

DATASET_PATH = "/kaggle/input/datasets/kehuang5/dataset-items/dataset_items.csv"

LABELS = ["Aligned", "Weakly Aligned", "Mismatched"]
RANDOM_STATE = 42
N_SPLITS = 10


def main():
    print("*" * 70)
    print("  Diagnostic H: Word-count shortcut detection")
    print("*" * 70)

    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f"OK. Loaded {len(df)} items")

    df["_word_count"] = df["item_text"].apply(lambda t: len(str(t).split()))
    print(f"\n{'Class':<20}{'mean words':>12}{'std':>10}{'n':>8}")
    print("*" * 50)
    for label in LABELS:
        sub = df.loc[df["gold_label"] == label, "_word_count"]
        print(f"{label:<20}{sub.mean():>12.1f}{sub.std():>10.1f}{len(sub):>8}")

    X = df[["_word_count"]].to_numpy(dtype=float)
    y = df["gold_label"].to_numpy()

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    dummy = DummyClassifier(strategy="most_frequent")
    dummy_pred = cross_val_predict(dummy, X, y, cv=skf)
    dummy_f1 = f1_score(y, dummy_pred, average="macro", zero_division=0)
    dummy_acc = accuracy_score(y, dummy_pred)

    # word-count-only classifier
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE))
    pred = cross_val_predict(clf, X, y, cv=skf)
    f1 = f1_score(y, pred, average="macro", zero_division=0)
    acc = accuracy_score(y, pred)

    print("\n" + "*" * 70)
    print("  10-fold CV results (out-of-sample)")
    print("*" * 70)
    print(f"{'Method':<38}{'Macro-F1':>12}{'Accuracy':>12}")
    print("-" * 70)
    print(f"{'Majority-class baseline':<38}{dummy_f1:>12.4f}{dummy_acc:>12.4f}")
    print(f"{'Word-count-ONLY classifier':<38}{f1:>12.4f}{acc:>12.4f}")
    print(f"{'(reference) 9-feature real classifier':<38}{'0.75~0.89':>12}{'--':>12}")

    print("\nPer-class report (word-count-only classifier):")
    print(classification_report(y, pred, labels=LABELS, zero_division=0))

    print("Confusion matrix (rows=true, cols=predicted), label order:", LABELS)
    print(confusion_matrix(y, pred, labels=LABELS))

    out = pd.DataFrame({
        "method": ["majority_baseline", "wordcount_only_classifier"],
        "macro_f1": [dummy_f1, f1],
        "accuracy": [dummy_acc, acc],
    })

    out.to_csv("diagnostic_H_wordcount_shortcut_results.csv", index=False)
    print("\nOK, Saved diagnostic_H_wordcount_shortcut_results.csv")


if __name__ == "__main__":
    main()
