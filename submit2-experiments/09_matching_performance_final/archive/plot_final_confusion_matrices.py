"""
Confusion-matrix heatmaps for the final frozen dataset, using the fully
symmetric 10-fold CV numbers from experiment_rigorous_cv_comparison.py
"""
import numpy as np
import matplotlib.pyplot as plt

LABELS = ["Aligned", "Weakly\nAligned", "Mismatched"]


DATA = {
    ("TF-IDF", "threshold (fair CV)"): [[29, 14, 7], [2, 3, 45], [6, 9, 85]],
    ("TF-IDF", "classifier (CV)"):     [[44, 2, 4], [2, 37, 11], [6, 38, 56]],
    ("MiniLM", "threshold (fair CV)"): [[30, 1, 19], [0, 0, 50], [4, 8, 88]],
    ("MiniLM", "classifier (CV)"):     [[41, 4, 5], [1, 41, 8], [5, 10, 85]],
    ("BGE-large-v1.5", "threshold (fair CV)"): [[47, 3, 0], [16, 34, 0], [95, 5, 0]],
    ("BGE-large-v1.5", "classifier (CV)"):     [[42, 2, 6], [2, 42, 6], [4, 7, 89]],
    ("Hybrid", "threshold (fair CV)"): [[31, 6, 13], [2, 0, 48], [8, 5, 87]],
    ("Hybrid", "classifier (CV)"):     [[45, 1, 4], [2, 41, 7], [2, 13, 85]],
}

MACRO_F1 = {
    ("TF-IDF", "threshold (fair CV)"): 0.4876,
    ("TF-IDF", "classifier (CV)"): 0.7001,
    ("MiniLM", "threshold (fair CV)"): 0.4664,
    ("MiniLM", "classifier (CV)"): 0.8283,
    ("BGE-large-v1.5", "threshold (fair CV)"): 0.3970,
    ("BGE-large-v1.5", "classifier (CV)"): 0.8581,
    ("Hybrid", "threshold (fair CV)"): 0.4610,
    ("Hybrid", "classifier (CV)"): 0.8525,
}


def plot_one(ax, cm, title, f1):
    cm = np.array(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / row_sums
    ax.set_xticks(range(3)); ax.set_xticklabels(LABELS, fontsize=8)
    ax.set_yticks(range(3)); ax.set_yticklabels(LABELS, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    for i in range(3):
        for j in range(3):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i,j]*100:.0f}%)", ha="center", va="center",
                     fontsize=8, color=color)
    ax.set_title(f"{title}\nmacro-F1={f1:.4f}", fontsize=9)


def main():
    methods = ["TF-IDF", "MiniLM", "BGE-large-v1.5", "Hybrid"]
    logics = ["threshold (fair CV)", "classifier (CV)"]

    fig, axes = plt.subplots(4, 2, figsize=(9, 16))
    for i, method in enumerate(methods):
        for j, logic in enumerate(logics):
            ax = axes[i, j]
            cm = DATA[(method, logic)]
            f1 = MACRO_F1[(method, logic)]
            plot_one(ax, cm, f"{method} -- {logic}", f1)
    fig.suptitle("Final dataset -- old threshold logic vs new 9-dim classifier\n(fully symmetric 10-fold CV, rows=gold, cols=predicted, %=row-normalized)",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("cm_final_all_methods_grid.png", dpi=150)
    print("[OK] Saved cm_final_all_methods_grid.png")

    # Also save individual images per method
    for method in methods:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        for j, logic in enumerate(logics):
            cm = DATA[(method, logic)]
            f1 = MACRO_F1[(method, logic)]
            plot_one(axes[j], cm, f"{method} -- {logic}", f1)
        fig.tight_layout()
        safe_name = method.replace(" ", "_").replace("-", "_").lower()
        fname = f"cm_final_{safe_name}.png"
        fig.savefig(fname, dpi=150)
        print(f"OK, Saved {fname}")

    plt.close("all")


if __name__ == "__main__":
    main()
