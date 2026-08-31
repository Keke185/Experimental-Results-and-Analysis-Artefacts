import numpy as np
from scipy import stats

FOLD_F1 = {
    ("TF-IDF", "threshold"): [0.5349, 0.4848, 0.4541, 0.6333, 0.5282, 0.5165, 0.3818, 0.4127, 0.4212, 0.6894],
    ("TF-IDF", "classifier"): [0.7608, 0.6528, 0.7236, 0.7633, 0.8047, 0.6239, 0.5476, 0.8553, 0.5934, 0.6007],
    ("MiniLM", "threshold"): [0.5067, 0.5363, 0.4633, 0.5527, 0.3086, 0.453, 0.4633, 0.3615, 0.4808, 0.4969],
    ("MiniLM", "classifier"): [0.8158, 0.7807, 0.8565, 0.9074, 0.718, 0.8553, 0.7646, 0.7231, 0.8993, 0.9],
    ("BGE", "threshold"): [0.4259, 0.3182, 0.4, 0.3667, 0.4434, 0.4785, 0.3354, 0.3354, 0.455, 0.3354],
    ("BGE", "classifier"): [0.9327, 0.7231, 0.8054, 0.8643, 0.9, 0.8993, 0.8956, 0.9471, 0.7894, 0.8253],
    ("Hybrid", "threshold"): [0.5833, 0.5091, 0.5067, 0.5924, 0.5064, 0.4127, 0.4453, 0.4343, 0.358, 0.6894],
    ("Hybrid", "classifier"): [0.9327, 0.8418, 0.8565, 0.9024, 0.8487, 0.8037, 0.7919, 0.8548, 0.7464, 0.7837],
}


def paired_report(name_a, arr_a, name_b, arr_b):
    a = np.array(arr_a)
    b = np.array(arr_b)
    diff = a - b
    n = len(diff)
    mean_diff = diff.mean()

    std_diff = diff.std(ddof=1)
    se_diff = std_diff / np.sqrt(n)
    t_stat, p_value = stats.ttest_rel(a, b)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    ci_low = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff

    sig = "YES (p<0.05, CI excludes 0)" if p_value < 0.05 and (ci_low > 0 or ci_high < 0) else "NOT at alpha=0.05"

    print(f"\n [{name_a} vs {name_b}] paired fold-wise differences ({name_a} - {name_b}):")
    print(f"  fold diffs: {np.round(diff, 4).tolist()}")
    print(f"  mean diff = {mean_diff:+.4f}, std = {std_diff:.4f}, n = {n}")
    print(f"  paired t-test: t({n-1}) = {t_stat:.3f}, p = {p_value:.5f}")
    print(f"  95% CI on mean diff (t-distribution, df={n-1}): [{ci_low:+.4f}, {ci_high:+.4f}]")
    print(f"  Significant at alpha=0.05: {sig}")

    return t_stat, p_value, ci_low, ci_high

def main():
    print("*" * 70)
    print("  A. Classifier vs. threshold comparison, within each representation (pairwise comparison)")
    print("*" * 70)

    for method in ["TF-IDF", "MiniLM", "BGE", "Hybrid"]:
        paired_report(f"{method} classifier", FOLD_F1[(method, "classifier")],
                       f"{method} threshold", FOLD_F1[(method, "threshold")])

    print("\n" + "*" * 70)
    print("  B. Classifier vs. classifier comparison, across representations (pairwise comparison, same fold)")
    print("*" * 70)
    pairs = [
        ("MiniLM", "BGE"),
        ("MiniLM", "Hybrid"),
        ("BGE", "Hybrid"),
        ("TF-IDF", "MiniLM"),
        ("TF-IDF", "BGE"),
        ("TF-IDF", "Hybrid"),
    ]

    for m1, m2 in pairs:
        paired_report(f"{m1} classifier", FOLD_F1[(m1, "classifier")],
                       f"{m2} classifier", FOLD_F1[(m2, "classifier")])

    print("\n" + "*" * 70)
    print("  Holm-Bonferroni correction for the 6 pairwise classifier comparisons (B)")
    print("*" * 70)

    labels = []
    raw_pvals = []

    for m1, m2 in pairs:
        a = np.array(FOLD_F1[(m1, "classifier")])
        b = np.array(FOLD_F1[(m2, "classifier")])
        _, p = stats.ttest_rel(a, b)
        raw_pvals.append(p)
        labels.append(f"{m1} vs {m2}")
    order = np.argsort(raw_pvals)
    m = len(raw_pvals)

    # Gradual Rejection Decision
    holm_reject = [False] * m
    for rank, idx in enumerate(order):
        threshold = 0.05 / (m - rank)
        if raw_pvals[idx] < threshold:
            holm_reject[idx] = True
        else:
            break


    holm_adj = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * raw_pvals[idx])
        running_max = max(running_max, adj)
        holm_adj[idx] = running_max

    print(f"  {'comparison':<28} {'raw p':>10} {'Holm-adj p':>12} {'Holm-significant':>18}")
    print("  " + "" * 70)

    for i, lbl in enumerate(labels):
        print(f"  {lbl:<28} {raw_pvals[i]:>10.5f} {holm_adj[i]:>12.4f} {str(holm_reject[i]):>18}")

    # Consistency check:the two formulations must always agree.
    for i in range(m):
        assert (holm_adj[i] < 0.05) == holm_reject[i], \
            f" Holm adjusted-p and step-down decisions disagree for {labels[i]}"
    print("\n Adjusted-p decisions match step-down rejection decisions")

if __name__ == "__main__":
    main()
