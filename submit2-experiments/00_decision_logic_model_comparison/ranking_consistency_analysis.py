"""
Ranking Consistency Analysis
"""
import pandas as pd
from scipy.stats import spearmanr, kendalltau


RESULTS_PATH = "/kaggle/working/hybrid_matching_results.csv"

GOLD_RANK = {"Aligned": 3, "Weakly Aligned": 2, "Mismatched": 1}

#Rating column to be evaluated
SCORE_COLUMNS = {
    "Semantic (SBERT)": "semantic_score",
    "Keyword (TF-IDF)": "keyword_score",
    "Hybrid (alpha=0.2)": "hybrid_score",
}


def analyse(df: pd.DataFrame, score_columns: dict):

    df = df.copy()
    df["gold_rank"] = df["gold_label"].map(GOLD_RANK)
    if df["gold_rank"].isna().any():
        bad = df.loc[df["gold_rank"].isna(), "gold_label"].unique()
        raise ValueError(f"Unrecognised gold_label values found: {bad}")

    print(f"{'Method':<22} {'Spearman rho':>13} {'p-value':>10} {'Kendall tau':>13} {'p-value':>10}")
    print("-" * 70)
    results = []
    for name, col in score_columns.items():
        if col not in df.columns:

            print(f"{name:<22}  (column '{col}' not found in this file -- skipped)")

            continue
        rho, p_rho = spearmanr(df[col], df["gold_rank"])
        tau, p_tau = kendalltau(df[col], df["gold_rank"])
        print(f"{name:<22} {rho:>13.4f} {p_rho:>10.2e} {tau:>13.4f} {p_tau:>10.2e}")
        results.append({"method": name, "spearman_rho": rho, "spearman_p": p_rho,
                         "kendall_tau": tau, "kendall_p": p_tau})

    return pd.DataFrame(results)


def main():
    print("*" * 70)
    print("  Ranking-Consistency Analysis (Spearman / Kendall vs. gold ranking)")
    print("*" * 70)
    print(f"\n Loading {RESULTS_PATH}")
    df = pd.read_csv(RESULTS_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f" Loaded {len(df)} items\n")

    results = analyse(df, SCORE_COLUMNS)

    out_path = "ranking_consistency_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\n Saved summary table to {out_path}")

if __name__ == "__main__":
    main()

