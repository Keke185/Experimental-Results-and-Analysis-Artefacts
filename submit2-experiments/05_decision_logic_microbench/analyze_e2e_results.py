import pandas as pd

df = pd.read_csv("decision_logic_e2e_results.csv")
df["total_processing_ms"] = pd.to_numeric(df["total_processing_ms"], errors="coerce")

g = df.groupby("label")["total_processing_ms"].agg(["count", "mean", "median", "std", lambda s: s.quantile(0.95)])
g.columns = ["n", "mean_ms", "median_ms", "sd_ms", "p95_ms"]
print(g.round(3).to_string())
