"""

Perform latency testing on the final dataset using a random sampling system.

"""
import random
import time

import pandas as pd
import requests

DATASET_PATH = "dataset_items_final.csv"

EDGE_URL = "http://localhost:8001/match/check"
CLOUD_URL = "http://localhost:8000/match/check"

N_PER_CLASS = 10          # stratified sample size of gold_label
SEED = 42
WARMUP_REQUESTS = 3

def sample_items(df: pd.DataFrame) -> pd.DataFrame:

    random.seed(SEED)
    parts = []
    for label, group in df.groupby("gold_label"):
        n = min(N_PER_CLASS, len(group))
        parts.append(group.sample(n=n, random_state=SEED))
    return pd.concat(parts).reset_index(drop=True)


def call_endpoint(url: str, item_id: str, item_text: str) -> dict:
    t0 = time.perf_counter()
    resp = requests.post(url, json={"item_id": item_id, "item_text": item_text}, timeout=30)
    wall_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    body = resp.json()

    return {
        "wall_clock_ms": wall_ms,
        "server_total_processing_ms": body["total_processing_ms"],
        "embedding_ms": body["embedding_ms"],
        "similarity_ms": body["similarity_ms"],
        "predicted_label": body["predicted_label"],
        "node_role": body["node_role"],
    }


def run_pass(url: str, node_label: str, sample: pd.DataFrame) -> pd.DataFrame:
    print(f"\n Warming up {node_label} ({WARMUP_REQUESTS} untimed requests)")
    for _, row in sample.head(WARMUP_REQUESTS).iterrows():
        try:
            call_endpoint(url, row["item_id"], row["item_text"])
        except Exception as e:
            print(f"  WARN! warm-up call failed: {e}")

    print(f" Timed run: {len(sample)} requests against {node_label} ({url})")
    rows = []
    for _, row in sample.iterrows():
        try:
            result = call_endpoint(url, row["item_id"], row["item_text"])
        except Exception as e:
            print(f"  ERROR! {row['item_id']} failed: {e}")

            continue
        result.update({
            "item_id": row["item_id"],
            "gold_label": row["gold_label"],
            "char_count": len(row["item_text"]),
            "node": node_label,
        })

        rows.append(result)

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, node_label: str):

    print(f"\n {node_label} latency summary (n={len(df)})")
    for col in ["wall_clock_ms", "server_total_processing_ms", "embedding_ms", "similarity_ms"]:
        s = df[col]

        print(f"  {col:<28} mean={s.mean():7.2f}ms  median={s.median():7.2f}ms  "
              f"p95={s.quantile(0.95):7.2f}ms  min={s.min():7.2f}ms  max={s.max():7.2f}ms")
    print("\n  By gold_label:")

    print(df.groupby("gold_label")[["wall_clock_ms", "server_total_processing_ms"]].mean().round(2))


def main():
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    sample = sample_items(df)

    print(f" OK! Sampled {len(sample)} items "
          f"({sample['gold_label'].value_counts().to_dict()})")

    all_results = []

    try:
        edge_df = run_pass(EDGE_URL, "edge", sample)
        summarize(edge_df, "edge")
        all_results.append(edge_df)
    except Exception as e:
        print(f"WARN! edge pass skipped/failed: {e}")

    try:
        cloud_df = run_pass(CLOUD_URL, "cloud", sample)
        summarize(cloud_df, "cloud")
        all_results.append(cloud_df)
    except Exception as e:
        print(f"WARN! cloud pass skipped/failed: {e}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv("latency_spotcheck_final_dataset.csv", index=False)
        print("\n OK, Saved latency_spotcheck_final_dataset.csv")
    else:
        print("\n FAIL! No successful requests , check that your Docker services are up "
              "and EDGE_URL/CLOUD_URL match your docker-compose.yml port mappings")

if __name__ == "__main__":
    main()
