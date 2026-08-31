"""
Targeted retesting for steady-state latency anomalies in the 0.5 to 0.75 core range
"""

import sys
import time
import csv
import os
import requests
import pandas as pd

DATASET_PATH = "dataset_items_final.csv"
N_ITEMS = 20
SANDBOXED_EDGE_URL = "http://localhost:8002/match/check"
SIMULATED_DELAY_MS = 5
WARMUP_REQUESTS = 3
OUT_PATH = "retest_0.5to1.0_results.csv"


def load_items():
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]

    # Same selection as the original run: first 20 rows of data
    first20 = df.head(N_ITEMS)

    return first20[["item_id", "item_text"]].to_dict("records")


def hit_once(item, timeout=30):
    t0 = time.perf_counter()
    resp = requests.post(
        SANDBOXED_EDGE_URL,
        json={"item_id": item["item_id"], "item_text": item["item_text"]},
        timeout=timeout,
    )
    end_to_end_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    body = resp.json()
    return end_to_end_ms, body.get("embedding_ms"), body.get("similarity_ms")


def main():
    if len(sys.argv) != 2:
        print("Usage: python retest_cpu_anomaly_client.py <cpu_label>")
        sys.exit(1)
    cpu_label = sys.argv[1]

    items = load_items()
    print(f" Loaded {len(items)} items ({[i['item_id'] for i in items]})")

    print(f" Warming up sandboxed edge at cpu={cpu_label}")
    for item in items[:WARMUP_REQUESTS]:
        try:
            hit_once(item)
        except Exception as e:
            print(f"  WARN! warmup request failed: {e}")

    print(f" Benchmarking cpu={cpu_label} ")
    rows = []
    for item in items:
        try:
            end_to_end_ms, embedding_ms, similarity_ms = hit_once(item)
            rows.append({
                "cpu_label": cpu_label,
                "item_id": item["item_id"],
                "simulated_delay_ms": SIMULATED_DELAY_MS,
                "end_to_end_ms": end_to_end_ms,
                "server_embedding_ms": embedding_ms,
                "server_similarity_ms": similarity_ms,
            })
            print(f"  {item['item_id']}: {end_to_end_ms:.1f} ms")
        except Exception as e:
            print(f"  WARN! request failed for {item['item_id']}: {e}")

    if not rows:
        print("FAIL! All requests failed. Is the sandbox container running on this port?")
        sys.exit(1)

    write_header = not os.path.exists(OUT_PATH)
    with open(OUT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    mean_ms = sum(r["end_to_end_ms"] for r in rows) / len(rows)
    print(f"\n OK! cpu={cpu_label}: mean end_to_end_ms={mean_ms:.1f} over n={len(rows)}, appended to {OUT_PATH}")


if __name__ == "__main__":
    main()
