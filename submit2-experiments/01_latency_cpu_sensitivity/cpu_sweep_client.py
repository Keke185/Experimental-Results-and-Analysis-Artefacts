"""
Client for the clean, single-pass re-run of the full steady-state CPU
sensitivity sweep (0.1 -> 6.0 cores, 10 levels), on the frozen dataset.

This replaces both the original cpu_sensitivity_results.csv run (which had a reporting error at 0.5
cores and a since-unreproduced anomaly at 0.75 cores) and the earlier 0.5~1.0-only retest
(which used a different container state and landed on a different absolute scale) with ONE internally
consistent dataset, collected the same way at every level, so the report and thesis don't need to explain two
batches with an offset between them
"""
import sys
import time
import csv
import os
import requests
import pandas as pd

DATASET_PATH = "dataset_items_final.csv"   # dataset
N_ITEMS = 20
SANDBOXED_EDGE_URL = "http://localhost:8002/match/check"
SIMULATED_DELAY_MS = 5
WARMUP_REQUESTS = 3
DEFAULT_OUT_PATH = "cpu_sensitivity_clean_rerun.csv"


def load_items():
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]

    #Select the first 20 rows of data
    first20 = df.head(N_ITEMS)
    return first20[["item_id", "item_text"]].to_dict("records")


def hit_once(item, timeout=60):

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
    if len(sys.argv) < 2:
        print("Usage: python cpu_sweep_client.py <cpu_label> [output_csv]")
        sys.exit(1)
    cpu_label = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT_PATH

    items = load_items()
    print(f" OK, Loaded {len(items)} items ({[i['item_id'] for i in items]})")

    print(f" Warming up sandboxed edge at cpu={cpu_label}")
    for item in items[:WARMUP_REQUESTS]:
        try:
            hit_once(item)
        except Exception as e:
            print(f"  WARN! warmup request failed: {e}")

    print(f" Benchmarking cpu={cpu_label}")
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
        print("FAIL! all requests failed , is the sandboxed container healthy?")
        sys.exit(1)

    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    mean_ms = sum(r["end_to_end_ms"] for r in rows) / len(rows)
    print(f"\n OK, cpu={cpu_label}: mean end_to_end_ms={mean_ms:.1f} over n={len(rows)}, appended to {out_path}")


if __name__ == "__main__":
    main()
