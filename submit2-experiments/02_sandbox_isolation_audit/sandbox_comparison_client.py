"""
    Client for the controlled sandboxed-vs-unsandboxed latency comparison.
"""

import sys
import time
import csv
import os
import requests
import pandas as pd

DATASET_PATH = "dataset_items_final.csv"
N_REQUESTS = 20          # first 20 items
WARMUP_REQUESTS = 3
TIMEOUT_SEC = 15
OUT_PATH_DEFAULT = "sandbox_comparison_clean_rerun.csv"


def load_items(n):
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df.head(n)[["item_id", "item_text"]].to_dict("records")


def main():
    if len(sys.argv) < 3:
        print("Usage: python sandbox_comparison_client.py")
        sys.exit(1)
    label = sys.argv[1]
    port = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else OUT_PATH_DEFAULT
    url = f"http://localhost:{port}/match/check"

    items = load_items(N_REQUESTS)
    print(f" OK,Loaded {len(items)} items for label={label} url={url}")

    for i, item in enumerate(items[:WARMUP_REQUESTS]):
        try:
            requests.post(url, json={"item_id": item["item_id"], "item_text": item["item_text"]}, timeout=TIMEOUT_SEC)
        except Exception as e:
            print(f"  WARN, warmup request {i+1} failed: {e}")

    results = []
    for i, item in enumerate(items):
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                url,
                json={"item_id": item["item_id"], "item_text": item["item_text"]},
                timeout=TIMEOUT_SEC,
            )

            elapsed = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            results.append({"label": label, "request_index": i + 1, "item_id": item["item_id"],
                             "status": "success", "elapsed_ms": round(elapsed, 1)})
            print(f"  req{i+1} {item['item_id']}: OK {elapsed:.1f}ms")

        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            results.append({"label": label, "request_index": i + 1, "item_id": item["item_id"],
                             "status": "timeout_or_error", "elapsed_ms": round(elapsed, 1)})
            print(f"  req{i+1} {item['item_id']}: FAIL ({e}) after {elapsed:.1f}ms")

    write_header = not os.path.exists(out_path)
    with open(out_path, "a", newline="") as f:
        fieldnames = ["label", "request_index", "item_id", "status", "elapsed_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in results:
            writer.writerow(r)

    ok = [r["elapsed_ms"] for r in results if r["status"] == "success"]

    if ok:
        mean_ = sum(ok) / len(ok)
        print(f"\nOK, label={label}: {len(ok)}/{len(results)} succeeded, mean={mean_:.1f}ms, appended to {out_path}")
    else:
        print(f"\nFAIL, label={label}: all requests failed")


if __name__ == "__main__":
    main()
