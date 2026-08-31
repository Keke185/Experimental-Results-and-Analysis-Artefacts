"""
Quick latency SANITY CHECK on the re-frozen dataset (dataset_items_final_v2.csv)
"""
import time
import statistics
import requests
import pandas as pd

DATASET_PATH = "dataset_items_final_v2.csv"
N_PER_CLASS = 10
WARMUP_REQUESTS = 3

ENDPOINTS = {
    "cloud":            "http://localhost:8000/match/check",
    "edge_unsandboxed":  "http://localhost:8001/match/check",
}


def sample_items(path, n_per_class):

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    sampled = (
        df.groupby("gold_label", group_keys=False)
        .apply(lambda g: g.sample(n=min(n_per_class, len(g)), random_state=42))
    )

    return sampled[["item_id", "item_text"]].to_dict("records")


def hit_endpoint(url, item, timeout=30):

    t0 = time.perf_counter()
    resp = requests.post(url, json={"item_id": item["item_id"], "item_text": item["item_text"]}, timeout=timeout)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()

    return elapsed_ms, resp.json()


def summarize(latencies_ms):

    arr = sorted(latencies_ms)
    n = len(arr)
    mean_ = statistics.mean(arr)
    median_ = statistics.median(arr)
    p95 = arr[min(int(0.95 * n), n - 1)]

    return {"n": n, "mean_ms": round(mean_, 2), "median_ms": round(median_, 2),
            "p95_ms": round(p95, 2), "min_ms": round(min(arr), 2), "max_ms": round(max(arr), 2)}


def main():

    items = sample_items(DATASET_PATH, N_PER_CLASS)
    print(f"Sampled {len(items)} items from {DATASET_PATH} "
          f"({N_PER_CLASS} per class, seeded for reproducibility)")

    results = {}
    for name, url in ENDPOINTS.items():
        print(f"\n Warming up {name} ({url})")
        for item in items[:WARMUP_REQUESTS]:
            try:
                hit_endpoint(url, item)
            except Exception as e:
                print(f" warmup request failed: {e}")

        print(f"Benchmarking {name}")
        latencies = []
        errors = 0
        for item in items:
            try:
                elapsed_ms, _ = hit_endpoint(url, item)
                latencies.append(elapsed_ms)
            except Exception as e:
                errors += 1
                print(f"  WARN! request failed for {item['item_id']}: {e}")
        if latencies:
            stats = summarize(latencies)
            results[name] = stats
            print(f"  {name}: {stats}  (errors={errors}/{len(items)})")
        else:
            print(f"  FAIL! all requests failed for {name} -- is the service running on that port?")

    print("\n" + "*" * 70)
    print("  SUMMARY (v2 dataset sanity check)")
    print("*" * 70)
    for name, stats in results.items():
        print(f"  {name:<20} mean={stats['mean_ms']:>8}ms  median={stats['median_ms']:>8}ms  "
              f"p95={stats['p95_ms']:>8}ms  n={stats['n']}")

    print("\n  Compare these numbers to your original benchmark_summary.csv from the \n"
          "    v1 (pre-cleanup) dataset run -- differences should be small (a few ms), \n"
          "    since only 20/200 items changed and only by removing a short label. \n"
          "    If a config here is much slower/faster than before, that's worth a \n"
          "    closer look; otherwise this just confirms nothing broke.")

    pd.DataFrame(results).T.to_csv("benchmark_summary_v2_sanity_check.csv")
    print("\n OK,Saved benchmark_summary_v2_sanity_check.csv")


if __name__ == "__main__":
    main()
