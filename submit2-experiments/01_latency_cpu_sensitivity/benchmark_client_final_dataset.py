

import time
import statistics
import httpx
import pandas as pd

CONFIGS = {
    "cloud (pure-cloud baseline, WAN)": {
        "url": "http://localhost:8000",
        "delay_levels_ms": [20, 50, 100, 200],
    },
    "edge_unsandboxed (LAN, no runtime limits)": {
        "url": "http://localhost:8001",
        "delay_levels_ms": [1, 5, 10],
    },
    "edge_sandboxed (LAN, CPU/mem capped + restricted net)": {
        "url": "http://localhost:8002",
        "delay_levels_ms": [1, 5, 10],
    },
}
DATASET_PATH = "dataset_items_final.csv"
SAMPLE_SIZE = 50


def check_health(client: httpx.Client, base_url: str) -> dict:
    r = client.get(f"{base_url}/health", timeout=10)
    r.raise_for_status()
    return r.json()


def run_one_config(name: str, base_url: str, delay_levels_ms, items: list[dict]) -> list[dict]:
    records = []
    with httpx.Client(trust_env=False) as client:
        try:
            health = check_health(client, base_url)
            print(f"OK, {name}: reachable, role_id={health.get('role_id')}, "
                  f"artifact_version={health.get('artifact_version')}")
        except Exception as e:
            print(f"SKIP, {name}: not reachable at {base_url} ({e}). "
                  f"Is the container running and the port published?")
            return records
        for delay_ms in delay_levels_ms:
            print(f" {name} @ simulated network delay = {delay_ms} ms "
                  f"({len(items)} items)")
            for row in items:
                time.sleep(delay_ms / 1000.0)
                t0 = time.perf_counter()
                try:
                    resp = client.post(
                        f"{base_url}/match/check",
                        json={"item_text": row["item_text"], "item_id": row["item_id"]},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    body = resp.json()
                    ok = True
                except Exception as e:
                    body = {}
                    ok = False
                    print(f"      request failed for {row['item_id']}: {e}")

                t1 = time.perf_counter()
                client_total_ms = (t1 - t0) * 1000
                records.append({
                    "config": name,
                    "simulated_delay_ms": delay_ms,
                    "item_id": row["item_id"],
                    "ok": ok,
                    "predicted_label": body.get("predicted_label"),
                    "similarity_score": body.get("similarity_score"),
                    "server_embedding_ms": body.get("embedding_ms"),
                    "server_similarity_ms": body.get("similarity_ms"),
                    "server_total_processing_ms": body.get("total_processing_ms"),
                    "client_http_round_trip_ms": client_total_ms,
                    "end_to_end_ms": delay_ms + client_total_ms,
                })

    return records


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    def p95(s):
        return s.quantile(0.95)

    def p99(s):
        return s.quantile(0.99)

    ok_df = df[df["ok"]]
    summary = ok_df.groupby(["config", "simulated_delay_ms"])["end_to_end_ms"].agg(
        mean="mean", median="median", std="std", p95=p95, p99=p99, min="min", max="max", n="count"
    ).reset_index()
    return summary


def main():
    print("*" * 70)
    print("  Deployment-Configuration Latency Benchmark -FINAL DATASET")
    print("*" * 70)
    df_items = pd.read_csv(DATASET_PATH)
    df_items.columns = [c.strip() for c in df_items.columns]
    items = df_items.head(SAMPLE_SIZE)[["item_id", "item_text"]].to_dict(orient="records")
    print(f"\n Using {len(items)} items per (config, delay level) combination "
          f"(SAMPLE_SIZE={SAMPLE_SIZE}), from {DATASET_PATH}.")

    all_records = []
    for name, cfg in CONFIGS.items():
        recs = run_one_config(name, cfg["url"], cfg["delay_levels_ms"], items)
        all_records.extend(recs)

    if not all_records:
        print("\n No records collected - none of the configured services were reachable")
        return

    df = pd.DataFrame(all_records)
    df.to_csv("benchmark_raw_results_final_dataset.csv", index=False)
    print(f"\n OK, Raw per-request results saved to benchmark_raw_results_final_dataset.csv "
          f"({len(df)} rows, {int(df['ok'].sum())} successful)")

    summary = summarise(df)
    summary.to_csv("benchmark_summary_final_dataset.csv", index=False)
    print("\n Summary (end-to-end latency, ms) by configuration and simulated delay level:")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(summary.round(2))
    print("\nOK, Summary saved to benchmark_summary_final_dataset.csv")


if __name__ == "__main__":
    main()
