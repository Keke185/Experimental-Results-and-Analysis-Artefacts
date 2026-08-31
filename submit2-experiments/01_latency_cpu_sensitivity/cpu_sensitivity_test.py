import os
import sys
import csv
import time
import pandas as pd
import httpx

URL = "http://localhost:8002"
SAMPLE_SIZE = 20
DELAY_MS = 5
CPU_LABEL = os.environ.get("CPU_LABEL", "unknown")
RESULTS_CSV = "cpu_sensitivity_results.csv"

print(f"DEBUG, script starting. python={sys.executable}  cwd={os.getcwd()}")

df_items = pd.read_csv("dataset_items.csv")
df_items.columns = [c.strip() for c in df_items.columns]
items = df_items.head(SAMPLE_SIZE)[["item_id", "item_text"]].to_dict(orient="records")

rows = []

with httpx.Client(trust_env=False) as client:
    health = client.get(f"{URL}/health", timeout=10).json()
    print(f"OK, connected. node_role={health.get('node_role')}  CPU_LABEL={CPU_LABEL}")

    latencies = []
    for row in items:
        time.sleep(DELAY_MS / 1000.0)
        t0 = time.perf_counter()
        r = client.post(f"{URL}/match/check",
                        json={"item_text": row["item_text"], "item_id": row["item_id"]},
                        timeout=60)
        r.raise_for_status()
        body = r.json()
        t1 = time.perf_counter()
        end_to_end_ms = DELAY_MS + (t1 - t0) * 1000
        latencies.append(end_to_end_ms)
        print(f"  {row['item_id']}: end_to_end={end_to_end_ms:.1f}ms  "
              f"server_embedding={body.get('embedding_ms'):.1f}ms")

        rows.append({
            "cpu_label": CPU_LABEL,
            "item_id": row["item_id"],
            "simulated_delay_ms": DELAY_MS,
            "end_to_end_ms": end_to_end_ms,
            "server_embedding_ms": body.get("embedding_ms"),
            "server_similarity_ms": body.get("similarity_ms"),
        })

    s = pd.Series(latencies)
    print(f"\n RESULT: CPU_LABEL={CPU_LABEL}  n={len(s)}  mean={s.mean():.1f}ms  "
          f"median={s.median():.1f}ms  min={s.min():.1f}ms  max={s.max():.1f}ms  std={s.std():.1f}ms")

print(f"DEBUG, finished the 'with httpx.Client' block normally. rows_collected={len(rows)}")

# Append to shared results CSV file (headers are only written to the new file)
print(f"DEBUG, about to write CSV. cwd={os.getcwd()}  RESULTS_CSV(abs)={os.path.abspath(RESULTS_CSV)}")
try:
    file_exists = os.path.isfile(RESULTS_CSV)
    print(f"DEBUG, file_exists={file_exists}")
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"OK, Appended {len(rows)} rows to {RESULTS_CSV}")
except Exception:
    import traceback

    print("ERROR, failed while writing CSV:")
    traceback.print_exc()

print("DEBUG, script reached the end")
