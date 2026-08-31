import subprocess
import threading
import time
import csv
import sys
import requests
import pandas as pd

CONTAINER = "vcse_edge_sandboxed"
URL = "http://localhost:8002/match/check"
DATASET_PATH = "dataset_items_final.csv"
DURATION_SEC = 30
CONCURRENCY = 8
SAMPLE_INTERVAL_SEC = 1

stop_flag = threading.Event()


def load_items():
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df.head(20)[["item_id", "item_text"]].to_dict("records")


def worker(items):
    i = 0
    while not stop_flag.is_set():
        item = items[i % len(items)]
        try:
            requests.post(URL, json={"item_id": item["item_id"], "item_text": item["item_text"]}, timeout=10)
        except Exception:
            pass
        i += 1

def sample_stats(rows):
    t0 = time.time()
    while not stop_flag.is_set():
        out = subprocess.run(
            ["docker", "stats", CONTAINER, "--no-stream", "--format", "{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}"],
            capture_output=True, text=True,
        ).stdout.strip()
        elapsed = round(time.time() - t0, 1)

        if out:
            parts = out.split("|")
            cpu_perc = parts[0].replace("%", "") if len(parts) > 0 else ""
            mem_usage = parts[1] if len(parts) > 1 else ""
            mem_perc = parts[2].replace("%", "") if len(parts) > 2 else ""
            rows.append({"elapsed_sec": elapsed, "cpu_perc": cpu_perc, "mem_usage": mem_usage, "mem_perc": mem_perc})

            print(f"t={elapsed}s  CPU={cpu_perc}%  Mem={mem_usage} ({mem_perc}%)")
        time.sleep(SAMPLE_INTERVAL_SEC)

def main():
    cpu_label = sys.argv[1] if len(sys.argv) > 1 else "current"
    trial = sys.argv[2] if len(sys.argv) > 2 else "1"
    out_csv = f"cpu_sustained_load_stats_{cpu_label}_trial{trial}.csv"

    items = load_items()
    print(f"cpu={cpu_label} trial={trial}: starting {CONCURRENCY} workers for {DURATION_SEC}s")

    workers = [threading.Thread(target=worker, args=(items,), daemon=True) for _ in range(CONCURRENCY)]

    for w in workers:
        w.start()

    rows = []
    sampler = threading.Thread(target=sample_stats, args=(rows,), daemon=True)
    sampler.start()

    time.sleep(DURATION_SEC)
    stop_flag.set()
    time.sleep(2)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["elapsed_sec", "cpu_perc", "mem_usage", "mem_perc"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    cpu_vals = [float(r["cpu_perc"]) for r in rows if r["cpu_perc"]]
    if cpu_vals:
        print(f"cpu={cpu_label} trial={trial}: samples={len(cpu_vals)} mean={sum(cpu_vals)/len(cpu_vals):.2f} max={max(cpu_vals):.2f} min={min(cpu_vals):.2f}")
    print(f"saved {out_csv}")


if __name__ == "__main__":
    main()