import sys
import os
import csv
import requests
import pandas as pd

DATASET_PATH = "dataset_items_final.csv"
N_WARMUP = 3
OUT_PATH = "decision_logic_e2e_results.csv"


def load_items(n):
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    base = df.head(20).to_dict("records")
    return [base[i % len(base)] for i in range(n)]

def main():
    label = sys.argv[1]
    port = int(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    url = f"http://localhost:{port}/match/check"

    items = load_items(n + N_WARMUP)

    for i in range(N_WARMUP):
        try:
            requests.post(url, json={"item_id": items[i]["item_id"], "item_text": items[i]["item_text"]}, timeout=10)
        except Exception:
            pass

    rows = []
    for i in range(n):
        item = items[N_WARMUP + i]
        try:
            resp = requests.post(url, json={"item_id": item["item_id"], "item_text": item["item_text"]}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            rows.append({
                "label": label, "request_index": i + 1, "item_id": item["item_id"],
                "embedding_ms": data["embedding_ms"], "similarity_ms": data["similarity_ms"],
                "total_processing_ms": data["total_processing_ms"], "predicted_label": data["predicted_label"],
            })
        except Exception as e:
            rows.append({
                "label": label, "request_index": i + 1, "item_id": item["item_id"],
                "embedding_ms": "", "similarity_ms": "", "total_processing_ms": "", "predicted_label": f"ERROR:{e}",
            })

    write_header = not os.path.exists(OUT_PATH)
    with open(OUT_PATH, "a", newline="") as f:
        fieldnames = ["label", "request_index", "item_id", "embedding_ms", "similarity_ms", "total_processing_ms", "predicted_label"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    valid = [r["total_processing_ms"] for r in rows if r["total_processing_ms"] != ""]
    if valid:
        print(f"{label}: n={len(valid)} mean_total_ms={sum(valid)/len(valid):.3f}")
    else:
        print(f"{label}: all requests failed")


if __name__ == "__main__":
    main()
