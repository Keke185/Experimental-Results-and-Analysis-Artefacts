"""
    Client for the clean re-run of the availability/loss-rate experiment.
"""

import sys
import time
import csv
import os
import re
import subprocess
import requests
import pandas as pd

DATASET_PATH = "dataset_items_final.csv"
N_REQUESTS = 15
TIMEOUT_SEC = 10
URL = "http://localhost:8002/match/check"
OUT_PATH = "loss_rate_clean_rerun.csv"


def load_items(n):
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]

    return df.head(n)[["item_id", "item_text"]].to_dict("records")


def get_tcp_retransmits():
    """
        Resolves the "Retransmission Segment Count" counter from `netstat -s -p tcp`.
        Returns None if it is unavailable or cannot be resolved on this system
    """
    try:
        out = subprocess.run(
            ["netstat", "-s", "-p", "tcp"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        m = re.search(r"Segments Retransmitted\s*=\s*(\d+)", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def main():
    if len(sys.argv) != 3:
        print("Usage: python loss_rate_client.py")
        sys.exit(1)
    cpu_label = sys.argv[1]
    phase = sys.argv[2]

    items = load_items(N_REQUESTS)
    print(f"OK, Loaded {len(items)} items for cpu={cpu_label} phase={phase}")

    retrans_before = get_tcp_retransmits()
    results = []
    for i, item in enumerate(items):
        t0 = time.perf_counter()

        try:
            resp = requests.post(
                URL,
                json={"item_id": item["item_id"], "item_text": item["item_text"]},
                timeout=TIMEOUT_SEC,
            )

            elapsed = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            results.append({
                "cpu_label": cpu_label, "phase": phase, "request_index": i + 1,
                "item_id": item["item_id"], "status": "success",
                "elapsed_ms": round(elapsed, 1),
            })

            print(f"  req{i+1} {item['item_id']}: OK {elapsed:.0f}ms")

        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            results.append({
                "cpu_label": cpu_label, "phase": phase, "request_index": i + 1,
                "item_id": item["item_id"], "status": "timeout_or_error",
                "elapsed_ms": round(elapsed, 1),
            })
            print(f"  req{i+1} {item['item_id']}: FAIL ({e}) after {elapsed:.0f}ms")

    retrans_after = get_tcp_retransmits()
    retrans_delta = (retrans_after - retrans_before) \
        if (retrans_before is not None and retrans_after is not None) \
        else ""

    write_header = not os.path.exists(OUT_PATH)

    with open(OUT_PATH, "a", newline="") as f:

        fieldnames = ["cpu_label", "phase", "request_index", "item_id", "status", "elapsed_ms", "tcp_retransmit_delta_for_batch"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        for r in results:
            r["tcp_retransmit_delta_for_batch"] = retrans_delta
            writer.writerow(r)

    n_success = sum(1 for r in results if r["status"] == "success")

    print(f"\n OK, cpu={cpu_label} phase={phase}: {n_success}/{len(results)} succeeded, "
          f"TCP retransmit delta={retrans_delta}, appended to {OUT_PATH}")


if __name__ == "__main__":
    main()
