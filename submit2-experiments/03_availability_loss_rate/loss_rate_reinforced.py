import sys
import time
import socket
import csv
import os
import uuid
import subprocess
import requests
import pandas as pd

CONTAINER = "vcse_edge_sandboxed"
HOST = "localhost"
PORT = 8002
URL = f"http://{HOST}:{PORT}/match/check"

DATASET_PATH = "dataset_items_final.csv"
N_REQUESTS = 30
TIMEOUT_SEC = 10
OUT_PATH = "loss_rate_reinforced.csv"


def load_items(n):
    df = pd.read_csv(DATASET_PATH)
    df.columns = [c.strip() for c in df.columns]
    base = df.head(20).to_dict("records")
    return [base[i % len(base)] for i in range(n)]


def read_cgroup_stat():
    for path in ["/sys/fs/cgroup/cpu.stat", "/sys/fs/cgroup/cpu/cpu.stat"]:
        try:
            out = subprocess.run(["docker", "exec", CONTAINER, "cat", path],
                                  capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                stats = {}
                for line in out.stdout.strip().splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[1].isdigit():
                        stats[parts[0]] = int(parts[1])
                stats["_source_path"] = path
                return stats
        except Exception:
            continue
    return {}


def tcp_connect_time(host, port, timeout):
    t0 = time.perf_counter()
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        elapsed = (time.perf_counter() - t0) * 1000
        s.close()
        return round(elapsed, 1), ""
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return round(elapsed, 1), type(e).__name__


def main():
    if len(sys.argv) != 3:
        print("Usage: python loss_rate_reinforced.py")
        sys.exit(1)
    cpu_label = sys.argv[1]
    phase = sys.argv[2]

    items = load_items(N_REQUESTS)
    cgroup_before = read_cgroup_stat()

    results = []
    for i, item in enumerate(items):
        req_id = str(uuid.uuid4())[:8]
        connect_ms, connect_err = tcp_connect_time(HOST, PORT, TIMEOUT_SEC)

        t0 = time.perf_counter()
        status = "success"
        error_type = ""
        try:
            resp = requests.post(URL, json={"item_id": item["item_id"], "item_text": item["item_text"]}, timeout=TIMEOUT_SEC)
            elapsed = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
        except requests.exceptions.ConnectTimeout:
            elapsed = (time.perf_counter() - t0) * 1000
            status, error_type = "fail", "connect_timeout"
        except requests.exceptions.ReadTimeout:
            elapsed = (time.perf_counter() - t0) * 1000
            status, error_type = "fail", "read_timeout"
        except requests.exceptions.ConnectionError:
            elapsed = (time.perf_counter() - t0) * 1000
            status, error_type = "fail", "connection_error"
        except Exception as e:
            elapsed = (time.perf_counter() - t0) * 1000
            status, error_type = "fail", type(e).__name__

        results.append({
            "cpu_label": cpu_label, "phase": phase, "request_index": i + 1, "request_id": req_id,
            "item_id": item["item_id"], "status": status, "error_type": error_type,
            "tcp_connect_ms": connect_ms, "tcp_connect_error": connect_err,
            "total_elapsed_ms": round(elapsed, 1),
        })
        print(f"  req{i+1} id={req_id} status={status} error={error_type} connect={connect_ms}ms total={elapsed:.1f}ms")

    cgroup_after = read_cgroup_stat()
    nr_periods_delta = cgroup_after.get("nr_periods", 0) - cgroup_before.get("nr_periods", 0)
    nr_throttled_delta = cgroup_after.get("nr_throttled", 0) - cgroup_before.get("nr_throttled", 0)
    throttled_time_delta = cgroup_after.get("throttled_time", 0) - cgroup_before.get("throttled_time", 0)

    write_header = not os.path.exists(OUT_PATH)
    with open(OUT_PATH, "a", newline="") as f:
        fieldnames = ["cpu_label", "phase", "request_index", "request_id", "item_id", "status", "error_type",
                      "tcp_connect_ms", "tcp_connect_error", "total_elapsed_ms",
                      "cgroup_nr_periods_delta", "cgroup_nr_throttled_delta", "cgroup_throttled_time_delta_ns"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in results:
            r["cgroup_nr_periods_delta"] = nr_periods_delta
            r["cgroup_nr_throttled_delta"] = nr_throttled_delta
            r["cgroup_throttled_time_delta_ns"] = throttled_time_delta
            writer.writerow(r)

    n_success = sum(1 for r in results if r["status"] == "success")
    print(f"\n cpu={cpu_label} phase={phase}: {n_success}/{len(results)} succeeded "
          f"cgroup delta: nr_periods={nr_periods_delta} nr_throttled={nr_throttled_delta} throttled_time_ns={throttled_time_delta} "
          f"cgroup stat source: {cgroup_before.get('_source_path', 'NOT AVAILABLE')}")


if __name__ == "__main__":
    main()