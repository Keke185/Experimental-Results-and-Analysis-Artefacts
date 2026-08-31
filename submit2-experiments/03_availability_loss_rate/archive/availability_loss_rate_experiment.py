import csv
import subprocess
import time

import requests


BASE_URL = "http://localhost:8002"
CONTAINER_NAME = "vcse_edge_sandboxed"
DATASET_PATH = "dataset_items.csv"


CPU_LEVELS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
N_REQUESTS_PER_LEVEL = 15
TIMEOUT_S = 10.0 # Timeout
SETTLE_S = 3.0 # Pause for 3.0 seconds before sending the request

def set_container_cpus(cpu: float) -> bool:

    try:
        subprocess.run(
            ["docker", "update", "--cpus", str(cpu), CONTAINER_NAME],
            check=True, capture_output=True, text=True,
        )
        return True

    except Exception as e:
        print(f"`docker update --cpus {cpu}` failed: {e} Fall back to manually restarting the sandboxed "
              f"container with the new --cpus value,"
              f"wait for it to become healthy, then press Enter to continue.Press Enter once ready")
        return False

def read_retransmit_count() -> int:

    try:
        out = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "cat", "/proc/net/snmp"],
            check=True, capture_output=True, text=True,
        ).stdout
        lines = out.splitlines()
        header = next(l for l in lines if l.startswith("Tcp:") and "RetransSegs" in l)
        values = next(l for l in lines if l.startswith("Tcp:") and "RetransSegs" not in l)
        fields = header.split()
        vals = values.split()
        idx = fields.index("RetransSegs")
        return int(vals[idx])
    except Exception as e:
        print(f"Could not read TCP retransmit counter ({e}); reporting -1 for this level")
        return -1


def load_sample_texts(n=5):
    try:
        import csv as _csv
        with open(DATASET_PATH, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            rows = [r["item_text"] for r in reader if r.get("item_text")]
        if rows:
            return rows[:n] if len(rows) >= n else rows
    except Exception as e:
        print(f" Could not load {DATASET_PATH} ({e}); using a fixed dummy payload instead")
    return ["Explain how you would design a scalable video conferencing signalling architecture"]


def send_one_request(session, text, item_id):
    payload = {"item_text": text, "item_id": item_id}
    t0 = time.perf_counter()
    try:
        resp = session.post(f"{BASE_URL}/match/check", json=payload, timeout=TIMEOUT_S)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            return "success", elapsed_ms
        else:
            return "http_error", elapsed_ms
    except requests.exceptions.Timeout:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return "timeout", elapsed_ms
    except requests.exceptions.ConnectionError:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return "connection_error", elapsed_ms


def main():
    print("*" * 70)
    print("  Availability / Request-Loss-Rate Experiment")
    print(f"  target={BASE_URL}  container={CONTAINER_NAME}  timeout={TIMEOUT_S}s  n/level={N_REQUESTS_PER_LEVEL}")
    print("*" * 70)

    texts = load_sample_texts(N_REQUESTS_PER_LEVEL)
    session = requests.Session()

    all_rows = []
    summary_rows = []

    for cpu in CPU_LEVELS:
        print(f"\n Setting CPU limit to {cpu} cores ")
        set_container_cpus(cpu)
        time.sleep(SETTLE_S)

        #Check health status
        try:
            h = session.get(f"{BASE_URL}/health", timeout=TIMEOUT_S)
            print(f"    /health -> {h.status_code}")
        except Exception as e:
            print(f"    [!] /health check failed before burst: {e}")

        retrans_before = read_retransmit_count()

        outcomes = {"success": 0, "timeout": 0, "connection_error": 0, "http_error": 0}
        latencies_success = []

        print(f"    Sending {N_REQUESTS_PER_LEVEL} requests (timeout={TIMEOUT_S}s each)")

        for i in range(N_REQUESTS_PER_LEVEL):
            text = texts[i % len(texts)]
            outcome, elapsed_ms = send_one_request(session, text, f"cpu{cpu}_req{i}")
            outcomes[outcome] += 1
            if outcome == "success":
                latencies_success.append(elapsed_ms)
            all_rows.append({"cpu": cpu, "request_index": i, "outcome": outcome, "elapsed_ms": round(elapsed_ms, 1)})

            print(f"      req {i+1}/{N_REQUESTS_PER_LEVEL}: {outcome} ({elapsed_ms:.0f} ms)")

        retrans_after = read_retransmit_count()
        retrans_delta = (retrans_after - retrans_before) if (retrans_before >= 0 and retrans_after >= 0) else -1

        total = N_REQUESTS_PER_LEVEL
        success_rate = outcomes["success"] / total
        loss_rate = (outcomes["timeout"] + outcomes["connection_error"]) / total
        mean_latency = sum(latencies_success) / len(latencies_success) if latencies_success else float("nan")

        summary_rows.append({
            "cpu_cores": cpu,
            "n_requests": total,
            "success": outcomes["success"],
            "timeout": outcomes["timeout"],
            "connection_error": outcomes["connection_error"],
            "http_error": outcomes["http_error"],
            "success_rate": round(success_rate, 3),
            "loss_rate_timeout_or_conn_error": round(loss_rate, 3),
            "mean_latency_success_ms": round(mean_latency, 1) if latencies_success else None,
            "tcp_retransmit_delta": retrans_delta,
        })

        print(f"    OK, CPU={cpu}: success_rate={success_rate:.1%}  loss_rate={loss_rate:.1%}  "
              f"mean_latency(success only)={mean_latency:.0f}ms  tcp_retransmit_delta={retrans_delta}")

    print("\n" + "*" * 90)
    print("  Summary: availability / loss rate by CPU level")
    print("*" * 90)
    header = f"{'CPU':>6}{'N':>5}{'success':>9}{'timeout':>9}{'conn_err':>10}{'http_err':>10}{'succ_rate':>11}{'loss_rate':>11}{'mean_ms(ok)':>13}{'tcp_retrans_Δ':>15}"
    print(header)

    for r in summary_rows:
        print(f"{r['cpu_cores']:>6}{r['n_requests']:>5}{r['success']:>9}{r['timeout']:>9}{r['connection_error']:>10}"
              f"{r['http_error']:>10}{r['success_rate']:>11.1%}{r['loss_rate_timeout_or_conn_error']:>11.1%}"
              f"{(r['mean_latency_success_ms'] if r['mean_latency_success_ms'] is not None else 0):>13.0f}"
              f"{r['tcp_retransmit_delta']:>15}")

    with open("availability_loss_rate_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with open("availability_loss_rate_raw.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nOK, Saved availability_loss_rate_summary.csv and availability_loss_rate_raw.csv")

if __name__ == "__main__":
    main()
