#!/usr/bin/env python3
"""
Driver script that:
1) Runs all 3 models once with a demo input and shows their console output.
2) Benchmarks all 3 models over multiple (income, deduction) pairs and
   plots bar charts of:
      - avg wall-clock time (user experience)
      - avg logical messages sent (no hardcoding)
      - avg bytes on wire (no hardcoding)
"""

import os
import sys
import time
import json
import subprocess
from statistics import mean

import matplotlib.pyplot as plt

PYTHON = sys.executable
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CONFIGS = {
    "ts": {
        "name": "TenSEAL FHE",
        "server": os.path.join(ROOT_DIR, "Model_A", "server_ts.py"),
        "client": os.path.join(ROOT_DIR, "Model_A", "client_ts.py"),
        "needs_client_id": False,
    },
    "phe": {
        "name": "Paillier (bands on client)",
        "server": os.path.join(ROOT_DIR, "Model_B", "server_phe.py"),
        "client": os.path.join(ROOT_DIR, "Model_B", "client_phe.py"),
        "needs_client_id": False,   # <-- IMPORTANT: your current client_phe does not ask for ID
    },
    "e2e": {
        "name": "E2E Fernet",
        "server": os.path.join(ROOT_DIR, "e2e", "server_e2e.py"),
        "client": os.path.join(ROOT_DIR, "e2e", "client_e2e.py"),
        "needs_client_id": False,
    },
}


def _parse_metrics(text: str):
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("__METRICS__ "):
            try:
                return json.loads(line[len("__METRICS__ "):].strip())
            except Exception:
                return None
    return None


def run_one_model(model_key, income, deductions, show_output=True):
    """
    Returns dict:
      {
        "elapsed": float or None,
        "wire": {"messages": int, "bytes": int} or None
      }
    """
    cfg = MODEL_CONFIGS[model_key]
    server_script = cfg["server"]
    client_script = cfg["client"]

    if not os.path.isfile(server_script):
        raise FileNotFoundError(f"Server script not found: {server_script}")
    if not os.path.isfile(client_script):
        raise FileNotFoundError(f"Client script not found: {client_script}")

    env_server = dict(os.environ, MODEL_KEY=model_key, ROLE="server")
    env_client = dict(os.environ, MODEL_KEY=model_key, ROLE="client")

    # Start server
    server_proc = subprocess.Popen(
        [PYTHON, server_script],
        stdout=subprocess.PIPE if not show_output else None,
        stderr=subprocess.PIPE if not show_output else None,
        text=True,
        env=env_server,
    )

    time.sleep(0.5)

    # Build piped input (no hardcoding packet counts; only input prompts)
    if cfg["needs_client_id"]:
        client_input = f"AutoClient\n{income}\n{deductions}\n"
    else:
        client_input = f"{income}\n{deductions}\n"

    start = time.perf_counter()
    client_proc = subprocess.run(
        [PYTHON, client_script],
        input=client_input,
        capture_output=not show_output,
        text=True,
        env=env_client,
    )
    end = time.perf_counter()

    elapsed = end - start

    # Collect server output if we captured it
    server_out = ""
    server_err = ""
    if not show_output:
        try:
            # Most of your servers exit after 1 client; give them a moment.
            server_out, server_err = server_proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                server_proc.terminate()
            except Exception:
                pass
            server_out, server_err = server_proc.communicate(timeout=5)
    else:
        # In demo mode, server stdout is live; just stop it after the run.
        try:
            server_proc.terminate()
        except Exception:
            pass

    # If client crashed → return None elapsed (and print error)
    if client_proc.returncode != 0:
        print(f"[ERROR] Client for model {model_key} exited with code {client_proc.returncode}")
        if client_proc.stderr:
            print("---- client stderr ----")
            print(client_proc.stderr)
            print("---- end stderr ----")
        return {"elapsed": None, "wire": None}

    # Show output in demo mode
    if show_output:
        if client_proc.stdout:
            print(client_proc.stdout)
        if client_proc.stderr:
            print(client_proc.stderr, file=sys.stderr)
        return {"elapsed": elapsed, "wire": None}

    # Benchmark mode: parse wire metrics from both client + server
    cm = _parse_metrics((client_proc.stdout or "") + "\n" + (client_proc.stderr or ""))
    sm = _parse_metrics((server_out or "") + "\n" + (server_err or ""))

    if not cm or not sm:
        wire = None
    else:
        # Count "messages on wire" as number of sends (not double-counting receives)
        messages = int(cm.get("packets_out", 0)) + int(sm.get("packets_out", 0))
        bytes_on_wire = int(cm.get("bytes_out", 0)) + int(sm.get("bytes_out", 0))
        wire = {"messages": messages, "bytes": bytes_on_wire}

    return {"elapsed": elapsed, "wire": wire}


def demo_run():
    print("=== DEMO RUN: income=91_000, deductions=5_000 ===\n")
    income = 91_000
    deductions = 5_000

    for key in ["ts", "phe", "e2e"]:
        name = MODEL_CONFIGS[key]["name"]
        print(f"\n================== {name} ==================\n")
        res = run_one_model(key, income, deductions, show_output=True)
        print(f"[{name}] Total wall-clock time (client start → finish): {res['elapsed']:.3f} seconds\n")


def benchmark_runs():
    print("\n=== BENCHMARK: running multiple inputs for all models (quiet) ===\n")

    test_cases = [
        (30_000, 0),
        (60_000, 5_000),
        (90_000, 10_000),
        (150_000, 15_000),
    ]

    times = {k: [] for k in MODEL_CONFIGS}
    msgs = {k: [] for k in MODEL_CONFIGS}
    bytes_ = {k: [] for k in MODEL_CONFIGS}

    for income, deductions in test_cases:
        print(f"  Running test case income={income}, deductions={deductions} ...")
        for key in MODEL_CONFIGS:
            res = run_one_model(key, income, deductions, show_output=False)
            if res["elapsed"] is not None:
                times[key].append(res["elapsed"])
            if res["wire"] is not None:
                msgs[key].append(res["wire"]["messages"])
                bytes_[key].append(res["wire"]["bytes"])

    ordered_keys = ["ts", "phe", "e2e"]
    labels = [MODEL_CONFIGS[k]["name"] for k in ordered_keys]

    avg_times = [mean(times[k]) if times[k] else float("nan") for k in ordered_keys]
    avg_msgs = [mean(msgs[k]) if msgs[k] else float("nan") for k in ordered_keys]
    avg_bytes = [mean(bytes_[k]) if bytes_[k] else float("nan") for k in ordered_keys]

    for k in ordered_keys:
        print(f"  {MODEL_CONFIGS[k]['name']}: "
              f"time_avg={mean(times[k]) if times[k] else float('nan'):.3f}s, "
              f"msgs_avg={mean(msgs[k]) if msgs[k] else float('nan'):.1f}, "
              f"bytes_avg={(mean(bytes_[k]) if bytes_[k] else float('nan'))/1024:.1f}KB")

    # 1) Time chart
    plt.figure(figsize=(9, 4))
    bars = plt.bar(labels, avg_times)
    plt.ylabel("Avg total time (seconds)")
    plt.title("Average user experience time")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, t in zip(bars, avg_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{t:.3f}s", ha="center", va="bottom")
    plt.tight_layout()
    plt.show()

    # 2) Messages chart
    plt.figure(figsize=(9, 4))
    bars = plt.bar(labels, avg_msgs)
    plt.ylabel("Average total logical messages (sends)")
    plt.title("Average total number of logical messages")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, m in zip(bars, avg_msgs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{m:.1f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.show()

    # 3) Bytes chart
    plt.figure(figsize=(9, 4))
    bars = plt.bar(labels, [b/1024 for b in avg_bytes])
    plt.ylabel("Avg bytes on wire (KB)")
    plt.title("Average bytes on wire (application-layer)")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, b in zip(bars, avg_bytes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{b/1024:.1f}KB", ha="center", va="bottom")
    plt.tight_layout()
    plt.show()


def main():
    demo_run()
    benchmark_runs()


if __name__ == "__main__":
    main()
