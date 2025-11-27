#!/usr/bin/env python3
"""
demo.py — Unified benchmark driver (tax calculator)

Runs 3 models:
  - TenSEAL BFV (FHE-style, interactive fixed comparisons)
  - Paillier (PHE, client does slicing; server sums)
  - E2E Fernet baseline (benchmark only)

Outputs ONE figure (single window) with 5 subplots:
  1) Time (wall clock, lower is better)
  2) Bytes on wire (application-layer, from __METRICS__ bytes_out, lower is better)
  3) Ciphertext size (raw serialized from __METRICS__ when available, lower is better)
  4) Key size (raw, lower is better)
  5) Computation error (abs(tax - expected), lower is better)

Notes:
- Bytes on wire is computed as: client.bytes_out + server.bytes_out (no hardcoded message counts).
- To avoid hanging forever: after each run, we ALWAYS terminate the server if it didn't exit on its own.
"""

import os
import sys
import time
import json
import math
import re
import csv
import subprocess
from statistics import mean
from typing import Dict, List, Optional, Tuple

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
        "needs_client_id": False,
    },
    "e2e": {
        "name": "E2E Fernet",
        "server": os.path.join(ROOT_DIR, "e2e", "server_e2e.py"),
        "client": os.path.join(ROOT_DIR, "e2e", "client_e2e.py"),
        "needs_client_id": False,
    },
}

# Tax law (same brackets across your demos)
_THRESH = (30_000, 60_000, 90_000)
_RATES = (1, 2, 3, 4)


def expected_progressive_tax(taxable: int) -> float:
    T1, T2, T3 = _THRESH
    r1, r2, r3, r4 = _RATES

    s1 = max(0, min(taxable, T1))
    s2 = max(0, min(max(taxable - T1, 0), T2 - T1))
    s3 = max(0, min(max(taxable - T2, 0), T3 - T2))
    s4 = max(0, taxable - T3)

    return (s1 * r1 + s2 * r2 + s3 * r3 + s4 * r4) / 100.0


def _parse_metrics(text: str) -> Optional[Dict]:
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("__METRICS__ "):
            try:
                return json.loads(line[len("__METRICS__ "):].strip())
            except Exception:
                return None
    return None


def _parse_tax(text: str) -> Optional[float]:
    if not text:
        return None

    # TenSEAL
    m = re.search(r"Total tax due \(server result\)\s*:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))

    # Paillier
    m = re.search(r"Tax from server \(after decrypt\)\s*:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))

    # E2E variants
    m = re.search(r"Total tax due\s*:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))

    m = re.search(r"'total_tax'\s*:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if m:
        return float(m.group(1))

    return None


def _nanmean(xs: List[Optional[float]]) -> float:
    vals = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return mean(vals) if vals else float("nan")


def _terminate_proc(proc: subprocess.Popen, grace_s: float = 2.0) -> None:
    """Terminate then kill a process if it refuses to exit (prevents hangs)."""
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
    except Exception:
        return
    t0 = time.time()
    while proc.poll() is None and (time.time() - t0) < grace_s:
        time.sleep(0.02)
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass


class AdvancedTaxPerformanceAnalyzer:
    def __init__(self):
        self.results_dir = os.path.join(ROOT_DIR, "results", "graphs")
        os.makedirs(self.results_dir, exist_ok=True)

        plt.rcParams.update({
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.25,
        })

    def run_one_model(self, model_key: str, income: int, deductions: int, quiet: bool = True) -> Dict:
        cfg = MODEL_CONFIGS[model_key]
        server_script = cfg["server"]
        client_script = cfg["client"]

        if not os.path.isfile(server_script):
            raise FileNotFoundError(f"Server script not found: {server_script}")
        if not os.path.isfile(client_script):
            raise FileNotFoundError(f"Client script not found: {client_script}")

        env_server = dict(os.environ, MODEL_KEY=model_key, ROLE="server")
        env_client = dict(os.environ, MODEL_KEY=model_key, ROLE="client")

        if cfg["needs_client_id"]:
            client_input = f"AutoClient\n{income}\n{deductions}\n"
        else:
            client_input = f"{income}\n{deductions}\n"

        expected_tax = expected_progressive_tax(max(income - deductions, 0))

        # Start server
        server_proc = subprocess.Popen(
            [PYTHON, server_script],
            stdout=subprocess.PIPE if quiet else None,
            stderr=subprocess.PIPE if quiet else None,
            text=True,
            env=env_server,
        )

        # tiny wait so server binds
        time.sleep(0.35)

        t0 = time.perf_counter()

        # Run client (blocking). This avoids leaving a client Popen around.
        if quiet:
            client_res = subprocess.run(
                [PYTHON, client_script],
                input=client_input,
                capture_output=True,
                text=True,
                env=env_client,
                timeout=120,
            )
            client_out = (client_res.stdout or "")
            client_err = (client_res.stderr or "")
        else:
            client_res = subprocess.run(
                [PYTHON, client_script],
                input=client_input,
                text=True,
                env=env_client,
                timeout=120,
            )
            client_out, client_err = "", ""

        t1 = time.perf_counter()
        elapsed = t1 - t0

        # Give server a SHORT grace period to exit on its own, then force-kill.
        server_out, server_err = "", ""
        if quiet:
            try:
                server_out, server_err = server_proc.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                _terminate_proc(server_proc, grace_s=1.5)
                try:
                    server_out, server_err = server_proc.communicate(timeout=2)
                except Exception:
                    server_out, server_err = "", ""
        else:
            # If not capturing, still make sure server is gone so script can exit.
            _terminate_proc(server_proc, grace_s=1.5)

        merged_client = (client_out or "") + "\n" + (client_err or "")
        merged_server = (server_out or "") + "\n" + (server_err or "")

        tax = _parse_tax(merged_client)
        error = abs(tax - expected_tax) if tax is not None else None

        cm = _parse_metrics(merged_client)
        sm = _parse_metrics(merged_server)

        # Bytes on wire: sum of bytes_out from both ends
        wire_bytes = None
        wire_messages = None
        if cm and sm:
            wire_messages = int(cm.get("packets_out", 0)) + int(sm.get("packets_out", 0))
            wire_bytes = int(cm.get("bytes_out", 0)) + int(sm.get("bytes_out", 0))

        key_bytes = int(cm.get("key_bytes")) if (cm and "key_bytes" in cm) else None

        ciphertext_bytes = None
        if cm:
            if "ciphertext_bytes_total" in cm:
                ciphertext_bytes = int(cm["ciphertext_bytes_total"])
            elif "ciphertext_bytes_out" in cm or "ciphertext_bytes_in" in cm:
                ciphertext_bytes = int(cm.get("ciphertext_bytes_out", 0)) + int(cm.get("ciphertext_bytes_in", 0))

        return {
            "model": model_key,
            "name": cfg["name"],
            "income": income,
            "deductions": deductions,
            "taxable": max(income - deductions, 0),
            "elapsed": elapsed,
            "wire_bytes": wire_bytes,
            "wire_messages": wire_messages,
            "key_bytes": key_bytes,
            "ciphertext_bytes": ciphertext_bytes,
            "tax": tax,
            "expected_tax": expected_tax,
            "error": error,
        }

    def demo_run(self):
        print("=== DEMO RUN: income=91_000, deductions=5_000 ===\n")
        for key in ["ts", "phe", "e2e"]:
            print(f"\n================== {MODEL_CONFIGS[key]['name']} ==================\n")
            res = self.run_one_model(key, 91_000, 5_000, quiet=False)
            print(f"[{MODEL_CONFIGS[key]['name']}] Total wall-clock time: {res['elapsed']:.3f}s\n")

    def benchmark(self, test_cases: List[Tuple[int, int]]) -> Dict:
        print("\n=== BENCHMARK: running multiple inputs for all models (quiet) ===\n")
        all_rows: List[Dict] = []

        for (income, deductions) in test_cases:
            print(f"  Running test case income={income}, deductions={deductions} ...")
            for key in ["ts", "phe", "e2e"]:
                row = self.run_one_model(key, income, deductions, quiet=True)
                all_rows.append(row)

        avg = {}
        for key in ["ts", "phe", "e2e"]:
            rows = [r for r in all_rows if r["model"] == key]
            avg[key] = {
                "name": MODEL_CONFIGS[key]["name"],
                "time": _nanmean([r["elapsed"] for r in rows]),
                "wire_kb": _nanmean([(r["wire_bytes"] / 1024) if r["wire_bytes"] is not None else None for r in rows]),
                "ct_kb": _nanmean([(r["ciphertext_bytes"] / 1024) if r["ciphertext_bytes"] else None for r in rows]),
                "key_kb": _nanmean([(r["key_bytes"] / 1024) if r["key_bytes"] else None for r in rows]),
                "err": _nanmean([r["error"] for r in rows]),
            }

        return {"rows": all_rows, "avg": avg}

    def save_csv(self, rows: List[Dict], filename: str = "tax_benchmark_results.csv"):
        path = os.path.join(self.results_dir, filename)
        cols = [
            "model", "name", "income", "deductions", "taxable",
            "elapsed",
            "wire_bytes", "wire_messages",
            "key_bytes", "ciphertext_bytes",
            "tax", "expected_tax", "error",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c) for c in cols})
        print(f"\n[+] Saved raw results CSV: {path}")

    def plot_summary(self, avg: Dict):
        ordered = ["ts", "phe", "e2e"]
        labels = [avg[k]["name"] for k in ordered]

        avg_time = [avg[k]["time"] for k in ordered]
        avg_wire = [avg[k]["wire_kb"] for k in ordered]
        avg_ct = [avg[k]["ct_kb"] for k in ordered]
        avg_key = [avg[k]["key_kb"] for k in ordered]
        avg_err = [avg[k]["err"] for k in ordered]

        # 2x3 grid with one slot disabled => 5 plots total
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        fig.suptitle("Benchmark Summary (Average over test cases)")

        def _bar(ax, xs, title, ylabel, fmt):
            draw = []
            na_mask = []
            for v in xs:
                is_na = (v is None) or (isinstance(v, float) and math.isnan(v))
                na_mask.append(is_na)
                draw.append(0.0 if is_na else float(v))

            bars = ax.bar(labels, draw)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", linestyle="--", alpha=0.3)
            ax.tick_params(axis="x", rotation=12)

            for bar, v, is_na in zip(bars, xs, na_mask):
                txt = "N/A" if is_na else fmt(float(v))
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    txt,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        _bar(axes[0, 0], avg_time, "Time (lower is better)", "seconds", lambda v: f"{v:.3f}s")
        _bar(axes[0, 1], avg_wire, "Bytes on wire (lower is better)", "KB (application-layer)", lambda v: f"{v:.1f}KB")
        _bar(axes[0, 2], avg_ct, "Ciphertext size (lower is better)", "KB (raw serialized)", lambda v: f"{v:.1f}KB")
        _bar(axes[1, 0], avg_key, "Key size (lower is better)", "KB (raw)", lambda v: f"{v:.2f}KB")
        _bar(axes[1, 1], avg_err, "Computation error (lower is better)", "abs(error)", lambda v: f"{v:.6f}")

        axes[1, 2].axis("off")  # hide unused 6th slot

        plt.tight_layout()

        out_png = os.path.join(self.results_dir, "benchmark_summary.png")
        plt.savefig(out_png, dpi=250, bbox_inches="tight")
        print(f"[+] Saved figure: {out_png}")

        plt.show()
        plt.close(fig)  # helps some backends exit cleanly after window closes

    def run_all(self):
        self.demo_run()
        results = self.benchmark([
            (30_000, 0),
            (60_000, 5_000),
            (90_000, 10_000),
            (150_000, 15_000),
        ])

        print("\n=== Summary (averages) ===")
        for key in ["ts", "phe", "e2e"]:
            a = results["avg"][key]
            print(
                f"{a['name']}: "
                f"time={a['time']:.3f}s, "
                f"wire={a['wire_kb']:.1f}KB, "
                f"ct_size={a['ct_kb']:.1f}KB, "
                f"key_size={a['key_kb']:.2f}KB, "
                f"error={a['err']:.6f}"
            )

        self.save_csv(results["rows"])
        self.plot_summary(results["avg"])


def main():
    analyzer = AdvancedTaxPerformanceAnalyzer()
    analyzer.run_all()


if __name__ == "__main__":
    main()
