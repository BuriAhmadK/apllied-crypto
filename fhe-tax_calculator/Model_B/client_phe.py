#!/usr/bin/env python3
import socket
import json
import time
from phe import paillier
import os
WIRE = {"packets_out": 0, "packets_in": 0, "bytes_out": 0, "bytes_in": 0}
EXTRA = {"key_bytes": 0, "ciphertext_bytes_out": 0, "ciphertext_bytes_in": 0}


HOST = "127.0.0.1"
PORT = 65432  # compute server port

# Demo progressive bands in MONTHLY PKR (toy example)
#  0  - 30,000  -> 1%
# 30k - 60,000  -> 2%
# 60k - 90k          -> 3%
# >90k - no upper limit -> 4%
BANDS = [
    {"lower": 0,      "upper": 30_000, "rate_percent": 1.0},
    {"lower": 30_000, "upper": 60_000, "rate_percent": 2.0},
    {"lower": 60_000, "upper": 90_000, "rate_percent": 3.0},
    {"lower": 90_000, "upper": None,   "rate_percent": 4.0},  # None = no upper cap
]


def recv_json(sock):
    data = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
        if b"\n" in chunk:
            break
    WIRE["packets_in"] += 1
    WIRE["bytes_in"] += len(data)
    return json.loads(data.decode().strip())



def send_json(sock, obj):
    msg = (json.dumps(obj) + "\n").encode("utf-8")
    WIRE["packets_out"] += 1
    WIRE["bytes_out"] += len(msg)
    sock.sendall(msg)



def get_int(prompt, min_value=None, max_value=None):
    while True:
        txt = input(prompt).strip()
        try:
            value = int(txt)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if min_value is not None and value < min_value:
            print(f"Value must be at least {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"Value must be at most {max_value}.")
            continue
        return value


def band_portion(taxable, lower, upper):
    """
    How much of 'taxable' lies in [lower, upper) for this band.
    upper=None means no upper limit.
    """
    if taxable <= lower:
        return 0
    cap = taxable if upper is None else min(taxable, upper)
    return max(cap - lower, 0)


def main():
    print("=== PHE CLIENT ===\n")

    # 1) Generate Paillier keys locally
    public_key, private_key = paillier.generate_paillier_keypair()
    EXTRA["key_bytes"] = (public_key.n.bit_length() + 7) // 8
    print("[CLIENT] Generated Paillier keypair.")

    # 2) User input (monthly)
    monthly_income = get_int("Monthly income in PKR (>= 0): ", min_value=0)

    print(
        "Monthly deductions in PKR (>= 0). Examples: Zakat, approved donations,\n"
        "pension contributions, interest on home loan, etc."
    )
    monthly_deductions = get_int("Enter monthly deductions: ", min_value=0)

    while monthly_deductions > monthly_income:
        print("Deductions cannot be greater than income. Try again.")
        monthly_deductions = get_int("Enter monthly deductions (<= income): ", min_value=0)

    taxable = monthly_income - monthly_deductions

    print(f"\n[CLIENT] Monthly income: {monthly_income}")
    print(f"[CLIENT] Monthly deductions: {monthly_deductions}")
    print(f"[CLIENT] Taxable monthly income: {taxable}")

    # 3) Split taxable income across bands (plaintext)
    band_details = []
    plain_tax_total = 0.0

    for band in BANDS:
        lower = band["lower"]
        upper = band["upper"]
        rate = band["rate_percent"]

        portion = band_portion(taxable, lower, upper)
        band_tax = portion * rate / 100.0
        plain_tax_total += band_tax

        band_details.append({
            "lower": lower,
            "upper": upper,
            "rate_percent": rate,
            "portion": portion,
            "plain_tax": band_tax,
        })

    print("\n[CLIENT] Band breakdown (plaintext, for checking):")
    for b in band_details:
        upper_str = "inf" if b["upper"] is None else str(b["upper"])
        print(
            f"  Band {b['lower']}–{upper_str}: "
            f"portion={b['portion']} at {b['rate_percent']}% -> tax={b['plain_tax']}"
        )
    print(f"[CLIENT] Total plain tax (sum of bands): {plain_tax_total}")

    # 4) Encrypt each band portion and prepare payload
    enc_bands = []
    for b in band_details:
        enc_portion = public_key.encrypt(b["portion"])
        EXTRA["ciphertext_bytes_out"] += (enc_portion.ciphertext().bit_length() + 7) // 8
        rate_percent = b["rate_percent"]
        rate_times1000 = int(rate_percent * 10)  # e.g. 1.0% -> 10, 2.0% -> 20, etc.

        enc_bands.append({
            "rate_times1000": rate_times1000,
            "portion": {
                "c": enc_portion.ciphertext(),
                "exp": enc_portion.exponent,
            },
        })

    print(f"\n[CLIENT] Encrypted all band portions.")

    # 5) Send encrypted band portions to server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"[CLIENT] Connecting to server...")
        s.connect((HOST, PORT))

        payload = {
            "public_key": {"n": public_key.n},
            "bands": enc_bands,
        }

        start_time = time.perf_counter()

        send_json(s, payload)
        print(f"[CLIENT] Sent encrypted band data to server.")

        res = recv_json(s)
        end_time = time.perf_counter()
        print(f"[CLIENT] Received response from server in {end_time - start_time:.4f} seconds.")

    # 6) Decrypt total tax
    tax_payload = res["tax_times1000"]
    EXTRA["ciphertext_bytes_in"] += (int(tax_payload["c"]).bit_length() + 7) // 8
    enc_tax_times1000 = paillier.EncryptedNumber(
        public_key,
        int(tax_payload["c"]),
        int(tax_payload["exp"]),
    )

    tax_times1000 = private_key.decrypt(enc_tax_times1000)
    tax_from_server = tax_times1000 / 1000.0

    print(f"\n[CLIENT] Tax from server (after decrypt): {tax_from_server}")
    print(
        f"[CLIENT] Matches plain-text tax? "
        f"{abs(tax_from_server - plain_tax_total) < 1e-6}"
    )
    print("__METRICS__ " + json.dumps({
    "model": os.getenv("MODEL_KEY", ""),
    "role": os.getenv("ROLE", "client"),
    **WIRE,
    **EXTRA,
    "ciphertext_bytes_total": EXTRA["ciphertext_bytes_out"] + EXTRA["ciphertext_bytes_in"],
    }))


if __name__ == "__main__":
    main()
