#!/usr/bin/env python3
"""
client_e2e.py - End-to-end  encrypted progressive tax client.

- Gathers income & deductions in plaintext on the client.
- Sends them in ENCRYPTED form to the server (Fernet).
- Server decrypts, computes full progressive tax in plaintext, re-encrypts result.
- Client decrypts final result.

Compared to:
Plain model: same computation, but packets are plaintext.
TenSEAL model: server never sees income or tax; here the server DOES see them.
"""

import json
import socket
import time 

from cryptography.fernet import Fernet
import os
WIRE = {"packets_out": 0, "packets_in": 0, "bytes_out": 0, "bytes_in": 0}

HOST = "127.0.0.1"
PORT = 7001  # must match server_e2e.py

# Same key as in server_e2e.py
SHARED_KEY = b"Zx8X1kQf9_qx5sFzfn57zk1Uy_sz1XkRTgqRvXKyB1Q="

fernet = Fernet(SHARED_KEY)


def recv_encrypted_json(sock):
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    if not buf:
        return None
    ct_b64 = buf.strip()
    plaintext = fernet.decrypt(ct_b64)
    WIRE["packets_in"] += 1
    WIRE["bytes_in"] += len(ct_b64) + 1
    return json.loads(plaintext.decode("utf-8"))


def send_encrypted_json(sock, obj):
    plaintext = (json.dumps(obj) + "\n").encode("utf-8")
    token = fernet.encrypt(plaintext)
    WIRE["packets_out"] += 1
    WIRE["bytes_out"] += len(token) + 1
    sock.sendall(token + b"\n")


def get_int(prompt, min_value=0):
    while True:
        txt = input(prompt).strip()
        try:
            value = int(txt)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if value < min_value:
            print(f"Value must be at least {min_value}.")
            continue
        return value


def main():
    income = get_int("Enter annual income (plaintext on client): ", min_value=0)
    deductions = get_int("Enter total deductions (plaintext on client): ", min_value=0)

    print("\n[CLIENT] Local plaintext values:")
    print(f"  income     = {income}")
    print(f"  deductions = {deductions}")

    req = {
        "type": "tax_request",
        "income": income,
        "deductions": deductions,
    }

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        print(f"\n[CLIENT] Connecting to {HOST}:{PORT} ...")
        sock.connect((HOST, PORT))

        start_time = time.perf_counter()

        print("[CLIENT] Sending ENCRYPTED request packet (JSON inside Fernet).")
        send_encrypted_json(sock, req)

        resp = recv_encrypted_json(sock)
        if resp is None:
            print("[CLIENT] No response from server.")
            return

        print("\n[CLIENT] Decrypted response from server:")
        print(resp)

        end_time = time.perf_counter()
        print(f"[CLIENT] Received response from server in {end_time - start_time:.4f} seconds.")

    taxable = resp["taxable"]
    total_tax = resp["total_tax"]
    slices = resp["slices"]
    band_taxes = resp["band_taxes"]
    thresholds = resp["thresholds"]
    rates = resp["rates"]

    print("\n[CLIENT] === RESULT (E2E transport-encrypted model) ===")
    print(f"Taxable income (computed on server): {taxable}")
    print(f"Total tax due                      : {total_tax:.2f}\n")

    print("[CLIENT] Breakdown per band (computed on SERVER):")
    T1, T2, T3 = thresholds
    r1, r2, r3, r4 = rates

    print(f"  Band 1 [0, {T1}):       slice={slices['band1']:7d}  rate={r1}%  tax={band_taxes['band1']:.2f}")
    print(f"  Band 2 [{T1}, {T2}):    slice={slices['band2']:7d}  rate={r2}%  tax={band_taxes['band2']:.2f}")
    print(f"  Band 3 [{T2}, {T3}):    slice={slices['band3']:7d}  rate={r3}%  tax={band_taxes['band3']:.2f}")
    print(f"  Band 4 [{T3}, inf):    slice={slices['band4']:7d}  rate={r4}%  tax={band_taxes['band4']:.2f}")

    print("__METRICS__ " + json.dumps({
    "model": os.getenv("MODEL_KEY", ""),
    "role": os.getenv("ROLE", "client"),
    **WIRE
    }))


if __name__ == "__main__":
    main()
