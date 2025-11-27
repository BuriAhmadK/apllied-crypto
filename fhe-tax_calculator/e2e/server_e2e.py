#!/usr/bin/env python3
"""
server_e2e.py/ End-to-end encrypted progressive tax server.

Uses symmetric encryption (Fernet) between client and server.
Tax model is the same progressive multi-band model as the FHE (TenSEAL) version:

    0–30k   -> 1%
    30–60k  -> 2%
    60–90k  -> 3%
    90k+    -> 4%

Behavior:
    Server sees everything in plaintext after decryption.
    Encryption only hides data from eavesdroppers, not from the server.
"""

import json
import socket

from cryptography.fernet import Fernet

HOST = "127.0.0.1"
PORT = 7001  # separate from other demos

# Same tax law as TenSEAL PF model
TAX_THRESHOLDS = [30_000, 60_000, 90_000]  # T1, T2, T3
TAX_RATES      = [1, 2, 3, 4]              # r1..r4 in percent

# Shared symmetric key (for demo).
# Generate once with:  >>> from cryptography.fernet import Fernet; Fernet.generate_key()
SHARED_KEY = b"Zx8X1kQf9_qx5sFzfn57zk1Uy_sz1XkRTgqRvXKyB1Q="  # <-- hardcoded demo key

fernet = Fernet(SHARED_KEY)


def compute_progressive_tax(taxable: int):
    """Plain multi-band progressive tax computation."""
    T1, T2, T3 = TAX_THRESHOLDS
    r1, r2, r3, r4 = TAX_RATES

    slice1 = max(0, min(taxable, T1))
    slice2 = max(0, min(max(taxable - T1, 0), T2 - T1))
    slice3 = max(0, min(max(taxable - T2, 0), T3 - T2))
    slice4 = max(0, taxable - T3)

    tax1 = slice1 * r1 / 100.0
    tax2 = slice2 * r2 / 100.0
    tax3 = slice3 * r3 / 100.0
    tax4 = slice4 * r4 / 100.0

    total = tax1 + tax2 + tax3 + tax4
    return (slice1, slice2, slice3, slice4), (tax1, tax2, tax3, tax4), total


def recv_encrypted_json(conn):
    """
    Receive ONE line-delimited ciphertext, decrypt with Fernet, parse JSON.
    Wire format: base64url(Fernet_token) + '\n'
    """
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buf += chunk
    if not buf:
        return None

    ct_b64 = buf.strip()                # e.g. b"gAAAAABl..."
    plaintext = fernet.decrypt(ct_b64)  # bytes
    return json.loads(plaintext.decode("utf-8"))


def send_encrypted_json(conn, obj):
    """
    Serialize JSON, encrypt with Fernet, send as one line.
    """
    plaintext = (json.dumps(obj) + "\n").encode("utf-8")
    token = fernet.encrypt(plaintext)  # bytes, URL-safe base64
    conn.sendall(token + b"\n")


def handle_client(conn, addr):
    print(f"[SERVER] Connected from {addr}")
    try:
        req = recv_encrypted_json(conn)
        if req is None:
            print("[SERVER] Empty request, closing.")
            return

        print(f"[SERVER] Decrypted request: {req}")

        income = int(req["income"])
        deductions = int(req["deductions"])
        taxable = max(income - deductions, 0)

        slices, band_taxes, total_tax = compute_progressive_tax(taxable)
        s1, s2, s3, s4 = slices
        t1, t2, t3, t4 = band_taxes

        print("[SERVER] Plaintext computation:")
        print(f"  income     = {income}")
        print(f"  deductions = {deductions}")
        print(f"  taxable    = {taxable}")
        print(f"  slices     = {slices}")
        print(f"  band_taxes = {band_taxes}")
        print(f"  total_tax  = {total_tax}")

        resp = {
            "type": "tax_response",
            "taxable": taxable,
            "total_tax": total_tax,
            "slices": {
                "band1": s1,
                "band2": s2,
                "band3": s3,
                "band4": s4,
            },
            "band_taxes": {
                "band1": t1,
                "band2": t2,
                "band3": t3,
                "band4": t4,
            },
            "thresholds": TAX_THRESHOLDS,
            "rates": TAX_RATES,
        }

        print(f"[SERVER] Sending encrypted response: {resp}")
        send_encrypted_json(conn, resp)
        print("[SERVER] Done.\n")

    finally:
        conn.close()


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[SERVER] Listening on {HOST}:{PORT} ...")
        # while True: #if you want to handkle multin reqs uncomment this
        #     conn, addr = s.accept()
        #     handle_client(conn, addr)
        conn, addr = s.accept() #comment this and uncomment above for multi reqs
        handle_client(conn, addr) #and this

        print("[SERVER] Exiting after one request.")


if __name__ == "__main__":
    main()
