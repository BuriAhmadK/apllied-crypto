#!/usr/bin/env python3
"""
client_ts.py - Privacy-preserving progressive tax calculator (TenSEAL BFV client)

Client knows in plaintext:
--------------------------
- Income, deductions, taxable income.
- Tax law (thresholds & rates).
- Final tax amount (after decryption).

Client NEVER sends in plaintext:
--------------------------------
- Income
- Deductions
- Taxable income
- Final tax
- Which band they are in.

Protocol:
---------
1) Client:
   - Generates TenSEAL BFV context & keys.
   - Encrypts taxable income as BFV vector of length 1.
   - Sends context (without secret key) + Enc(taxable) to server.

2) For each threshold T_i:
   - Server sends Enc(taxable - T_i).
   - Client decrypts, compares to 0, creates bit = 1 if > 0 else 0.
   - Client encrypts bit as Enc(bit) and sends back.

3) Server computes full progressive tax using homomorphic ops only,
   giving tax_scaled = 100 * tax.

4) Client decrypts Enc(tax_scaled) and divides by 100 to get the final tax.
"""

import base64
import json
import socket
import time
from typing import Dict, Tuple
import zlib

import tenseal as ts
import os

WIRE = {"packets_out": 0, "packets_in": 0, "bytes_out": 0, "bytes_in": 0}
#beep
EXTRA = {"key_bytes": 0, "ciphertext_bytes_out": 0, "ciphertext_bytes_in": 0}


# Same public tax law as on server
TAX_THRESHOLDS = [30_000, 60_000, 90_000]  # T1, T2, T3
TAX_RATES      = [1, 2, 3, 4]              # r1..r4 in percent

HOST = "127.0.0.1"
PORT = 5000


def _b64encode(b: bytes) -> str:
    # compress + base85 to reduce JSON blowup
    return base64.b85encode(zlib.compress(b, level=9)).decode("ascii")


def _b64decode(s: str) -> bytes:
    return zlib.decompress(base64.b85decode(s.encode("ascii")))


def plain_progressive_tax_breakdown(taxable: int) -> Tuple[Tuple[int, int, int, int], float]:
    """
    Local plaintext breakdown for display ONLY (never sent to server).
    """
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
    return (slice1, slice2, slice3, slice4), total


def setup_fhe() -> ts.Context:
    """
    Create BFV context & keys on client.

    plain_modulus must be > max expected tax result, otherwise wrap-around.
    For demo purposes (income up to ~200k), 1_032_193 is fine.
    """
    context = ts.context(
        ts.SCHEME_TYPE.BFV,
        poly_modulus_degree=8192,
        plain_modulus=1032193,
    )

    # You do NOT use rotations anywhere -> galois keys are unnecessary (saves big bytes on wire)
    context.generate_relin_keys()  # needed for ciphertext-ciphertext multiplications

    print(f"[CLIENT] BFV context generated with poly_modulus_degree=8192, plain_modulus=1032193")
    return context


def main():
    # ------------------------------------------------------------
    # 1. Get user input and compute taxable income locally.
    # ------------------------------------------------------------
    income = int(input("Enter annual income (e.g. 91000): ").strip())
    deductions = int(input("Enter total deductions (e.g. 5000): ").strip())
    taxable = max(income - deductions, 0)

    print("\n[CLIENT] Local plaintext values (NEVER sent in clear):")
    print(f"  Gross income : {income}")
    print(f"  Deductions   : {deductions}")
    print(f"  Taxable      : {taxable}")

    # ------------------------------------------------------------
    # 2. Setup TenSEAL context and encrypt taxable income.
    # ------------------------------------------------------------
    context = setup_fhe()
    ctxt_taxable = ts.bfv_vector(context, [taxable])  # Enc([taxable])

    # Serialize context WITHOUT secret key to send to server.
    # Explicit flags to avoid accidentally sending galois keys later.
    context_bytes_public = context.serialize(
        save_secret_key=False,
        save_public_key=True,
        save_galois_keys=False,
        save_relin_keys=True,
    )
    EXTRA["key_bytes"] = len(context_bytes_public)  #beep

    init_packet = {
        "type": "init",
        "context": _b64encode(context_bytes_public),
        "ctxt_taxable": _b64encode(ctxt_taxable.serialize()),
    }
    EXTRA["ciphertext_bytes_out"] += len(ctxt_taxable.serialize())
    # ------------------------------------------------------------
    # 3. Connect to server and run protocol (length-prefixed packets).
    # ------------------------------------------------------------
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        start_time = time.perf_counter()  # start counting time

        def send_packet(pkt: Dict) -> None:
            # compact JSON (smaller bytes)
            body = json.dumps(pkt, separators=(",", ":")).encode("utf-8")
            header = len(body).to_bytes(4, "big")
            print(
                f"[CLIENT] Sending packet: type={pkt.get('type')} "
                f"keys={list(pkt.keys())} (ciphertexts only, no plaintext values)"
            )
            WIRE["packets_out"] += 1
            WIRE["bytes_out"] += len(header) + len(body)
            sock.sendall(header + body)

        def recv_exact(n: int) -> bytes | None:
            data = b""
            while len(data) < n:
                chunk = sock.recv(n - len(data))
                if not chunk:
                    return None
                data += chunk
            return data

        def recv_packet() -> Dict | None:
            header = recv_exact(4)
            if header is None:
                return None
            length = int.from_bytes(header, "big")
            body = recv_exact(length)
            if body is None:
                return None
            pkt = json.loads(body.decode("utf-8"))
            print(f"[CLIENT] Received packet: type={pkt.get('type')} keys={list(pkt.keys())}")
            WIRE["packets_in"] += 1
            WIRE["bytes_in"] += 4 + length
            return pkt

        # Send initial packet.
        send_packet(init_packet)

        ctxt_tax = None

        while True:
            pkt = recv_packet()
            if pkt is None:
                print("[CLIENT] Connection closed before result.")
                break

            ptype = pkt.get("type")

            if ptype == "cmp_request":
                # ---------------------------------------------
                # Secure comparison:
                #   - Server sends Enc(taxable - T_i).
                #   - Client decrypts to plaintext integer diff.
                #   - bit = 1 if diff > 0 else 0.
                #   - Client encrypts bit as Enc(bit) and returns.
                # ---------------------------------------------
                ctxt_diff_bytes = _b64decode(pkt["ctxt_diff"])
                EXTRA["ciphertext_bytes_in"] += len(ctxt_diff_bytes)
                ctxt_diff = ts.bfv_vector_from(context, ctxt_diff_bytes)

                diff_plain = ctxt_diff.decrypt()[0]  # BFV signed int
                bit = 1 if diff_plain > 0 else 0

                ctxt_bit = ts.bfv_vector(context, [bit])
                EXTRA["ciphertext_bytes_out"] += len(ctxt_bit.serialize())


                resp = {
                    "type": "cmp_response",
                    "id": pkt["id"],
                    "ctxt_bit": _b64encode(ctxt_bit.serialize()),
                }
                send_packet(resp)

            elif ptype == "result":
                ctxt_tax_bytes = _b64decode(pkt["ctxt_tax"])
                EXTRA["ciphertext_bytes_in"] += len(ctxt_tax_bytes)

                ctxt_tax = ts.bfv_vector_from(context, ctxt_tax_bytes)
                end_time = time.perf_counter()  # end counting time
                print("Tenseal computation time: {:.2f} seconds".format(end_time - start_time))
                break
            else:
                print(f"[CLIENT] Unexpected packet type {ptype}, ignoring.")

    # ------------------------------------------------------------
    # 4. Decrypt tax, show result and local per-band breakdown.
    # ------------------------------------------------------------
    if ctxt_tax is None:
        print("[CLIENT] No tax result obtained.")
        return

    tax_plain_vec = ctxt_tax.decrypt()
    tax_scaled = float(tax_plain_vec[0])  # = 100 * tax
    tax_due = tax_scaled / 100.0

    print("\n[CLIENT] === FINAL RESULT (decrypted only on client) ===")
    print(f"Taxable income (plaintext)     : {taxable}")
    print(f"Total tax due (server result)  : {tax_due:.2f}")

    slices, recomputed_tax = plain_progressive_tax_breakdown(taxable)
    s1, s2, s3, s4 = slices

    print("\n[CLIENT] Local plaintext breakdown per band (NOT sent to server):")
    print(f"  Band 1 [0, {TAX_THRESHOLDS[0]}):    slice={s1:7d}  rate={TAX_RATES[0]}%")
    print(f"  Band 2 [{TAX_THRESHOLDS[0]}, {TAX_THRESHOLDS[1]}): slice={s2:7d}  rate={TAX_RATES[1]}%")
    print(f"  Band 3 [{TAX_THRESHOLDS[1]}, {TAX_THRESHOLDS[2]}): slice={s3:7d}  rate={TAX_RATES[2]}%")
    print(f"  Band 4 [{TAX_THRESHOLDS[2]}, inf):  slice={s4:7d}  rate={TAX_RATES[3]}%")
    print(f"  Recomputed tax (plaintext)    : {recomputed_tax:.2f}")

    print("\n[CLIENT] Note: tax_due (from FHE) and recomputed_tax should match up to rounding.")
    print("__METRICS__ " + json.dumps({
        "model": os.getenv("MODEL_KEY", ""),
        "role": os.getenv("ROLE", "client"),
        **WIRE,
        **EXTRA,
        "ciphertext_bytes_total": EXTRA["ciphertext_bytes_out"] + EXTRA["ciphertext_bytes_in"],
    }))


if __name__ == "__main__":
    main()
