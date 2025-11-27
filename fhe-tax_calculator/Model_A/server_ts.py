#!/usr/bin/env python3
"""
server_ts.py - Privacy-preserving progressive tax calculator (TenSEAL BFV server)

im doing protection against some analytical attacks as number of comparisons is fixed and
the server never sees the plaintext results of the comparisons.

Privacy model (informal)
------------------------
- The server NEVER sees:
    * user income
    * user deductions
    * taxable income
    * total tax
    * which band the user is in

- The server ONLY sees:
    * encrypted taxable income (TenSEAL BFV ciphertext: bfv_vector)
    * encrypted comparison bits b1, b2, b3 (0/1)
    * public tax law (thresholds & rates), hard-coded.

- Band selection protocol:
    * Server computes Enc(taxable - T_i) for each threshold T_i.
    * Server sends Enc(taxable - T_i) to the client.
    * Client decrypts, decides if taxable > T_i, encrypts bit (0/1) and sends Enc(bit) back.
    * Server only ever sees Enc(bit); due to semantic security, it cannot tell 0 from 1.

- All slices and per-band tax computation are done HOMOMORPHICALLY on the server.

Limitations
-----------
- No TLS / authentication on the socket.
"""

import base64
import json
import socket
from typing import Dict
import os
import zlib

WIRE = {"packets_out": 0, "packets_in": 0, "bytes_out": 0, "bytes_in": 0}

import tenseal as ts

# -------------------------
# Public tax law (server & client both know this)
# 0–30k   -> 1%
# 30–60k  -> 2%
# 60–90k  -> 3%
# 90k+    -> 4%
# -------------------------
TAX_THRESHOLDS = [30_000, 60_000, 90_000]  # T1, T2, T3
TAX_RATES      = [1, 2, 3, 4]              # r1..r4 in percent

HOST = "127.0.0.1"
PORT = 5000


def _b64encode(b: bytes) -> str:
    # compress + base85 to reduce JSON blowup
    return base64.b85encode(zlib.compress(b, level=9)).decode("ascii")


def _b64decode(s: str) -> bytes:
    return zlib.decompress(base64.b85decode(s.encode("ascii")))


def handle_client(conn: socket.socket) -> None:
    """Handle a single client from init → result."""

    def recv_exact(n: int):
        data = b""
        while len(data) < n:
            chunk = conn.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def recv_packet():
        # 4-byte big-endian length prefix, then JSON body
        header = recv_exact(4)
        if header is None:
            return None
        length = int.from_bytes(header, "big")
        body = recv_exact(length)
        if body is None:
            return None
        pkt = json.loads(body.decode("utf-8"))
        print(f"[SERVER] Received packet: type={pkt.get('type')} keys={list(pkt.keys())}")
        WIRE["packets_in"] += 1
        WIRE["bytes_in"] += 4 + length
        return pkt

    def send_packet(pkt: Dict) -> None:
        # compact JSON (smaller bytes)
        body = json.dumps(pkt, separators=(",", ":")).encode("utf-8")
        header = len(body).to_bytes(4, "big")
        print(
            f"[SERVER] Sending packet: type={pkt.get('type')} "
            f"keys={list(pkt.keys())} (ciphertexts only, no plaintext user values)"
        )
        WIRE["packets_out"] += 1
        WIRE["bytes_out"] += len(header) + len(body)
        conn.sendall(header + body)

    try:
        # ------------------------------------------------------------
        # 1. Receive TenSEAL context (without secret key) and Enc(taxable)
        # ------------------------------------------------------------
        init_pkt = recv_packet()
        if init_pkt is None or init_pkt.get("type") != "init":
            print("[SERVER] No valid init packet, closing.")
            return

        context_bytes = _b64decode(init_pkt["context"])
        context = ts.context_from(context_bytes)  # public context only

        ctxt_taxable_bytes = _b64decode(init_pkt["ctxt_taxable"])
        ctxt_taxable = ts.bfv_vector_from(context, ctxt_taxable_bytes)

        print("[SERVER] Context loaded (no secret key). Starting secure band selection...")

        # ------------------------------------------------------------
        # 2. Secure comparison to obtain ENCRYPTED bits:
        #    b1 = [taxable > T1], b2 = [taxable > T2], b3 = [taxable > T3]
        # ------------------------------------------------------------
        bits = []
        for i, threshold in enumerate(TAX_THRESHOLDS, start=1):
            # TenSEAL expects list[int] or BFVVector, not raw int
            ctxt_diff = ctxt_taxable - [threshold]  # Enc(y - T_i)

            cmp_req = {
                "type": "cmp_request",
                "id": f"gt_T{i}",
                "ctxt_diff": _b64encode(ctxt_diff.serialize()),
            }
            send_packet(cmp_req)

            resp = recv_packet()
            if resp is None or resp.get("type") != "cmp_response" or resp.get("id") != f"gt_T{i}":
                print("[SERVER] Missing/invalid comparison response, aborting.")
                return

            bit_ctxt_bytes = _b64decode(resp["ctxt_bit"])
            bit_ctxt = ts.bfv_vector_from(context, bit_ctxt_bytes)
            bits.append(bit_ctxt)

        b1, b2, b3 = bits  # all encrypted 0/1

        # ------------------------------------------------------------
        # 3. Homomorphic progressive tax computation
        #
        # Slices:
        #   slice1 = y*(1-b1)           + T1*b1
        #   slice2 = (y-T1)*(b1-b2)     + (T2-T1)*b2
        #   slice3 = (y-T2)*(b2-b3)     + (T3-T2)*b3
        #   slice4 = (y-T3)*b3
        #
        # tax_scaled = sum_i r_i * slice_i
        # (scaled by 100, since r_i is a percent; client divides by 100)
        # ------------------------------------------------------------
        T1, T2, T3 = TAX_THRESHOLDS
        r1, r2, r3, r4 = TAX_RATES

        # Enc(1)
        one = ts.bfv_vector(context, [1])

        # slice1
        b1_inv = one - b1                      # Enc(1 - b1)
        slice1 = ctxt_taxable * b1_inv         # Enc(y * (1 - b1))
        slice1 = slice1 + (b1 * [T1])          # + Enc(T1 * b1)

        # slice2
        y_minus_T1 = ctxt_taxable - [T1]
        b1_minus_b2 = b1 - b2
        slice2 = y_minus_T1 * b1_minus_b2
        slice2 = slice2 + (b2 * [T2 - T1])

        # slice3
        y_minus_T2 = ctxt_taxable - [T2]
        b2_minus_b3 = b2 - b3
        slice3 = y_minus_T2 * b2_minus_b3
        slice3 = slice3 + (b3 * [T3 - T2])

        # slice4
        y_minus_T3 = ctxt_taxable - [T3]
        slice4 = y_minus_T3 * b3

        # Multiply each slice by its rate (ciphertext * list[int]).
        tax1 = slice1 * [r1]
        tax2 = slice2 * [r2]
        tax3 = slice3 * [r3]
        tax4 = slice4 * [r4]

        ctxt_total_tax = tax1 + tax2 + tax3 + tax4  # encrypted vector length 1

        # ------------------------------------------------------------
        # 4. Send encrypted tax back to client.
        # ------------------------------------------------------------
        result_pkt = {
            "type": "result",
            "ctxt_tax": _b64encode(ctxt_total_tax.serialize()),
        }
        send_packet(result_pkt)
        print("[SERVER] Finished homomorphic tax computation for client.")
    finally:
        conn.close()
        print("[SERVER] Connection closed.")


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[SERVER] Listening on {HOST}:{PORT} ...")

        conn, addr = s.accept()
        print(f"[SERVER] Accepted connection from {addr}")
        handle_client(conn)
        print("__METRICS__ " + json.dumps({
            "model": os.getenv("MODEL_KEY", ""),
            "role": os.getenv("ROLE", "server"),
            **WIRE
        }))


if __name__ == "__main__":
    main()
