#!/usr/bin/env python3
# multi band tax computation server using Paillier homomorphic encryption

# Model B (Paillier):
#
# Server sees:
#   - Public tax law (band structure & rates).
#   - For each band: Enc(portion) and its rate.
#
# Because of Paillier’s semantic security, it cannot tell:
#   - Which portions are 0 vs non-zero.
#   - Which band is the “top” band.
#
# The client has already done the full band splitting and tax in plaintext.
# Server’s role is simpler: homomorphically scale-and-sum encrypted portions.

import socket
import json
from phe import paillier
import os
WIRE = {"packets_out": 0, "packets_in": 0, "bytes_out": 0, "bytes_in": 0}


HOST = "127.0.0.1"
PORT = 65432  # compute server port


def recv_json(conn):
    data = b""
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
        if b"\n" in chunk:
            break
    WIRE["packets_in"] += 1
    WIRE["bytes_in"] += len(data)
    return json.loads(data.decode().strip())


def send_json(conn, obj):
    msg = (json.dumps(obj) + "\n").encode("utf-8")
    WIRE["packets_out"] += 1
    WIRE["bytes_out"] += len(msg)
    conn.sendall(msg)



def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[SERVER] Listening on {HOST}:{PORT} ...")

        print("[SERVER] Waiting for single client...")
        conn, addr = s.accept()
        with conn:
            print(f"[SERVER] Connected by {addr}")

            try:
                req = recv_json(conn)
                print("[SERVER] Received keys:", req.keys())

                # Public key from client
                n = int(req["public_key"]["n"])
                pub_key = paillier.PaillierPublicKey(n)

                bands = req["bands"]  # list of {rate_times1000, portion: {c, exp}}

                print(f"[SERVER] Computing tax over {len(bands)} bands.")

                enc_total_tax_times1000 = None

                for i, band in enumerate(bands):
                    rate_times1000 = int(band["rate_times1000"])
                    p = band["portion"]
                    enc_portion = paillier.EncryptedNumber(
                        pub_key,
                        int(p["c"]),
                        int(p["exp"]),
                    )

                    # Enc(band_tax * 1000) = Enc(portion * rate_times1000)
                    enc_band_tax_times1000 = enc_portion * rate_times1000

                    if enc_total_tax_times1000 is None:
                        enc_total_tax_times1000 = enc_band_tax_times1000
                    else:
                        enc_total_tax_times1000 += enc_band_tax_times1000

                if enc_total_tax_times1000 is None:
                    # no bands? tax = 0
                    enc_total_tax_times1000 = pub_key.encrypt(0)

                res = {
                    "tax_times1000": {
                        "c": enc_total_tax_times1000.ciphertext(),
                        "exp": enc_total_tax_times1000.exponent,
                    }
                }

                send_json(conn, res)
                print("[SERVER] Sent encrypted total tax back to client.\n")

            except Exception as e:
                print("[SERVER] Error handling client:", e)

        print("[SERVER] Exiting after one client.")
        print("__METRICS__ " + json.dumps({
        "model": os.getenv("MODEL_KEY", ""),
        "role": os.getenv("ROLE", "server"),
        **WIRE
        }))



if __name__ == "__main__":
    main()
