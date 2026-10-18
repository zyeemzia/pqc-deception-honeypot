"""Decoy PQC gateway: speaks the same protocol as the ESP32 server.

It performs genuine ML-KEM-768 (so keys/ciphertexts are well-formed), but pads
its compute time to match the ESP32's measured keygen/decaps distributions, and
reports those padded timings. The goal is that an attacker cannot tell decoy
from real by banner, message sizes, or server-side timing.

Usage: python decoy_server.py [port]      (default 5001)
Every interaction is logged to eval/decoy_log.csv for later analysis.
"""
import csv
import hashlib
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from kyber_py.ml_kem import ML_KEM_768

ROOT = Path(__file__).resolve().parents[1]
BANNER = b"PQGW/1.0 ML-KEM-768\n"
PK_LEN, CT_LEN = 1184, 1088
LOG = ROOT / "eval" / "decoy_log.csv"

# Calibrate from the real ESP32 network run (server-side timings in ms).
_real = pd.read_csv(ROOT / "eval" / "network_handshake.csv")
_real = _real[_real.get("timeout", 0) == 0] if "timeout" in _real else _real
KEYGEN_MS = _real.server_keygen_ms.astype(float).to_numpy()
DECAPS_MS = _real.server_decaps_ms.astype(float).to_numpy()
rng = np.random.default_rng(42)
lock = threading.Lock()


def sample(arr):
    with lock:
        return float(rng.choice(arr))


def pad_to(start, target_ms):
    remaining = target_ms / 1e3 - (time.perf_counter() - start)
    if remaining > 0:
        time.sleep(remaining)


def recv_exact(c, n):
    buf = b""
    while len(buf) < n:
        chunk = c.recv(n - len(buf))
        if not chunk:
            raise ConnectionError
        buf += chunk
    return buf


def log(addr, cmd, note):
    with lock:
        new = not LOG.exists()
        with open(LOG, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "src_ip", "cmd", "note"])
            w.writerow([f"{time.time():.3f}", addr[0], cmd, note])


def handle(c, addr):
    with c:
        c.settimeout(10)
        try:
            cmd = recv_exact(c, 1)
            if cmd == b"B":
                c.sendall(BANNER)
                log(addr, "B", "banner")
            elif cmd == b"K":
                t_kg, t_dc = sample(KEYGEN_MS), sample(DECAPS_MS)
                s0 = time.perf_counter()
                ek, dk = ML_KEM_768.keygen()
                pad_to(s0, t_kg)
                c.sendall(ek)
                ct = recv_exact(c, CT_LEN)
                s1 = time.perf_counter()
                ss = ML_KEM_768.decaps(dk, ct)
                pad_to(s1, t_dc)
                c.sendall(hashlib.sha3_256(ss).digest() +
                          struct.pack("<II", int(t_kg * 1e3), int(t_dc * 1e3)))
                log(addr, "K", "handshake")
            elif cmd == b"C":
                c.sendall(b"ERR NO-CLASSIC\n")
                log(addr, "C", "downgrade_attempt")
            else:
                c.sendall(b"ERR UNKNOWN\n")
                log(addr, cmd.hex(), "unknown")
        except (ConnectionError, OSError):
            log(addr, "?", "aborted")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5001
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(16)
    print(f"decoy listening on port {port} (calibrated from {len(KEYGEN_MS)} real handshakes)")
    while True:
        c, addr = srv.accept()
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=handle, args=(c, addr), daemon=True).start()


if __name__ == "__main__":
    main()
