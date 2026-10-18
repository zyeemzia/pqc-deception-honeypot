"""Real ML-KEM-768 handshakes against a PQC gateway (ESP32 or decoy).

Usage: python client.py <IP> [n_runs] [port] [out_name]
  ESP32: python client.py 192.168.1.50 100
  decoy: python client.py 127.0.0.1 100 5001 decoy_handshake.csv
Timed-out handshakes are recorded (timeout=1), not fatal.
"""
import csv
import hashlib
import socket
import struct
import sys
import time
from pathlib import Path

from kyber_py.ml_kem import ML_KEM_768

PK_LEN, CT_LEN = 1184, 1088
TIMEOUT_S = 15
EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
FIELDS = ["run", "total_ms", "client_encaps_ms", "server_keygen_ms",
          "server_decaps_ms", "confirm_ok", "timeout"]


def recv_exact(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed early")
        buf += chunk
    return buf


def handshake(ip, port):
    t0 = time.perf_counter()
    with socket.create_connection((ip, port), timeout=TIMEOUT_S) as s:
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.sendall(b"K")
        pk = recv_exact(s, PK_LEN)
        t_pk = time.perf_counter()
        ss, ct = ML_KEM_768.encaps(pk)
        t_enc = time.perf_counter()
        s.sendall(ct)
        reply = recv_exact(s, 40)
    t_end = time.perf_counter()
    srv_keygen_us, srv_decaps_us = struct.unpack("<II", reply[32:40])
    return dict(total_ms=(t_end - t0) * 1e3,
                client_encaps_ms=(t_enc - t_pk) * 1e3,
                server_keygen_ms=srv_keygen_us / 1e3,
                server_decaps_ms=srv_decaps_us / 1e3,
                confirm_ok=int(reply[:32] == hashlib.sha3_256(ss).digest()),
                timeout=0)


def probe(ip, port, cmd):
    with socket.create_connection((ip, port), timeout=TIMEOUT_S) as s:
        s.sendall(cmd)
        return s.recv(256).decode(errors="ignore").strip()


def main():
    ip = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    out = EVAL_DIR / (sys.argv[4] if len(sys.argv) > 4 else "network_handshake.csv")
    print("banner:", probe(ip, port, b"B"))
    print("downgrade request reply:", probe(ip, port, b"C"))
    rows = []
    for i in range(n):
        try:
            r = handshake(ip, port)
        except (TimeoutError, socket.timeout, ConnectionError, OSError) as e:
            r = dict(total_ms="", client_encaps_ms="", server_keygen_ms="",
                     server_decaps_ms="", confirm_ok=0, timeout=1)
            print(f"{i:3d} FAILED ({type(e).__name__})")
        else:
            print(f"{i:3d} total={r['total_ms']:.1f}ms ok={r['confirm_ok']}")
        r["run"] = i
        rows.append(r)
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out}  (timeouts: {sum(r['timeout'] for r in rows)}/{n})")


if __name__ == "__main__":
    main()
