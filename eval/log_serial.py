import sys
import time
import serial

port = sys.argv[1] if len(sys.argv) > 1 else "COM5"
out = "esp32_kyber.csv"
meta = "esp32_kyber_meta.txt"

s = serial.Serial()
s.port = port
s.baudrate = 115200
s.timeout = 120
s.dtr = False  # set before open so Windows doesn't hold the chip in boot mode
s.rts = False
s.open()

# EN pulse via RTS (IO0 stays high) -> normal boot
s.rts = True
time.sleep(0.1)
s.rts = False
s.reset_input_buffer()

with s, open(out, "w") as f, open(meta, "w") as m:
    while True:
        raw = s.readline()
        if not raw:
            print("timeout waiting for data")
            break
        line = raw.decode(errors="ignore").strip()
        if not line:
            continue
        print(line)
        if line.startswith("#"):
            m.write(line + "\n")
            if line.startswith("# DONE"):
                break
        elif line.startswith("run,") or line[0].isdigit():
            f.write(line + "\n")

print("saved", out, "and", meta)
