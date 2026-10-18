# Hardware measurements

All numbers below were measured on the physical device; raw data is in `eval/`.

## Device

| Item | Value |
|---|---|
| Board | ESP32 DevKit with ESP-32D module (WiFi + BT + BLE), CP2102 USB-UART |
| CPU | Xtensa dual-core, 240 MHz (benchmark pinned to core 1) |
| Framework | Arduino-ESP32 via PlatformIO (espressif32 7.1.3) |
| ML-KEM implementation | PQClean `ml-kem-768/clean` (portable C, no assembly) |
| RNG | `esp_fill_random` (hardware RNG; fully seeded only with radio on) |

## On-device ML-KEM-768 (WiFi off, n = 100) — `eval/esp32_kyber.csv`

| Operation | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| KeyGen | 8.171 | 0.099 | 8.145 | 9.104 |
| Encaps | 9.390 | 0.033 | 9.372 | 9.527 |
| Decaps | 11.150 | 0.030 | 11.133 | 11.247 |

- All 100 runs produced matching shared secrets.
- Peak stack: 15,096 bytes. Free heap during runs: 297,172 bytes (minimum 291,448).
- Sizes: pk 1184 B, sk 2400 B, ct 1088 B, ss 32 B.
- Run 0 is a cache warm-up outlier (KeyGen 9.10 ms).

## Networked handshakes over WiFi (n = 100) — `eval/network_handshake.csv`

ESP32 as server (keygen + decaps on device); laptop client does encaps (kyber-py).
Key confirmation: server returns SHA3-256(shared secret); client verifies.

| Component | Mean (ms) | Std (ms) | Median (ms) | 90th pct (ms) |
|---|---|---|---|---|
| ESP32 KeyGen | 8.64 | 0.03 | 8.63 | 8.65 |
| ESP32 Decaps | 11.60 | 0.03 | 11.60 | 11.62 |
| Client Encaps (Python) | 8.96 | 1.83 | 8.52 | 9.95 |
| Network / other | 1220.6 | 2134.0 | 292.1 | 3304.4 |
| Total | 1249.8 | 2134.3 | 320.8 | 3332.7 |

- 100/100 key confirmations succeeded; 25/100 handshakes exceeded 1 s.
- The network was a shared institutional 2.4 GHz WLAN; latency is dominated by the
  link, not by ML-KEM computation (~20 ms on device).

## Device used for all measurements

<img src="esp32_setup.jpg" alt="The ESP32 DevKit (ESP-32D module) used for all measurements, connected over USB" width="360">
