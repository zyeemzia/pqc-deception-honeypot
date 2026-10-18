#include <Arduino.h>
#include <WiFi.h>
#include "secrets.h"
extern "C" {
#include "api.h"
#include "fips202.h"
}

// ML-KEM needs ~15 KB stack (measured in Phase 2); default loop stack is 8 KB.
SET_LOOP_TASK_STACK_SIZE(32 * 1024);

#define PORT 5000
#define BANNER "PQGW/1.0 ML-KEM-768\n"

static WiFiServer server(PORT);
static uint8_t pk[PQCLEAN_MLKEM768_CLEAN_CRYPTO_PUBLICKEYBYTES];
static uint8_t sk[PQCLEAN_MLKEM768_CLEAN_CRYPTO_SECRETKEYBYTES];
static uint8_t ct[PQCLEAN_MLKEM768_CLEAN_CRYPTO_CIPHERTEXTBYTES];
static uint8_t ss[PQCLEAN_MLKEM768_CLEAN_CRYPTO_BYTES];

static bool read_exact(WiFiClient &c, uint8_t *buf, size_t n, uint32_t timeout_ms) {
  size_t got = 0;
  uint32_t t0 = millis();
  while (got < n && millis() - t0 < timeout_ms && c.connected()) {
    int r = c.read(buf + got, n - got);
    if (r > 0) got += r; else delay(1);
  }
  return got == n;
}

static void put_u32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }

// Protocol (1-byte command):
//  'B' -> banner (fingerprinting target)
//  'K' -> ML-KEM handshake: server sends pk, client sends ct,
//         server replies SHA3-256(ss) as key confirmation + server timings
//  'C' -> request classical ECDH fallback (downgrade attempt) -> refused
static void handle(WiFiClient &c) {
  uint8_t cmd;
  if (!read_exact(c, &cmd, 1, 5000)) return;
  if (cmd == 'B') {
    c.print(BANNER);
  } else if (cmd == 'K') {
    int64_t t0 = esp_timer_get_time();
    PQCLEAN_MLKEM768_CLEAN_crypto_kem_keypair(pk, sk);
    int64_t t1 = esp_timer_get_time();
    c.write(pk, sizeof pk);
    if (!read_exact(c, ct, sizeof ct, 10000)) return;
    int64_t t2 = esp_timer_get_time();
    PQCLEAN_MLKEM768_CLEAN_crypto_kem_dec(ss, ct, sk);
    int64_t t3 = esp_timer_get_time();
    uint8_t out[32 + 8];
    sha3_256(out, ss, sizeof ss);
    put_u32(out + 32, (uint32_t)(t1 - t0));
    put_u32(out + 36, (uint32_t)(t3 - t2));
    c.write(out, sizeof out);
    Serial.printf("K keygen_us=%lld decaps_us=%lld heap=%u\n", t1 - t0, t3 - t2, ESP.getFreeHeap());
  } else if (cmd == 'C') {
    c.print("ERR NO-CLASSIC\n");
    Serial.println("C downgrade attempt refused");
  } else {
    c.print("ERR UNKNOWN\n");
  }
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);  // power-save adds tens of ms of jitter to latency
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.printf("\nIP=%s PORT=%d\n", WiFi.localIP().toString().c_str(), PORT);
  server.begin();
}

void loop() {
  WiFiClient c = server.available();
  if (c) {
    c.setNoDelay(true);
    handle(c);
    c.stop();
  }
  delay(1);
}
