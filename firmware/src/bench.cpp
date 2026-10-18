#include <Arduino.h>
extern "C" {
#include "api.h"
}

#define N_RUNS 100
#define TASK_STACK 49152

static uint8_t pk[PQCLEAN_MLKEM768_CLEAN_CRYPTO_PUBLICKEYBYTES];
static uint8_t sk[PQCLEAN_MLKEM768_CLEAN_CRYPTO_SECRETKEYBYTES];
static uint8_t ct[PQCLEAN_MLKEM768_CLEAN_CRYPTO_CIPHERTEXTBYTES];
static uint8_t ss_a[PQCLEAN_MLKEM768_CLEAN_CRYPTO_BYTES];
static uint8_t ss_b[PQCLEAN_MLKEM768_CLEAN_CRYPTO_BYTES];

static void bench(void *) {
  Serial.printf("# cpu_mhz=%u\n", getCpuFrequencyMhz());
  Serial.println("run,keygen_us,encaps_us,decaps_us,ok,free_heap,min_free_heap");
  for (int i = 0; i < N_RUNS; i++) {
    int64_t t0 = esp_timer_get_time();
    PQCLEAN_MLKEM768_CLEAN_crypto_kem_keypair(pk, sk);
    int64_t t1 = esp_timer_get_time();
    PQCLEAN_MLKEM768_CLEAN_crypto_kem_enc(ct, ss_a, pk);
    int64_t t2 = esp_timer_get_time();
    PQCLEAN_MLKEM768_CLEAN_crypto_kem_dec(ss_b, ct, sk);
    int64_t t3 = esp_timer_get_time();
    int ok = memcmp(ss_a, ss_b, sizeof ss_a) == 0;
    Serial.printf("%d,%lld,%lld,%lld,%d,%u,%u\n", i, t1 - t0, t2 - t1, t3 - t2, ok,
                  ESP.getFreeHeap(), ESP.getMinFreeHeap());
  }
  UBaseType_t hw = uxTaskGetStackHighWaterMark(NULL);
  Serial.printf("# peak_stack_bytes=%u\n", (unsigned)(TASK_STACK - hw * sizeof(StackType_t)));
  Serial.printf("# pk=%u sk=%u ct=%u ss=%u\n", (unsigned)sizeof pk, (unsigned)sizeof sk,
                (unsigned)sizeof ct, (unsigned)sizeof ss_a);
  Serial.println("# DONE");
  vTaskDelete(NULL);
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  xTaskCreatePinnedToCore(bench, "bench", TASK_STACK, NULL, 1, NULL, 1);
}

void loop() { delay(1000); }
