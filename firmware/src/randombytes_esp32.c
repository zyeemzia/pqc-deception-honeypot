#include <stddef.h>
#include <stdint.h>
#include "randombytes.h"
#include "esp_random.h"

// Hardware RNG is only a true RNG while the WiFi/BT radio is enabled; see paper limitations.
int randombytes(uint8_t *output, size_t n) {
    esp_fill_random(output, n);
    return 0;
}
