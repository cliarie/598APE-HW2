#include <philox_rng.h>
#include <math.h>

/* Global generator state */
static philox_state_t global_philox_state = {
    .counter = {1, 2, 3, 3},
    .key = {56, 712},
    .initialized = false
};

/* Helper function to return the lower and higher 32-bits from two 32-bit
 * integer multiplications. */
static void multiply_high_low(uint32_t a, uint32_t b, uint32_t* result_low,
                              uint32_t* result_high) {
    uint64_t product = (uint64_t)a * b;
    *result_low = (uint32_t)product;
    *result_high = (uint32_t)(product >> 32);
}

/* Helper function for a single round of the underlying Philox algorithm. */
static void philox_oneround(uint32_t counter[4], uint32_t key[2]) {
    uint32_t lo0, hi0, lo1, hi1;
    
    multiply_high_low(PHILOX_M4x32A, counter[0], &lo0, &hi0);
    multiply_high_low(PHILOX_M4x32B, counter[2], &lo1, &hi1);
    
    uint32_t result[4];
    result[0] = hi1 ^ counter[1] ^ key[0];
    result[1] = lo1;
    result[2] = hi0 ^ counter[3] ^ key[1];
    result[3] = lo0;
    
    counter[0] = result[0];
    counter[1] = result[1];
    counter[2] = result[2];
    counter[3] = result[3];
}

static void philox_raisekey(uint32_t key[2]) {
    key[0] += PHILOX_W32A;
    key[1] += PHILOX_W32B;
}

/* Seed the RNG with a single value */
void philox_seed(uint32_t seed) {
    global_philox_state.counter[0] = seed;
    global_philox_state.counter[1] = 2;
    global_philox_state.counter[2] = 3;
    global_philox_state.counter[3] = 4;
    global_philox_state.key[0] = PHILOX_W32A;
    global_philox_state.key[1] = PHILOX_W32B;
    global_philox_state.initialized = true;
}

/* Seed the RNG with a value and custom keys */
void philox_seed_with_key(uint32_t seed, uint32_t key0, uint32_t key1) {
    global_philox_state.counter[0] = seed;
    global_philox_state.counter[1] = 2;
    global_philox_state.counter[2] = 3;
    global_philox_state.counter[3] = 4;
    global_philox_state.key[0] = key0;
    global_philox_state.key[1] = key1;
    global_philox_state.initialized = true;
}

/* Reset the RNG to its default state */
void philox_reset(void) {
    global_philox_state.counter[0] = 1;
    global_philox_state.counter[1] = 2;
    global_philox_state.counter[2] = 3;
    global_philox_state.counter[3] = 3;
    global_philox_state.key[0] = 56;
    global_philox_state.key[1] = 712;
    global_philox_state.initialized = true;
}

/* Skip the specified number of samples of 128-bits in the current stream. */
void philox_skip(uint64_t count) {
    if (!global_philox_state.initialized) {
        philox_reset();
    }
    
    const uint32_t count_lo = (uint32_t)count;
    uint32_t count_hi = (uint32_t)(count >> 32);
    
    global_philox_state.counter[0] += count_lo;
    if (global_philox_state.counter[0] < count_lo) {
        ++count_hi;
    }
    
    global_philox_state.counter[1] += count_hi;
    if (global_philox_state.counter[1] < count_hi) {
        if (++global_philox_state.counter[2] == 0) {
            ++global_philox_state.counter[3];
        }
    }
}

/* Returns a group of four random numbers using the underlying Philox algorithm. */
void philox_next4(uint32_t out[4]) {
    if (!global_philox_state.initialized) {
        philox_reset();
    }
    
    for (int i = 0; i < 4; i++) {
        out[i] = global_philox_state.counter[i];
    }
    
    uint32_t key[2] = {global_philox_state.key[0], global_philox_state.key[1]};
    
    for (int i = 0; i < 10; i++) {
        philox_oneround(out, key);
        if (i == 9) {
            philox_skip(1);
        } else {
            philox_raisekey(key);
        }
    }
}


/* Generate a random 64-bit unsigned integer */
uint64_t philox_random_uint64(void) {
    uint32_t out[4];
    philox_next4(out);
    return ((uint64_t)out[0] << 32) | out[1];
}

/* Generate a random double in the range [0, 1) */
double philox_random_double(void) {
    uint32_t out[4];
    philox_next4(out);
    /* Use 53 bits of randomness for double precision */
    uint64_t x = ((uint64_t)out[0] << 32) | out[1];
    return (x >> 11) * (1.0 / (UINT64_C(1) << 53));
}

/* Generate a random float in the range [0, 1) */
float philox_random_float(void) {
    uint32_t x = philox_random_uint32();
    /* Use 24 bits of randomness for single precision */
    return (x >> 8) * (1.0f / (UINT32_C(1) << 24));
}

/* Generate a random int32 in the range [min, max] */
int32_t philox_random_int32_range(int32_t min, int32_t max) {
    if (min >= max) {
        return min;
    }
    uint32_t range = (uint32_t)(max - min + 1);
    uint32_t x = philox_random_uint32();
    uint32_t r = x % range;
    return min + (int32_t)r;
}

/* Generate a random int64 in the range [min, max] */
int64_t philox_random_int64_range(int64_t min, int64_t max) {
    if (min >= max) {
        return min;
    }
    uint64_t range = (uint64_t)(max - min + 1);
    uint64_t x = philox_random_uint64();
    uint64_t r = x % range;
    return min + (int64_t)r;
}

/* Generate a random double in the range [min, max) */
double philox_random_double_range(double min, double max) {
    return min + (max - min) * philox_random_double();
}

/* Generate a random float in the range [min, max) */
float philox_random_float_range(float min, float max) {
    return min + (max - min) * philox_random_float();
}
