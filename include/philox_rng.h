#ifndef PHILOX_RNG_H
#define PHILOX_RNG_H

#include <stdint.h>
#include <stdbool.h>

/*
 * Implementation of the Philox counter-based random number generator
 * Based on: Salmon et al. SC 2011. Parallel random numbers: as easy as 1, 2, 3.
 * http://www.thesalmons.org/john/random123/papers/random123sc11.pdf
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Constants for the Philox algorithm */
#define PHILOX_W32A 0x9E3779B9
#define PHILOX_W32B 0xBB67AE85
#define PHILOX_M4x32A 0xD2511F53
#define PHILOX_M4x32B 0xCD9E8D57

/* Generator state */
typedef struct {
    uint32_t counter[4];
    uint32_t key[2];
    bool initialized;
} philox_state_t;

/* Function prototypes */
void philox_seed(uint32_t seed);
void philox_seed_with_key(uint32_t seed, uint32_t key0, uint32_t key1);
void philox_reset(void);
void philox_skip(uint64_t count);
void philox_next4(uint32_t out[4]);
inline uint32_t philox_random_uint32(void) {
    uint32_t out[4];
    philox_next4(out);
    return out[0];
}
uint64_t philox_random_uint64(void);
double philox_random_double(void);
float philox_random_float(void);
int32_t philox_random_int32_range(int32_t min, int32_t max);
int64_t philox_random_int64_range(int64_t min, int64_t max);
double philox_random_double_range(double min, double max);
float philox_random_float_range(float min, float max);

#ifdef __cplusplus
}
#endif

#endif /* PHILOX_RNG_H */
