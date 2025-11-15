/*
================================================================================
File: sampler.h
Purpose: Token sampling utilities for autoregressive text generation
================================================================================

Overview:
---------
During inference, language models output logits (unnormalized probabilities)
for each token in the vocabulary. This file provides utilities to sample
the next token from these logits, enabling text generation.

Sampling Strategies:
-------------------
This file implements:
1. Random number generation (xorshift RNG)
2. Softmax sampling (multinomial sampling from logits)

Autoregressive Generation:
--------------------------
Text generation loop:
1. Start with prompt tokens
2. Forward pass through model -> get logits for next token
3. Sample next token from logits
4. Append token to sequence
5. Repeat until end-of-text or max length

Why Sampling (not argmax)?
--------------------------
- Argmax (greedy): Always picks highest probability token
  -> Repetitive, boring text
- Sampling: Picks from probability distribution
  -> More diverse, natural text
- Temperature: Controls randomness (covered in main training code)

Random Number Generation:
-------------------------
Uses xorshift: A fast, simple RNG suitable for generating random tokens.
Not cryptographically secure, but fine for text generation.

Usage:
------
    unsigned long long rng_state = 12345;
    float logits[vocab_size];
    // ... get logits from model ...
    float random_coin = random_f32(&rng_state);
    int token = sample_softmax(logits, vocab_size, random_coin);
*/
#ifndef SAMPLER_H
#define SAMPLER_H

#include <math.h>

/**
 * random_u32 - Generates a random 32-bit unsigned integer using xorshift
 *
 * @param state: Pointer to RNG state (modified by this function)
 * @return: Random 32-bit unsigned integer
 *
 * Xorshift RNG: Fast, simple, decent quality for non-cryptographic use.
 * State must be non-zero initially.
 */
unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

/**
 * random_f32 - Generates a random float in [0, 1)
 *
 * @param state: Pointer to RNG state
 * @return: Random float in range [0, 1)
 */
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

/**
 * sample_softmax - Samples a token index from logits using softmax probabilities
 *
 * @param logits: Array of logit values (unnormalized log-probabilities)
 * @param n: Number of tokens (vocabulary size)
 * @param coin: Random value in [0, 1) from random_f32()
 * @return: Sampled token index
 *
 * Algorithm:
 * 1. Compute softmax probabilities: p_i = exp(logit_i) / sum(exp(logit_j))
 * 2. Create cumulative distribution: CDF[i] = sum(p_j for j <= i)
 * 3. Sample: Find smallest i where CDF[i] >= coin
 *
 * Optimization: Instead of normalizing probabilities (expensive division),
 * we multiply coin by the normalization constant. Same result, faster.
 *
 * Example:
 *   logits = [1.0, 2.0, 0.5]
 *   -> probs ≈ [0.24, 0.66, 0.14] (after softmax)
 *   -> CDF = [0.24, 0.90, 1.00]
 *   If coin = 0.5, returns index 1 (since 0.24 < 0.5 < 0.90)
 */
int sample_softmax(const float* logits, int n, float coin) {
    // sample index from logits (converted to probabilities using softmax)
    // coin is a random number in [0, 1), usually from random_f32()
    double norm = 0;
    for (int i = 0; i < n; i++) {
        norm += expf(logits[i]);
    }
    // instead of dividing all exp(logits), we can just multiply coin.
    coin *= norm;
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += expf(logits[i]);
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

#endif