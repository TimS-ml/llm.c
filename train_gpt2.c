/*
GPT-2 Training Implementation - CPU Reference Version
=====================================================

This file provides a clean, minimal, educational implementation of GPT-2 model training.

Design Philosophy:
- Runs entirely on CPU (no GPU required)
- Optimized for readability and understanding, not raw performance
- Avoids processor-specific intrinsics and complex optimizations
- Uses OpenMP pragmas for basic parallelization (significant speedup with minimal code complexity)
- Serves as the algorithmic reference implementation

This is the "readable baseline" - other versions (train_gpt2.cu) provide optimized GPU implementations.

Key Features:
- Complete forward and backward propagation for all GPT-2 layers
- AdamW optimizer implementation
- Training loop with validation and text generation
- ~1000 lines of straightforward C code

Architecture:
The GPT-2 model consists of:
1. Token + Position Embeddings (encoder)
2. N Transformer Blocks, each containing:
   - Layer Normalization
   - Multi-Head Self-Attention
   - Residual Connection
   - Layer Normalization
   - MLP with GELU activation
   - Residual Connection
3. Final Layer Normalization
4. Output Projection to Vocabulary

Notation Used Throughout:
- B = Batch size (number of independent sequences processed in parallel)
- T = Sequence length (number of tokens in each sequence)
- C = Channels/embedding dimension (e.g., 768 for GPT-2 small)
- V = Vocabulary size (actual, e.g., 50257)
- Vp = Padded vocabulary size (for efficiency, e.g., 50304)
- L = Number of transformer layers (e.g., 12 for GPT-2 small)
- NH = Number of attention heads (e.g., 12 for GPT-2 small)
- hs = Head size (C / NH, the dimension of each attention head)
- OC = Output channels (varies by layer)

Tensor Shapes:
Most tensors follow the pattern (B, T, C) meaning:
- B independent sequences
- T tokens per sequence
- C-dimensional vectors at each position
*/

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#ifdef OMP
#include <omp.h>
#endif
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#include "llmc/dataloader.h"

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes
// B = batch_size, T = sequence_length, C = channels, V = vocab_size

/**
 * Encoder Forward Pass
 *
 * Purpose: Converts input token IDs into embeddings by combining token embeddings and position embeddings.
 * This is the first operation in GPT-2, creating the initial representation of the input sequence.
 *
 * Algorithm:
 * For each token position (b,t):
 *   1. Look up the token embedding from wte using the token ID as index
 *   2. Look up the position embedding from wpe using the position t as index
 *   3. Add these two vectors element-wise to get the final embedding
 *
 * Parameters:
 *   out - Output tensor (B, T, C): The combined token + position embeddings
 *   inp - Input tensor (B, T): Token IDs, integers in range [0, V)
 *   wte - Weight tensor (V, C): Token embedding lookup table ("weight token embeddings")
 *   wpe - Weight tensor (maxT, C): Position embedding lookup table ("weight positional embeddings")
 *   B - Batch size: Number of independent sequences
 *   T - Sequence length: Number of tokens in each sequence
 *   C - Channels: Embedding dimension (e.g., 768 for GPT-2 small)
 *
 * Shape transformations:
 *   Input:  (B, T) integer indices
 *   Output: (B, T, C) float embeddings
 *
 * Reference: "Attention Is All You Need" (Vaswani et al., 2017)
 *            GPT-2 uses learned position embeddings rather than sinusoidal
 */
void encoder_forward(float* out,
                   int* inp, float* wte, float* wpe,
                   int B, int T, int C) {
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            // get the index of the token at inp[b, t]
            int ix = inp[b * T + t];
            // seek to the position in wte corresponding to the token
            float* wte_ix = wte + ix * C;
            // seek to the position in wpe corresponding to the position
            float* wpe_t = wpe + t * C;
            // add the two vectors and store the result in out[b,t,:]
            for (int i = 0; i < C; i++) {
                out_bt[i] = wte_ix[i] + wpe_t[i];
            }
        }
    }
}

/**
 * Encoder Backward Pass
 *
 * Purpose: Backpropagates gradients through the encoder, computing gradients for both
 * token embeddings and position embeddings.
 *
 * Algorithm:
 * Since forward pass computed out = wte[token_id] + wpe[position], the backward pass
 * distributes the gradient from out equally to both wte and wpe:
 *   dwte[token_id] += dout  (accumulated for all positions that use this token)
 *   dwpe[position] += dout  (accumulated for all batches)
 *
 * Parameters:
 *   dwte - Gradient tensor (V, C): Gradients with respect to token embeddings
 *   dwpe - Gradient tensor (maxT, C): Gradients with respect to position embeddings
 *   dout - Gradient tensor (B, T, C): Gradients flowing back from the next layer
 *   inp - Input tensor (B, T): Token IDs (same as in forward pass, needed for indexing)
 *   B, T, C - Dimensions (same as forward pass)
 *
 * Note: Gradients are accumulated (+=) because the same token/position embeddings
 *       may be used multiple times across the batch.
 */
void encoder_backward(float* dwte, float* dwpe,
                      float* dout, int* inp,
                      int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            int ix = inp[b * T + t];
            float* dwte_ix = dwte + ix * C;
            float* dwpe_t = dwpe + t * C;
            for (int i = 0; i < C; i++) {
                float d = dout_bt[i];
                dwte_ix[i] += d;  // Accumulate gradient to token embedding
                dwpe_t[i] += d;   // Accumulate gradient to position embedding
            }
        }
    }
}

/**
 * Layer Normalization Forward Pass
 *
 * Purpose: Normalizes activations across the channel dimension to stabilize training.
 * Layer normalization is crucial for training deep transformers, helping with gradient flow.
 *
 * Algorithm:
 * For each position (b,t), treating the C-dimensional vector independently:
 *   1. Compute mean: μ = (1/C) * Σ x[i]
 *   2. Compute variance: σ² = (1/C) * Σ (x[i] - μ)²
 *   3. Normalize: x_norm[i] = (x[i] - μ) / sqrt(σ² + ε)
 *   4. Scale and shift: out[i] = γ[i] * x_norm[i] + β[i]
 *
 * Parameters:
 *   out - Output tensor (B, T, C): Normalized, scaled, and shifted activations
 *   mean - Output buffer (B, T): Mean values (cached for backward pass)
 *   rstd - Output buffer (B, T): Reciprocal std dev 1/sqrt(σ² + ε) (cached for backward)
 *   inp - Input tensor (B, T, C): Input activations to normalize
 *   weight - Weight tensor (C,): Learnable scale parameters (γ)
 *   bias - Bias tensor (C,): Learnable shift parameters (β)
 *   B, T, C - Dimensions
 *
 * Rationale:
 *   - eps (1e-5) prevents division by zero when variance is very small
 *   - Normalization is per-position, not across the batch (unlike BatchNorm)
 *   - mean and rstd are cached because they're needed for efficient backward pass
 *
 * Reference: "Layer Normalization" (Ba et al., 2016)
 *            https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
 */
void layernorm_forward(float* out, float* mean, float* rstd,
                       float* inp, float* weight, float* bias,
                       int B, int T, int C) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // seek to the input position inp[b,t,:]
            float* x = inp + b * T * C + t * C;
            // calculate the mean
            float m = 0.0f;
            for (int i = 0; i < C; i++) {
                m += x[i];
            }
            m = m/C;
            // calculate the variance (without any bias correction)
            float v = 0.0f;
            for (int i = 0; i < C; i++) {
                float xshift = x[i] - m;
                v += xshift * xshift;
            }
            v = v/C;
            // calculate the rstd (reciprocal standard deviation)
            float s = 1.0f / sqrtf(v + eps);
            // seek to the output position in out[b,t,:]
            float* out_bt = out + b * T * C + t * C;
            for (int i = 0; i < C; i++) {
                float n = (s * (x[i] - m)); // normalize
                float o = n * weight[i] + bias[i]; // scale and shift
                out_bt[i] = o; // write
            }
            // cache the mean and rstd for the backward pass later
            mean[b * T + t] = m;
            rstd[b * T + t] = s;
        }
    }
}

/**
 * Layer Normalization Backward Pass
 *
 * Purpose: Backpropagates gradients through layer normalization, computing gradients
 * with respect to inputs, weights (scale), and biases (shift).
 *
 * Algorithm:
 * The backward pass for layer norm is mathematically complex because each output
 * depends on all inputs through the mean and variance. The gradient w.r.t. input is:
 *   dinp[i] = (1/σ) * (dnorm[i] - mean(dnorm) - norm[i] * mean(dnorm * norm))
 * where norm[i] = (inp[i] - μ) / σ
 *
 * This implementation uses two passes:
 *   Pass 1: Compute two reduction terms needed for the input gradient
 *   Pass 2: Compute actual gradients for weight, bias, and input
 *
 * Parameters:
 *   dinp - Gradient tensor (B, T, C): Gradients w.r.t. inputs
 *   dweight - Gradient tensor (C,): Gradients w.r.t. scale parameters
 *   dbias - Gradient tensor (C,): Gradients w.r.t. shift parameters
 *   dout - Gradient tensor (B, T, C): Gradients from upstream layer
 *   inp - Input tensor (B, T, C): Original inputs (from forward pass)
 *   weight - Weight tensor (C,): Scale parameters (from forward pass)
 *   mean - Buffer (B, T): Cached mean values from forward pass
 *   rstd - Buffer (B, T): Cached reciprocal std dev from forward pass
 *   B, T, C - Dimensions
 *
 * Rationale:
 *   - Two-pass algorithm enables efficient computation of the complex gradient formula
 *   - Gradients are accumulated (+=) to support residual connections
 *   - Uses cached mean and rstd from forward pass for efficiency
 */
void layernorm_backward(float* dinp, float* dweight, float* dbias,
                        float* dout, float* inp, float* weight, float* mean, float* rstd,
                        int B, int T, int C) {
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dout_bt = dout + b * T * C + t * C;
            float* inp_bt = inp + b * T * C + t * C;
            float* dinp_bt = dinp + b * T * C + t * C;
            float mean_bt = mean[b * T + t];
            float rstd_bt = rstd[b * T + t];

            // first: two reduce operations
            // Compute mean of gradients and mean of (gradient * normalized_value)
            // These are needed for the input gradient formula
            float dnorm_mean = 0.0f;
            float dnorm_norm_mean = 0.0f;
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                dnorm_mean += dnorm_i;
                dnorm_norm_mean += dnorm_i * norm_bti;
            }
            dnorm_mean = dnorm_mean / C;
            dnorm_norm_mean = dnorm_norm_mean / C;

            // now iterate again and accumulate all the gradients
            for (int i = 0; i < C; i++) {
                float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
                float dnorm_i = weight[i] * dout_bt[i];
                // gradient contribution to bias
                dbias[i] += dout_bt[i];
                // gradient contribution to weight
                dweight[i] += norm_bti * dout_bt[i];
                // gradient contribution to input (three-term formula)
                float dval = 0.0f;
                dval += dnorm_i; // term 1: direct gradient
                dval -= dnorm_mean; // term 2: mean correction
                dval -= norm_bti * dnorm_norm_mean; // term 3: variance correction
                dval *= rstd_bt; // final scale by reciprocal std dev
                dinp_bt[i] += dval;
            }
        }
    }
}

/**
 * Matrix Multiplication Forward Pass (Naive Implementation)
 *
 * Purpose: Performs matrix multiplication with optional bias, serving as an algorithmic
 * reference and fallback for cases where the optimized version cannot be used.
 *
 * Algorithm:
 * Computes: out = inp @ weight^T + bias
 * For each position (b,t) and output channel o:
 *   out[b,t,o] = Σ(inp[b,t,i] * weight[o,i]) + bias[o]
 *
 * Parameters:
 *   out - Output tensor (B, T, OC): Result of matrix multiplication
 *   inp - Input tensor (B, T, C): Input activations
 *   weight - Weight tensor (OC, C): Weight matrix (note: row-major storage)
 *   bias - Bias tensor (OC,): Optional bias vector (can be NULL)
 *   B - Batch size
 *   T - Sequence length
 *   C - Input channels
 *   OC - Output channels
 *
 * Shape transformations:
 *   Input:  (B, T, C)
 *   Weight: (OC, C)
 *   Output: (B, T, OC)
 *
 * Note: Uses OpenMP to parallelize over B and T dimensions for modest speedup.
 */
void matmul_forward_naive(float* out,
                         const float* inp, const float* weight, const float* bias,
                         int B, int T, int C, int OC) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int bt = b * T + t;
            for (int o = 0; o < OC; o++) {
                float val = (bias != NULL) ? bias[o] : 0.0f;
                for (int i = 0; i < C; i++) {
                    val += inp[bt * C + i] * weight[o*C + i];
                }
                out[bt * OC + o] = val;
            }
        }
    }
}

/**
 * Matrix Multiplication Forward Pass (Optimized Implementation)
 *
 * Purpose: Performs the same operation as matmul_forward_naive but with mild optimizations
 * to improve performance. This is one of the computational hotspots of GPT-2 training.
 *
 * Algorithm:
 * Same as naive version: out = inp @ weight^T + bias
 * But uses loop tiling/unrolling to improve cache reuse and enable compiler optimizations.
 *
 * Optimization strategy:
 *   1. Collapse B and T dimensions into a single loop dimension (B*T positions)
 *   2. Process LOOP_UNROLL positions simultaneously in the inner loop
 *   3. Load each weight value once and reuse it LOOP_UNROLL times
 *   4. Keep intermediate results in local array (likely in registers)
 *   5. Compiler with -Ofast will fuse multiply-add into FMA instructions
 *
 * Parameters: (same as matmul_forward_naive)
 *   out - Output tensor (B, T, OC)
 *   inp - Input tensor (B, T, C)
 *   weight - Weight tensor (OC, C)
 *   bias - Bias tensor (OC,) or NULL
 *   B, T, C, OC - Dimensions
 *
 * Rationale:
 *   - Falls back to naive version if B*T not divisible by LOOP_UNROLL (for correctness)
 *   - LOOP_UNROLL=8 chosen as good tradeoff between register pressure and reuse
 *   - Weight reuse reduces memory bandwidth requirements
 *   - OpenMP parallelizes over the B*T dimension
 */
void matmul_forward(float* out,
                    const float* inp, const float* weight, const float* bias,
                    int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_backward
    // therefore, the implementation below is very mildly optimized
    // this function is otherwise identical to that of matmul_forward_naive()
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    // make sure the tiled loop will be correct or fallback to naive version
    const int LOOP_UNROLL = 8;
    if (B*T % LOOP_UNROLL != 0) {
        matmul_forward_naive(out, inp, weight, bias, B, T, C, OC);
        return;
    }

    // collapse the B and T loops into one and turn it into a strided loop.
    // then we can tile the inner loop, and reuse the loaded weight LOOP_UNROLL many times
    #pragma omp parallel for
    for (int obt = 0; obt < B * T; obt += LOOP_UNROLL) {
        for (int o = 0; o < OC; o++) {
            // we'll keep LOOP_UNROLL many results in registers
            float result[LOOP_UNROLL];
            // initialize the bias, if it exists
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                result[ibt] = (bias != NULL) ? bias[o] : 0.0f;
            }
            // inner loops. Because we do LOOP_UNROLL steps of inner bt, we can cache
            // the value of weight[i + o * C] and reuse it.
            // we compile with -Ofast, so the compiler will turn the inner loop into FMAs
            for (int i = 0; i < C; i++) {
                float w = weight[i + o * C];
                for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                    int bt = obt + ibt;
                    result[ibt] += inp[bt * C + i] * w;
                }
            }
            // write back results to main memory
            for (int ibt = 0; ibt < LOOP_UNROLL; ibt++) {
                int bt = obt + ibt;
                out[bt * OC + o] = result[ibt];
            }
        }
    }
}

/**
 * Matrix Multiplication Backward Pass
 *
 * Purpose: Backpropagates gradients through matrix multiplication, computing gradients
 * with respect to inputs, weights, and bias. This is the other computational hotspot
 * (along with matmul_forward) in GPT-2 training.
 *
 * Algorithm:
 * Given: dout (gradient w.r.t. output), inp (from forward pass), weight (from forward pass)
 * Compute:
 *   dinp[b,t,i] = Σ_o (dout[b,t,o] * weight[o,i])     - gradient w.r.t. input
 *   dweight[o,i] = Σ_{b,t} (dout[b,t,o] * inp[b,t,i]) - gradient w.r.t. weight
 *   dbias[o] = Σ_{b,t} dout[b,t,o]                     - gradient w.r.t. bias
 *
 * Implementation strategy:
 * Splits into two separate loops for efficient parallelization:
 *   Loop 1: Compute dinp, parallelized over (B, T)
 *   Loop 2: Compute dweight and dbias, parallelized over OC
 *
 * Parameters:
 *   dinp - Gradient tensor (B, T, C): Gradients w.r.t. inputs
 *   dweight - Gradient tensor (OC, C): Gradients w.r.t. weights
 *   dbias - Gradient tensor (OC,): Gradients w.r.t. bias (can be NULL)
 *   dout - Gradient tensor (B, T, OC): Gradients from upstream layer
 *   inp - Input tensor (B, T, C): Inputs from forward pass
 *   weight - Weight tensor (OC, C): Weights from forward pass
 *   B, T, C, OC - Dimensions
 *
 * Rationale:
 *   - Two-loop structure enables better parallelization (avoid write conflicts)
 *   - Could be done in one loop but would be harder to parallelize efficiently
 *   - Gradients accumulated (+=) to support gradient accumulation and residual connections
 */
void matmul_backward(float* dinp, float* dweight, float* dbias,
                     const float* dout, const float* inp, const float* weight,
                     int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            const float* dout_bt = dout + b * T * OC + t * OC;
            float* dinp_bt = dinp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                const float* wrow = weight + o*C;
                float d = dout_bt[o];
                for (int i = 0; i < C; i++) {
                    dinp_bt[i] += wrow[i] * d;
                }
            }
        }
    }
    // backward into weight/bias, parallelize over output channels OC
    #pragma omp parallel for
    for (int o = 0; o < OC; o++) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                const float* dout_bt = dout + b * T * OC + t * OC;
                const float* inp_bt = inp + b * T * C + t * C;
                float* dwrow = dweight + o*C;
                float d = dout_bt[o];
                if (dbias != NULL) { dbias[o] += d; }
                for (int i = 0; i < C; i++) {
                    dwrow[i] += inp_bt[i] * d;
                }
            }
        }
    }
}

/**
 * Multi-Head Self-Attention Forward Pass
 *
 * Purpose: Implements the core attention mechanism of transformers. This is the only layer
 * that allows tokens to communicate with each other across time. Every other layer operates
 * independently at each position.
 *
 * Algorithm:
 * Multi-head scaled dot-product attention with causal masking:
 *   1. For each head h and position t, compute attention scores with all positions t2 <= t
 *      score[t, t2] = (Q[t] · K[t2]) / sqrt(head_size)
 *   2. Apply softmax to get attention weights: att[t, t2] = softmax(scores[t, :t])
 *   3. Compute weighted sum of values: out[t] = Σ(att[t, t2] * V[t2])
 *
 * Four-pass implementation:
 *   Pass 1: Compute Q·K scores and track max (for numerical stability)
 *   Pass 2: Compute exp(score - max) and sum
 *   Pass 3: Normalize to get softmax probabilities
 *   Pass 4: Weighted sum of values
 *
 * Parameters:
 *   out - Output tensor (B, T, C): Attention output
 *   preatt - Buffer (B, NH, T, T): Pre-softmax scores (cached for backward)
 *   att - Buffer (B, NH, T, T): Post-softmax attention weights (cached for backward)
 *   inp - Input tensor (B, T, 3*C): Concatenated Q, K, V vectors
 *         Layout: [Q (0:C), K (C:2C), V (2C:3C)]
 *   B - Batch size
 *   T - Sequence length
 *   C - Channels (total across all heads)
 *   NH - Number of attention heads
 *
 * Shape transformations:
 *   Input:  (B, T, 3*C) - interleaved Q, K, V
 *   Output: (B, T, C)
 *   Each head processes (hs,) vectors where hs = C / NH
 *
 * Rationale:
 *   - Causal mask (t2 <= t) ensures autoregressive property: position t only sees past/present
 *   - Scaling by 1/sqrt(hs) prevents dot products from growing too large
 *   - Max subtraction in softmax prevents numerical overflow in exp()
 *   - Multi-head design allows attending to different representation subspaces
 *
 * Reference: "Attention Is All You Need" (Vaswani et al., 2017)
 */
void attention_forward(float* out, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    #pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;

                // pass 1: calculate query dot key and maxval
                float maxval = -10000.0f; // TODO something better
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                    // (query_t) dot (key_t2)
                    float val = 0.0f;
                    for (int i = 0; i < hs; i++) {
                        val += query_t[i] * key_t2[i];
                    }
                    val *= scale;
                    if (val > maxval) {
                        maxval = val;
                    }

                    preatt_bth[t2] = val;
                }

                // pass 2: calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                float expsum = 0.0f;
                for (int t2 = 0; t2 <= t; t2++) {
                    float expv = expf(preatt_bth[t2] - maxval);
                    expsum += expv;
                    att_bth[t2] = expv;
                }
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                // pass 3: normalize to get the softmax
                for (int t2 = 0; t2 < T; t2++) {
                    if (t2 <= t) {
                        att_bth[t2] *= expsum_inv;
                    } else {
                        // causal attention mask. not strictly necessary to set to zero here
                        // only doing this explicitly for debugging and checking to PyTorch
                        att_bth[t2] = 0.0f;
                    }
                }

                // pass 4: accumulate weighted values into the output of attention
                float* out_bth = out + b * T * C + t * C + h * hs;
                for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float att_btht2 = att_bth[t2];
                    for (int i = 0; i < hs; i++) {
                        out_bth[i] += att_btht2 * value_t2[i];
                    }
                }
            }
        }
    }
}

/**
 * Multi-Head Self-Attention Backward Pass
 *
 * Purpose: Backpropagates gradients through the attention mechanism, computing gradients
 * with respect to the Q, K, V inputs. This implements the reverse of the four-pass
 * forward attention algorithm.
 *
 * Algorithm:
 * Reverses the four forward passes in backward order:
 *   Pass 4 (backward): Gradient through weighted value sum
 *     - datt[t2] from: out[i] = Σ(att[t2] * value[t2,i])
 *     - dvalue from: out[i] = Σ(att[t2] * value[t2,i])
 *   Pass 3 & 2 (backward): Gradient through softmax
 *     - Uses Jacobian of softmax: ∂softmax_i/∂x_j = softmax_i * (δ_ij - softmax_j)
 *   Pass 1 (backward): Gradient through Q·K dot product
 *     - dquery[i] from: score = Σ(query[i] * key[i])
 *     - dkey[i] from: score = Σ(query[i] * key[i])
 *
 * Parameters:
 *   dinp - Gradient tensor (B, T, 3*C): Gradients w.r.t. Q, K, V inputs
 *   dpreatt - Gradient buffer (B, NH, T, T): Gradients w.r.t. pre-softmax scores
 *   datt - Gradient buffer (B, NH, T, T): Gradients w.r.t. attention weights
 *   dout - Gradient tensor (B, T, C): Gradients from upstream layer
 *   inp - Input tensor (B, T, 3*C): Q, K, V from forward pass
 *   att - Attention weights (B, NH, T, T): From forward pass
 *   B, T, C, NH - Dimensions
 *
 * Rationale:
 *   - Softmax backward uses the identity: d(softmax)/dx doesn't need input x,
 *     only needs the output softmax values
 *   - All gradients accumulated (+=) to support gradient flow through residual connections
 *   - Scale factor applied in Q·K backward to match forward scaling
 *
 * Note: This is NOT parallelized (no OpenMP) unlike forward pass, as backward is
 *       already part of a larger backward pass loop that's parallelized at a higher level.
 */
void attention_backward(float* dinp, float* dpreatt, float* datt,
                        float* dout, float* inp, float* att,
                        int B, int T, int C, int NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.f / sqrtf(hs);

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int h = 0; h < NH; h++) {
                float* att_bth = att + b*NH*T*T + h*T*T + t*T;
                float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
                float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
                float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
                float* query_t = inp + b * T * C3 + t * C3 + h * hs;

                // backward pass 4, through the value accumulation
                float* dout_bth = dout + b * T * C + t * C + h * hs;
                for (int t2 = 0; t2 <= t; t2++) {
                    float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
                    float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // out_bth[i] += att_bth[t2] * value_t2[i];
                        // so now we have:
                        datt_bth[t2] += value_t2[i] * dout_bth[i];
                        dvalue_t2[i] += att_bth[t2] * dout_bth[i];
                    }
                }

                // backward pass 2 & 3, the softmax
                // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
                for (int t2 = 0; t2 <= t; t2++) {
                    for (int t3 = 0; t3 <= t; t3++) {
                        float indicator = t2 == t3 ? 1.0f : 0.0f;
                        float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
                        dpreatt_bth[t3] += local_derivative * datt_bth[t2];
                    }
                }

                // backward pass 1, the query @ key matmul
                for (int t2 = 0; t2 <= t; t2++) {
                    float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
                    for (int i = 0; i < hs; i++) {
                        // in the forward pass this was:
                        // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                        // so now we have:
                        dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
                        dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
                    }
                }
            }
        }
    }
}

/**
 * GELU Activation Forward Pass
 *
 * Purpose: Applies the Gaussian Error Linear Unit (GELU) activation function, which is
 * the non-linearity used in GPT-2's MLP blocks. GELU is smoother than ReLU and has been
 * shown to work better in transformers.
 *
 * Algorithm:
 * Uses the tanh approximation of GELU (faster than the exact erf-based version):
 *   GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
 *
 * Parameters:
 *   out - Output tensor (N,): Activated values
 *   inp - Input tensor (N,): Input values
 *   N - Number of elements
 *
 * Rationale:
 *   - GELU is smooth and differentiable everywhere (unlike ReLU)
 *   - The approximation is very close to exact GELU but much faster to compute
 *   - Used in BERT, GPT-2, GPT-3, and many other modern transformers
 *
 * Reference: "Gaussian Error Linear Units (GELUs)" (Hendrycks & Gimpel, 2016)
 */
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float* out, float* inp, int N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}

/**
 * GELU Activation Backward Pass
 *
 * Purpose: Backpropagates gradients through the GELU activation function.
 *
 * Algorithm:
 * Computes the derivative of the GELU approximation:
 *   d(GELU)/dx = 0.5 * (1 + tanh(arg)) + x * 0.5 * sech²(arg) * d(arg)/dx
 * where arg = √(2/π) * (x + 0.044715 * x³)
 * and sech²(arg) = 1 / cosh²(arg)
 *
 * Parameters:
 *   dinp - Gradient tensor (N,): Gradients w.r.t. inputs
 *   inp - Input tensor (N,): Inputs from forward pass
 *   dout - Gradient tensor (N,): Gradients from upstream layer
 *   N - Number of elements
 *
 * Note: This function has special compiler pragmas to disable certain optimizations
 *       that cause numerical issues with the GELU backward computation.
 */
// we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
#pragma float_control(precise, on, push)
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-finite-math-only")))
#endif
void gelu_backward(float* dinp, float* inp, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
        float tanh_out = tanhf(tanh_arg);
        float coshf_out = coshf(tanh_arg);
        float sech_out = 1.0f / (coshf_out * coshf_out);
        float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
        dinp[i] += local_grad * dout[i];
    }
}
#pragma float_control(pop)

/**
 * Residual Connection Forward Pass
 *
 * Purpose: Implements a residual (skip) connection by element-wise addition of two tensors.
 * Residual connections are critical for training deep networks, allowing gradients to flow
 * directly through the network.
 *
 * Algorithm:
 *   out[i] = inp1[i] + inp2[i]
 *
 * Parameters:
 *   out - Output tensor (N,): Sum of inputs
 *   inp1 - Input tensor 1 (N,): First input (typically the residual/skip connection)
 *   inp2 - Input tensor 2 (N,): Second input (typically output from a layer)
 *   N - Number of elements
 *
 * Rationale:
 *   - Residual connections enable training of very deep networks (ResNet, Transformer)
 *   - Help gradients flow backwards through many layers without vanishing
 *   - In GPT-2: Used after attention and after MLP in each transformer block
 *
 * Reference: "Deep Residual Learning for Image Recognition" (He et al., 2015)
 */
void residual_forward(float* out, float* inp1, float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}

/**
 * Residual Connection Backward Pass
 *
 * Purpose: Backpropagates gradients through the residual connection. Since forward
 * is addition, backward simply copies gradients to both inputs.
 *
 * Algorithm:
 *   dinp1[i] += dout[i]  (gradient flows to first input)
 *   dinp2[i] += dout[i]  (gradient flows to second input)
 *
 * Parameters:
 *   dinp1 - Gradient tensor 1 (N,): Gradients w.r.t. first input
 *   dinp2 - Gradient tensor 2 (N,): Gradients w.r.t. second input
 *   dout - Gradient tensor (N,): Gradients from upstream layer
 *   N - Number of elements
 *
 * Rationale:
 *   - Derivative of addition: both branches get the same gradient
 *   - This allows gradients to flow through both the residual path and the layer path
 *   - Gradients accumulated (+=) to support multiple connections
 */
void residual_backward(float* dinp1, float* dinp2, float* dout, int N) {
    for (int i = 0; i < N; i++) {
        dinp1[i] += dout[i];
        dinp2[i] += dout[i];
    }
}

/**
 * Softmax Forward Pass
 *
 * Purpose: Converts logits (unnormalized scores) into probabilities that sum to 1.
 * This is applied to the final output layer to produce a probability distribution over
 * the vocabulary for next-token prediction.
 *
 * Algorithm:
 * For each position (b,t):
 *   1. Find max logit value (for numerical stability)
 *   2. Compute exp(logit - max) for each vocabulary item
 *   3. Normalize by sum to get probabilities
 *   softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)
 *
 * Parameters:
 *   probs - Output tensor (B, T, Vp): Probability distributions (each sums to 1.0)
 *   logits - Input tensor (B, T, Vp): Unnormalized log probabilities
 *   B - Batch size
 *   T - Sequence length
 *   V - Vocabulary size (actual, e.g., 50257)
 *   Vp - Padded vocabulary size (for memory alignment, e.g., 50304)
 *
 * Rationale:
 *   - Subtracting max prevents numerical overflow in exp() computation
 *   - Only compute over V elements (actual vocab), set padding (V:Vp) to zero
 *   - Padding exists for computational efficiency (e.g., making Vp divisible by 128)
 *   - Parallelized over batch and time dimensions with OpenMP
 *
 * Note: This is the final layer before computing the loss during training, or before
 *       sampling during generation.
 */
void softmax_forward(float* probs, float* logits, int B, int T, int V, int Vp) {
    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // probs <- softmax(logits)
            float* logits_bt = logits + b * T * Vp + t * Vp;
            float* probs_bt = probs + b * T * Vp + t * Vp;

            // maxval is only calculated and subtracted for numerical stability
            float maxval = -10000.0f; // TODO something better
            for (int i = 0; i < V; i++) {
                if (logits_bt[i] > maxval) {
                    maxval = logits_bt[i];
                }
            }
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
                probs_bt[i] = expf(logits_bt[i] - maxval);
                sum += probs_bt[i];
            }
            // note we only loop to V, leaving the padded dimensions
            for (int i = 0; i < V; i++) {
                probs_bt[i] /= sum;
            }
            // for extra super safety we may wish to include this too,
            // forcing the probabilities here to be zero, but it shouldn't matter
            for (int i = V; i < Vp; i++) {
                probs_bt[i] = 0.0f;
            }
        }
    }
}

/**
 * Cross-Entropy Loss Forward Pass
 *
 * Purpose: Computes the cross-entropy loss between predicted probabilities and target tokens.
 * This is the training objective for language modeling: maximize the probability of the
 * correct next token.
 *
 * Algorithm:
 * For each position (b,t):
 *   loss = -log(probability of correct token)
 *   loss[b,t] = -log(probs[b, t, targets[b,t]])
 *
 * Parameters:
 *   losses - Output tensor (B, T): Individual loss at each position
 *   probs - Input tensor (B, T, Vp): Probability distributions from softmax
 *   targets - Input tensor (B, T): Correct token indices
 *   B - Batch size
 *   T - Sequence length
 *   Vp - Padded vocabulary size
 *
 * Rationale:
 *   - Negative log probability is minimized when model assigns high probability to correct token
 *   - Loss is computed independently at each position
 *   - Final training loss is the mean of all positions
 *
 * Reference: Standard information theory / maximum likelihood estimation
 */
void crossentropy_forward(float* losses,
                          float* probs, int* targets,
                          int B, int T, int Vp) {
    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            // loss = -log(probs[target])
            float* probs_bt = probs + b * T * Vp + t * Vp;
            int ix = targets[b * T + t];
            losses[b * T + t] = -logf(probs_bt[ix]);
        }
    }
}

/**
 * Cross-Entropy and Softmax Backward Pass (Fused)
 *
 * Purpose: Efficiently backpropagates through both cross-entropy loss and softmax in
 * a single pass. This fusion is mathematically elegant and computationally efficient.
 *
 * Algorithm:
 * The combined derivative of cross-entropy and softmax has a simple form:
 *   dlogits[i] = probs[i] - 1  (if i == target index)
 *   dlogits[i] = probs[i]      (if i != target index)
 * Or more compactly: dlogits[i] = (probs[i] - indicator) * dloss
 * where indicator = 1 if i is the target, 0 otherwise
 *
 * Parameters:
 *   dlogits - Gradient tensor (B, T, Vp): Gradients w.r.t. logits
 *   dlosses - Gradient tensor (B, T): Gradients w.r.t. losses (typically 1/(B*T))
 *   probs - Probability tensor (B, T, Vp): Softmax outputs from forward pass
 *   targets - Target tensor (B, T): Correct token indices
 *   B, T, V, Vp - Dimensions
 *
 * Rationale:
 *   - Fusing cross-entropy and softmax backward is more efficient than separate passes
 *   - The gradient formula is elegant: probs[i] - indicator
 *   - Only operates on actual vocabulary V, leaving padding region at zero
 *   - This is a standard optimization in deep learning frameworks
 *
 * Reference: Common in ML literature, e.g., CS231n notes on softmax and cross-entropy
 */
void crossentropy_softmax_backward(float* dlogits,
                           float* dlosses, float* probs, int* targets,
                           int B, int T, int V, int Vp) {
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * Vp + t * Vp;
            float* probs_bt = probs + b * T * Vp + t * Vp;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

/**
 * GPT2Config: Model hyperparameters
 *
 * Defines the architecture of the GPT-2 model. These values are read from the
 * checkpoint file and determine the model's capacity and behavior.
 */
typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;

/**
 * ParameterTensors: All learnable parameters of the model
 *
 * This struct holds pointers to all weight matrices and bias vectors in GPT-2.
 * All parameters are stored in a single contiguous memory block, with these
 * pointers indicating where each parameter tensor starts.
 *
 * Parameter organization:
 * - Embeddings: wte (token), wpe (position)
 * - For each of L layers:
 *   - Pre-attention LayerNorm: ln1w, ln1b
 *   - Attention: qkvw, qkvb (projects to Q,K,V), attprojw, attprojb (output projection)
 *   - Pre-MLP LayerNorm: ln2w, ln2b
 *   - MLP: fcw, fcb (expansion to 4*C), fcprojw, fcprojb (projection back to C)
 * - Final LayerNorm: lnfw, lnfb
 *
 * Note: The same ParameterTensors struct is reused for gradients (grads), where
 *       each pointer points to the gradient of the corresponding parameter.
 */
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float* wte; // (V, C) - token embeddings
    float* wpe; // (maxT, C) - position embeddings
    float* ln1w; // (L, C) - layernorm1 weights (scale)
    float* ln1b; // (L, C) - layernorm1 biases (shift)
    float* qkvw; // (L, 3*C, C) - qkv projection weights
    float* qkvb; // (L, 3*C) - qkv projection biases
    float* attprojw; // (L, C, C) - attention output projection weights
    float* attprojb; // (L, C) - attention output projection biases
    float* ln2w; // (L, C) - layernorm2 weights
    float* ln2b; // (L, C) - layernorm2 biases
    float* fcw; // (L, 4*C, C) - MLP first layer weights (expansion)
    float* fcb; // (L, 4*C) - MLP first layer biases
    float* fcprojw; // (L, C, 4*C) - MLP second layer weights (projection)
    float* fcprojb; // (L, C) - MLP second layer biases
    float* lnfw; // (C) - final layernorm weights
    float* lnfb; // (C) - final layernorm biases
} ParameterTensors;

/**
 * Fill in parameter sizes
 *
 * Purpose: Calculates the number of elements in each parameter tensor based on model config.
 * This is used to allocate memory and to partition the memory into individual tensors.
 *
 * Parameters:
 *   param_sizes - Output array[NUM_PARAMETER_TENSORS]: Will be filled with sizes
 *   config - Model configuration specifying architecture dimensions
 */
void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
}

/**
 * Allocate memory for parameters and set up pointers
 *
 * Purpose: Allocates a single contiguous block of memory for all parameters and sets up
 * the ParameterTensors struct to point to the appropriate locations within that block.
 *
 * Algorithm:
 *   1. Sum all parameter sizes to get total memory needed
 *   2. Allocate one large block of memory
 *   3. Partition the block by setting each pointer in ParameterTensors
 *
 * Parameters:
 *   params - ParameterTensors struct to be populated with pointers
 *   param_sizes - Array of sizes for each parameter tensor
 *
 * Returns:
 *   Pointer to the beginning of the allocated memory block
 *
 * Rationale:
 *   - Single allocation is more efficient than many small allocations
 *   - Contiguous memory improves cache locality
 *   - Makes it easy to save/load all parameters as one block
 */
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    float* params_memory = (float*)mallocCheck(num_parameters * sizeof(float));
    // assign all the tensors
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

/**
 * ActivationTensors: All intermediate activations during forward pass
 *
 * This struct holds pointers to all intermediate values computed during the forward pass.
 * These values are needed for the backward pass to compute gradients. Like parameters,
 * all activations are stored in one contiguous memory block.
 *
 * Activation flow through one transformer block:
 *   residual -> ln1 -> qkv -> attention -> atty -> attproj -> residual2
 *   residual2 -> ln2 -> fch -> fch_gelu -> fcproj -> residual3
 *
 * The same struct is reused for activation gradients (grads_acts).
 *
 * Note: Activations are batch-dependent (include B, T dimensions), while parameters are not.
 */
#define NUM_ACTIVATION_TENSORS 23
typedef struct {
    float* encoded; // (B, T, C) - output of encoder (token + position embeddings)
    float* ln1; // (L, B, T, C) - output of first layernorm in each layer
    float* ln1_mean; // (L, B, T) - mean values from ln1 (cached for backward)
    float* ln1_rstd; // (L, B, T) - reciprocal std dev from ln1 (cached for backward)
    float* qkv; // (L, B, T, 3*C) - query, key, value vectors (concatenated)
    float* atty; // (L, B, T, C) - output of attention (before projection)
    float* preatt; // (L, B, NH, T, T) - attention scores before softmax
    float* att; // (L, B, NH, T, T) - attention weights after softmax
    float* attproj; // (L, B, T, C) - attention output after projection
    float* residual2; // (L, B, T, C) - residual after adding attention
    float* ln2; // (L, B, T, C) - output of second layernorm in each layer
    float* ln2_mean; // (L, B, T) - mean values from ln2
    float* ln2_rstd; // (L, B, T) - reciprocal std dev from ln2
    float* fch; // (L, B, T, 4*C) - MLP first layer output (before GELU)
    float* fch_gelu; // (L, B, T, 4*C) - MLP first layer output (after GELU)
    float* fcproj; // (L, B, T, C) - MLP second layer output (projection)
    float* residual3; // (L, B, T, C) - residual after adding MLP output
    float* lnf; // (B, T, C) - output of final layernorm
    float* lnf_mean; // (B, T) - mean values from final layernorm
    float* lnf_rstd; // (B, T) - reciprocal std dev from final layernorm
    float* logits; // (B, T, Vp) - output logits over vocabulary
    float* probs; // (B, T, Vp) - probabilities after softmax
    float* losses; // (B, T) - cross-entropy loss at each position
} ActivationTensors;

/**
 * Fill in activation sizes
 *
 * Purpose: Calculates the number of elements in each activation tensor based on model
 * config and batch dimensions. Similar to fill_in_parameter_sizes but depends on B and T.
 *
 * Parameters:
 *   act_sizes - Output array[NUM_ACTIVATION_TENSORS]: Will be filled with sizes
 *   config - Model configuration
 *   B - Batch size
 *   T - Sequence length
 */
void fill_in_activation_sizes(size_t* act_sizes, GPT2Config config, int B, int T) {
    size_t C = config.channels;
    size_t NH = config.num_heads;
    size_t L = config.num_layers;
    size_t Vp = config.padded_vocab_size;
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = L * B * T; // ln1_mean
    act_sizes[3] = L * B * T; // ln1_rstd
    act_sizes[4] = L * B * T * 3 * C; // qkv
    act_sizes[5] = L * B * T * C; // atty
    act_sizes[6] = L * B * NH * T * T; // preatt
    act_sizes[7] = L * B * NH * T * T; // att
    act_sizes[8] = L * B * T * C; // attproj
    act_sizes[9] = L * B * T * C; // residual2
    act_sizes[10] = L * B * T * C; // ln2
    act_sizes[11] = L * B * T; // ln2_mean
    act_sizes[12] = L * B * T; // ln2_rstd
    act_sizes[13] = L * B * T * 4 * C; // fch
    act_sizes[14] = L * B * T * 4 * C; // fch_gelu
    act_sizes[15] = L * B * T * C; // fcproj
    act_sizes[16] = L * B * T * C; // residual3
    act_sizes[17] = B * T * C; // lnf
    act_sizes[18] = B * T; // lnf_mean
    act_sizes[19] = B * T; // lnf_rstd
    act_sizes[20] = B * T * Vp; // logits
    act_sizes[21] = B * T * Vp; // probs
    act_sizes[22] = B * T; // losses
}

/**
 * Allocate memory for activations and set up pointers
 *
 * Purpose: Allocates a single contiguous block for all activations and sets up the
 * ActivationTensors struct to point to appropriate locations. Same pattern as parameters.
 *
 * Parameters:
 *   acts - ActivationTensors struct to be populated
 *   act_sizes - Array of sizes for each activation tensor
 *
 * Returns:
 *   Pointer to the beginning of the allocated memory block
 */
float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    float* acts_memory = (float*)mallocCheck(num_activations * sizeof(float));
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

/**
 * GPT2: The complete model state
 *
 * This struct contains everything needed to train or run inference with GPT-2:
 * - Model architecture configuration
 * - All parameters (weights and biases)
 * - All gradients (for training)
 * - Optimizer state (first and second moments for AdamW)
 * - Activations and activation gradients (forward/backward pass intermediates)
 * - Current batch state
 *
 * Memory organization:
 * - Parameters, gradients, and activations each use a single contiguous memory block
 * - The *Tensors structs just hold pointers into these blocks
 * - Optimizer state (m_memory, v_memory) allocated lazily on first update
 *
 * Lazy allocation:
 * - Activations allocated on first forward pass (depends on B, T)
 * - Gradients allocated on first backward pass
 * - Optimizer state allocated on first update
 */
typedef struct {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    float* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;  // first moment estimates
    float* v_memory;  // second moment estimates
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    float* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass
    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
} GPT2;

/**
 * Build GPT-2 Model from Checkpoint
 *
 * Purpose: Loads a pre-trained GPT-2 model from a binary checkpoint file. This initializes
 * the model structure and loads all parameters from disk.
 *
 * Checkpoint file format:
 *   - Header (256 ints):
 *     [0]: Magic number (20240326) for file format validation
 *     [1]: Version number (must be 3)
 *     [2]: max_seq_len
 *     [3]: vocab_size
 *     [4]: num_layers
 *     [5]: num_heads
 *     [6]: channels
 *     [7]: padded_vocab_size
 *   - Parameters: All model weights as floats, in the order defined by ParameterTensors
 *
 * Parameters:
 *   model - GPT2 struct to be initialized
 *   checkpoint_path - Path to the checkpoint file (e.g., "gpt2_124M.bin")
 *
 * Post-conditions:
 *   - model->config filled with architecture info
 *   - model->params_memory allocated and loaded with weights from file
 *   - All other buffers (activations, gradients, optimizer state) set to NULL (lazy allocation)
 */
void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(1); }
    if (model_header[1] != 3) {
        printf("Bad version in model file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_size = Vp = model_header[7];
    printf("[GPT-2]\n");
    printf("max_seq_len: %zu\n", maxT);
    printf("vocab_size: %zu\n", V);
    printf("padded_vocab_size: %zu\n", Vp);
    printf("num_layers: %zu\n", L);
    printf("num_heads: %zu\n", NH);
    printf("channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes,  model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // read in all the parameters from file
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_memory, sizeof(float), num_parameters, model_file);
    fcloseCheck(model_file);

    // other inits - use NULL to indicate not yet allocated (lazy allocation)
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
}

/**
 * GPT-2 Forward Pass
 *
 * Purpose: Executes the complete forward pass through the GPT-2 model, computing all
 * activations from input tokens to final output probabilities. If targets are provided,
 * also computes the loss.
 *
 * Algorithm:
 * 1. Encode inputs: token embeddings + position embeddings
 * 2. For each transformer layer (repeated L times):
 *    a. LayerNorm -> Multi-head attention -> Residual connection
 *    b. LayerNorm -> MLP (expand 4x, GELU, project back) -> Residual connection
 * 3. Final LayerNorm
 * 4. Project to vocabulary (reusing token embedding matrix wte)
 * 5. Softmax to get probabilities
 * 6. If targets provided: compute cross-entropy loss
 *
 * Architecture note: This is the "Pre-LN" transformer variant where LayerNorm comes
 * before each sub-layer, rather than after (as in the original Transformer paper).
 *
 * Parameters:
 *   model - The GPT2 model (will be modified with new activations)
 *   inputs - Input tokens (B, T): Indices into vocabulary [0, V)
 *   targets - Target tokens (B, T): Ground truth for loss computation (can be NULL)
 *   B - Batch size
 *   T - Sequence length (must be <= max_seq_len)
 *
 * Side effects:
 *   - Allocates activation memory on first call (lazy allocation)
 *   - Stores activations in model->acts (needed for backward pass)
 *   - If targets provided: stores loss in model->mean_loss
 *   - Caches inputs and targets in model for backward pass
 *
 * Pre-conditions:
 *   - model must be initialized via gpt2_build_from_checkpoint
 *   - B and T must match any previous forward pass (no dynamic batch size yet)
 */
void gpt2_forward(GPT2 *model, int* inputs, int* targets, size_t B, size_t T) {
    // targets are optional and could be NULL

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    // convenience parameters (size_t to help prevent int overflow)
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate space for all the activations if needed (done here, lazily)
    if(model->acts_memory == NULL) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        fill_in_activation_sizes(model->act_sizes, model->config, B, T);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        printf("num_activations: %zu\n", num_activations);
        model->num_activations = num_activations;
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        // also create memory for caching inputs and targets
        model->inputs = (int*)mallocCheck(B * T * sizeof(int));
        model->targets = (int*)mallocCheck(B * T * sizeof(int)); // might be unused if we never have targets but it's small
    } else {
        // validate B,T is consistent with how we've allocated the memory before
        // in principle we could get more clever here in the future, for now this is safest
        if (B != model->batch_size || T != model->seq_len) {
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", model->batch_size, model->seq_len, (int)B, (int)T);
            exit(EXIT_FAILURE);
        }
    }

    // cache the inputs/targets (needed for backward pass)
    memcpy(model->inputs, inputs, B * T * sizeof(int));
    if (targets != NULL) {
        memcpy(model->targets, targets, B * T * sizeof(int));
    }

    // forward pass through the network
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float* residual;

    // Initial encoding: token embeddings + position embeddings
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, B, T, C);

    // Process through L transformer layers
    for (int l = 0; l < L; l++) {
        // Residual connection input: encoded for first layer, previous layer's output for others
        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // Get pointers to this layer's parameters (offset into parameter arrays)
        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        // Get pointers to this layer's activations (offset into activation arrays)
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_preatt = acts.preatt + l * B * NH * T * T;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;

        // Transformer layer: Attention block
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);  // Project to Q, K, V
        attention_forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);  // Multi-head attention
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);  // Output projection
        residual_forward(l_residual2, residual, l_attproj, B*T*C);  // Add residual

        // Transformer layer: MLP block
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);  // Expand to 4*C
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);  // Non-linearity
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);  // Project back to C
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);  // Add residual
    }

    // Final processing: LayerNorm -> Project to vocabulary -> Softmax
    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.logits, acts.lnf, params.wte, NULL, B, T, C, Vp);  // Reuse wte for output projection
    softmax_forward(acts.probs, acts.logits, B, T, V, Vp);

    // Compute loss if targets are provided
    if (targets != NULL) {
        crossentropy_forward(model->acts.losses, model->acts.probs, targets, B, T, Vp);
        // Calculate mean loss across all positions
        float mean_loss = 0.0f;
        for (int i=0; i<B*T; i++) { mean_loss += model->acts.losses[i]; }
        mean_loss /= B*T;
        model->mean_loss = mean_loss;
    } else {
        // if we don't have targets, we don't have a loss
        model->mean_loss = -1.0f;
    }
}

/**
 * Zero All Gradients
 *
 * Purpose: Resets all gradients to zero before starting a backward pass. This is necessary
 * because gradients are accumulated (+=) in the backward pass, so we need a clean slate.
 *
 * Parameters:
 *   model - The GPT2 model
 *
 * Note: Should be called before gpt2_backward(). Some training frameworks might accumulate
 *       gradients over multiple batches (gradient accumulation), in which case you'd only
 *       zero_grad every N steps instead of every step.
 */
void gpt2_zero_grad(GPT2 *model) {
    if(model->grads_memory != NULL) { memset(model->grads_memory, 0, model->num_parameters * sizeof(float)); }
    if(model->grads_acts_memory != NULL) { memset(model->grads_acts_memory, 0, model->num_activations * sizeof(float)); }
}

/**
 * GPT-2 Backward Pass
 *
 * Purpose: Executes backpropagation through the entire GPT-2 model, computing gradients
 * for all parameters. This implements the reverse-mode automatic differentiation of the
 * forward pass.
 *
 * Algorithm:
 * Processes layers in reverse order of the forward pass, applying the chain rule:
 * 1. Initialize: Set dloss = 1/(B*T) (gradient of mean loss)
 * 2. Backward through: softmax+cross-entropy (fused)
 * 3. Backward through: final LayerNorm and vocabulary projection
 * 4. For each transformer layer in reverse (L-1 down to 0):
 *    a. MLP backward: residual -> fcproj -> GELU -> fc -> LayerNorm
 *    b. Attention backward: residual -> attproj -> attention -> qkv -> LayerNorm
 * 5. Backward through: encoder (position and token embeddings)
 *
 * Parameters:
 *   model - The GPT2 model (must have run forward pass with targets first)
 *
 * Side effects:
 *   - Allocates gradient memory on first call (lazy allocation)
 *   - Populates model->grads with gradients for all parameters
 *   - Populates model->grads_acts with gradients for all activations
 *
 * Pre-conditions:
 *   - Must have called gpt2_forward() with targets (mean_loss != -1.0f)
 *   - Should call gpt2_zero_grad() first to clear previous gradients
 *
 * Note: The backward pass mirrors the forward pass exactly, calling the _backward
 *       version of each function in reverse order.
 */
void gpt2_backward(GPT2 *model) {

    // double check we forwarded previously, with targets
    if (model->mean_loss == -1.0f) {
        printf("Error: must forward with targets before backward\n");
        exit(1);
    }

    // lazily allocate the memory for gradients of the weights and activations, if needed
    if (model->grads_memory == NULL) {
        model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_sizes);
        model->grads_acts_memory = malloc_and_point_activations(&model->grads_acts, model->act_sizes);
        gpt2_zero_grad(model);
    }

    // convenience shortcuts (and size_t to help prevent int overflow)
    size_t B = model->batch_size;
    size_t T = model->seq_len;
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // backward pass: go in the reverse order of the forward pass, and call backward() functions
    ParameterTensors params = model->params; // for brevity
    ParameterTensors grads = model->grads;
    ActivationTensors acts = model->acts;
    ActivationTensors grads_acts = model->grads_acts;

    // Kick off the chain rule by initializing the gradient of the loss
    // Since final loss = mean(losses), the gradient is dloss/dloss[i] = 1/(B*T) for all i
    float dloss_mean = 1.0f / (B*T);
    for (int i = 0; i < B*T; i++) { grads_acts.losses[i] = dloss_mean; }

    // Backward through output layers
    crossentropy_softmax_backward(grads_acts.logits, grads_acts.losses, acts.probs, model->targets, B, T, V, Vp);
    matmul_backward(grads_acts.lnf, grads.wte, NULL, grads_acts.logits, acts.lnf, params.wte, B, T, C, Vp);
    float* residual = acts.residual3 + (L-1) * B * T * C; // last layer's residual
    float* dresidual = grads_acts.residual3 + (L-1) * B * T * C; // write to last layer's residual
    layernorm_backward(dresidual, grads.lnfw, grads.lnfb, grads_acts.lnf, residual, params.lnfw, acts.lnf_mean, acts.lnf_rstd, B, T, C);

    // Backward through transformer layers in reverse order
    for (int l = L-1; l >= 0; l--) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;
        dresidual = l == 0 ? grads_acts.encoded : grads_acts.residual3 + (l-1) * B * T * C;

        // Get pointers to this layer's parameters
        float* l_ln1w = params.ln1w + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        // Get pointers to this layer's parameter gradients
        float* dl_ln1w = grads.ln1w + l * C;
        float* dl_ln1b = grads.ln1b + l * C;
        float* dl_qkvw = grads.qkvw + l * 3*C * C;
        float* dl_qkvb = grads.qkvb + l * 3*C;
        float* dl_attprojw = grads.attprojw + l * C * C;
        float* dl_attprojb = grads.attprojb + l * C;
        float* dl_ln2w = grads.ln2w + l * C;
        float* dl_ln2b = grads.ln2b + l * C;
        float* dl_fcw = grads.fcw + l * 4*C * C;
        float* dl_fcb = grads.fcb + l * 4*C;
        float* dl_fcprojw = grads.fcprojw + l * C * 4*C;
        float* dl_fcprojb = grads.fcprojb + l * C;
        // Get pointers to this layer's activations
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkv = acts.qkv + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        // Get pointers to this layer's activation gradients
        float* dl_ln1 = grads_acts.ln1 + l * B * T * C;
        float* dl_qkv = grads_acts.qkv + l * B * T * 3*C;
        float* dl_atty = grads_acts.atty + l * B * T * C;
        float* dl_preatt = grads_acts.preatt + l * B * NH * T * T;
        float* dl_att = grads_acts.att + l * B * NH * T * T;
        float* dl_attproj = grads_acts.attproj + l * B * T * C;
        float* dl_residual2 = grads_acts.residual2 + l * B * T * C;
        float* dl_ln2 = grads_acts.ln2 + l * B * T * C;
        float* dl_fch = grads_acts.fch + l * B * T * 4*C;
        float* dl_fch_gelu = grads_acts.fch_gelu + l * B * T * 4*C;
        float* dl_fcproj = grads_acts.fcproj + l * B * T * C;
        float* dl_residual3 = grads_acts.residual3 + l * B * T * C;

        // Backpropagate through this layer (reverse of forward pass)
        // MLP block backward
        residual_backward(dl_residual2, dl_fcproj, dl_residual3, B*T*C);
        matmul_backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
        gelu_backward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C);
        matmul_backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C);
        layernorm_backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        // Attention block backward
        residual_backward(dresidual, dl_attproj, dl_residual2, B*T*C);
        matmul_backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
        attention_backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
        matmul_backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
        layernorm_backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
    }

    // Backward through encoder (embeddings)
    encoder_backward(grads.wte, grads.wpe, grads_acts.encoded, model->inputs, B, T, C);
}

/**
 * GPT-2 Parameter Update (AdamW Optimizer)
 *
 * Purpose: Updates model parameters using the AdamW optimization algorithm. AdamW combines
 * the benefits of Adam (adaptive learning rates) with proper weight decay regularization.
 *
 * Algorithm (AdamW):
 * For each parameter θ:
 *   1. Update first moment (exponential moving average of gradients):
 *      m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
 *   2. Update second moment (exponential moving average of squared gradients):
 *      v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
 *   3. Compute bias-corrected moments (corrects initialization bias):
 *      m_hat = m_t / (1 - β₁^t)
 *      v_hat = v_t / (1 - β₂^t)
 *   4. Update parameters with weight decay:
 *      θ_t = θ_{t-1} - lr * (m_hat / (√v_hat + ε) + λ * θ_{t-1})
 *
 * Parameters:
 *   model - The GPT2 model to update
 *   learning_rate - Step size (lr), typically 1e-4 to 1e-3
 *   beta1 - First moment decay rate (β₁), typically 0.9
 *   beta2 - Second moment decay rate (β₂), typically 0.999
 *   eps - Small constant for numerical stability (ε), typically 1e-8
 *   weight_decay - L2 regularization coefficient (λ), typically 0.01 or 0.0
 *   t - Current time step (iteration number), used for bias correction
 *
 * Side effects:
 *   - Allocates m_memory and v_memory on first call (lazy allocation)
 *   - Updates all parameters in model->params_memory
 *   - Updates optimizer state (m_memory, v_memory)
 *
 * Rationale:
 *   - AdamW separates weight decay from gradient-based update (better than L2 in Adam)
 *   - Adaptive learning rates help with varying gradient magnitudes across parameters
 *   - Bias correction ensures proper behavior in early training steps
 *
 * Reference: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, 2019)
 *            https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
 */
void gpt2_update(GPT2 *model, float learning_rate, float beta1, float beta2, float eps, float weight_decay, int t) {
    // reference: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

    // lazily allocate the memory for m_memory and v_memory
    if (model->m_memory == NULL) {
        model->m_memory = (float*)calloc(model->num_parameters, sizeof(float));
        model->v_memory = (float*)calloc(model->num_parameters, sizeof(float));
    }

    for (size_t i = 0; i < model->num_parameters; i++) {
        float param = model->params_memory[i];
        float grad = model->grads_memory[i];

        // update the first moment (momentum)
        float m = beta1 * model->m_memory[i] + (1.0f - beta1) * grad;
        // update the second moment (RMSprop)
        float v = beta2 * model->v_memory[i] + (1.0f - beta2) * grad * grad;
        // bias-correct both moments
        float m_hat = m / (1.0f - powf(beta1, t));
        float v_hat = v / (1.0f - powf(beta2, t));

        // update parameter: adaptive step + weight decay
        model->m_memory[i] = m;
        model->v_memory[i] = v;
        model->params_memory[i] -= learning_rate * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * param);
    }
}

/**
 * Free GPT-2 Model Memory
 *
 * Purpose: Deallocates all memory associated with the GPT-2 model. Should be called
 * when done with the model to avoid memory leaks.
 *
 * Parameters:
 *   model - The GPT2 model to free
 *
 * Note: Frees all allocated blocks (parameters, gradients, activations, optimizer state,
 *       input/target buffers). Checks for NULL are done by free() internally.
 */
void gpt2_free(GPT2 *model) {
    free(model->params_memory);
    free(model->grads_memory);
    free(model->m_memory);
    free(model->v_memory);
    free(model->acts_memory);
    free(model->grads_acts_memory);
    free(model->inputs);
    free(model->targets);
}

#ifndef TESTING
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// Random number generation and sampling utilities

/**
 * XorShift Random Number Generator (32-bit unsigned integer)
 *
 * Purpose: Fast, lightweight pseudo-random number generator. Used for sampling
 * tokens during text generation.
 *
 * Algorithm: XorShift* variant with period 2^64 - 1
 *
 * Parameters:
 *   state - Pointer to RNG state (modified in-place)
 *
 * Returns:
 *   32-bit unsigned random integer
 *
 * Reference: https://en.wikipedia.org/wiki/Xorshift#xorshift*
 *
 * Note: Not cryptographically secure, but sufficient for sampling in ML.
 */
unsigned int random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

/**
 * Random Float Generator
 *
 * Purpose: Converts random 32-bit integer to float in [0, 1) range.
 *
 * Parameters:
 *   state - Pointer to RNG state
 *
 * Returns:
 *   Random float in range [0, 1)
 */
float random_f32(uint64_t *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

/**
 * Sample from Categorical Distribution
 *
 * Purpose: Samples an index from a discrete probability distribution. Used for
 * selecting the next token during text generation.
 *
 * Algorithm:
 * Uses inverse transform sampling:
 *   1. Compute cumulative distribution function (CDF)
 *   2. Find first index where CDF > random coin flip
 *
 * Parameters:
 *   probabilities - Array of probabilities (must sum to 1.0)
 *   n - Number of elements in probabilities array
 *   coin - Random number in [0, 1) (from random_f32)
 *
 * Returns:
 *   Sampled index in range [0, n-1]
 *
 * Example:
 *   probs = [0.1, 0.3, 0.6]
 *   coin = 0.5 -> returns 2 (third element)
 *   coin = 0.05 -> returns 0 (first element)
 */
int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

// ----------------------------------------------------------------------------
/**
 * Main Training Loop
 *
 * Purpose: Demonstrates a complete GPT-2 training pipeline on CPU. This is an educational
 * example showing all the pieces working together.
 *
 * Training procedure:
 * 1. Load pre-trained GPT-2 model from checkpoint
 * 2. Initialize data loaders for training and validation data
 * 3. Run training loop:
 *    - Every 10 steps: Evaluate validation loss
 *    - Every 20 steps: Generate sample text to check quality
 *    - Every step: Forward pass -> Backward pass -> Parameter update
 * 4. Clean up and exit
 *
 * Training configuration:
 * - Batch size (B): 4 sequences
 * - Sequence length (T): 64 tokens
 * - Training steps: 40
 * - Learning rate: 1e-4
 * - Optimizer: AdamW with β₁=0.9, β₂=0.999, weight_decay=0.0
 *
 * Note: This is a minimal example. Real training would:
 * - Train for many more steps (100k+)
 * - Use larger batches and sequences
 * - Include learning rate scheduling
 * - Save checkpoints periodically
 * - Use more sophisticated evaluation metrics
 */
int main() {

    // Load the GPT-2 model from a checkpoint file
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

    // Initialize data loaders
    // Try tiny_shakespeare first, fall back to tiny_stories if not available
    const char* tiny_stories_train = "dev/data/tinystories/TinyStories_train.bin";
    const char* tiny_stories_val = "dev/data/tinystories/TinyStories_val.bin";
    const char* tiny_shakespeare_train = "dev/data/tinyshakespeare/tiny_shakespeare_train.bin";
    const char* tiny_shakespeare_val = "dev/data/tinyshakespeare/tiny_shakespeare_val.bin";
    const char* train_tokens = access(tiny_shakespeare_train, F_OK) != -1 ? tiny_shakespeare_train : tiny_stories_train;
    const char* val_tokens = access(tiny_shakespeare_val, F_OK) != -1 ? tiny_shakespeare_val : tiny_stories_val;

    // Training hyperparameters
    int B = 4;   // Batch size: 4 independent sequences processed in parallel
    int T = 64;  // Sequence length: 64 tokens per sequence (must be <= max_seq_len)

    DataLoader train_loader, val_loader;
    dataloader_init(&train_loader, train_tokens, B, T, 0, 1, 1);
    dataloader_init(&val_loader, val_tokens, B, T, 0, 1, 0);
    printf("train dataset num_batches: %zu\n", train_loader.num_tokens / (B*T));
    printf("val dataset num_batches: %zu\n", val_loader.num_tokens / (B*T));
    int val_num_batches = 5;  // Number of batches to use for validation evaluation

    // Initialize tokenizer for text generation
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    // Allocate memory for text generation
    uint64_t rng_state = 1337;  // RNG seed for reproducible generation
    int* gen_tokens = (int*)mallocCheck(B * T * sizeof(int));
    const int genT = 64;  // Number of tokens to generate

    // Main training loop
    struct timespec start, end;
    for (int step = 0; step <= 40; step++) {

        // Validation: Estimate loss on held-out data
        if (step % 10 == 0) {
            float val_loss = 0.0f;
            dataloader_reset(&val_loader);
            for (int i = 0; i < val_num_batches; i++) {
                dataloader_next_batch(&val_loader);
                gpt2_forward(&model, val_loader.inputs, val_loader.targets, B, T);
                val_loss += model.mean_loss;
            }
            val_loss /= val_num_batches;
            printf("val loss %f\n", val_loss);
        }

        // Text generation: Sample from the model to check quality
        if (step > 0 && step % 20 == 0) {
            // Initialize with end-of-text token to start generation
            for(int i = 0; i < B * T; ++i) {
                gen_tokens[i] = tokenizer.eot_token;
            }
            // Autoregressive generation: one token at a time
            printf("generating:\n---\n");
            for (int t = 1; t < genT; t++) {
                // Forward pass to get probabilities for next token
                // Note: This is inefficient (recomputes entire sequence each time)
                // but simple and correct. Production systems use KV caching.
                gpt2_forward(&model, gen_tokens, NULL, B, T);

                // Sample next token from probability distribution
                // Using only first sequence (b=0) from the batch
                float* probs = model.acts.probs + (t-1) * model.config.padded_vocab_size;
                float coin = random_f32(&rng_state);
                int next_token = sample_mult(probs, model.config.vocab_size, coin);
                gen_tokens[t] = next_token;

                // Decode and print the generated token
                if (tokenizer.init_ok) {
                    const char* token_str = tokenizer_decode(&tokenizer, next_token);
                    safe_printf(token_str);
                } else {
                    // Fallback: print token ID if tokenizer failed to load
                    printf("%d ", next_token);
                }
                fflush(stdout);
            }
            printf("\n---\n");
        }

        // Training step: Forward -> Zero gradients -> Backward -> Update
        clock_gettime(CLOCK_MONOTONIC, &start);
        dataloader_next_batch(&train_loader);
        gpt2_forward(&model, train_loader.inputs, train_loader.targets, B, T);
        gpt2_zero_grad(&model);
        gpt2_backward(&model);
        gpt2_update(&model, 1e-4f, 0.9f, 0.999f, 1e-8f, 0.0f, step+1);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        printf("step %d: train loss %f (took %f ms)\n", step, model.mean_loss, time_elapsed_s * 1000);
    }

    // Clean up: Free all allocated memory
    dataloader_free(&train_loader);
    dataloader_free(&val_loader);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    free(gen_tokens);
    return 0;
}
#endif
