/*
================================================================================
File: schedulers.h
Purpose: Learning rate scheduling strategies for training optimization
================================================================================

Overview:
---------
Learning rate scheduling is critical for successful LLM training. This file
implements several common scheduling strategies that adjust the learning rate
over the course of training.

Why Learning Rate Scheduling?
-----------------------------
- Too high LR: Training is unstable, loss diverges
- Too low LR: Training is slow, may get stuck in local minima
- Optimal LR changes during training:
  - Early: High LR for fast progress
  - Middle: Moderate LR for refinement
  - Late: Low LR for fine-tuning

Implemented Schedules:
---------------------
1. Cosine: Smooth cosine decay (most common for LLMs)
2. Linear: Linear decay (simpler, sometimes works as well)
3. Constant: No decay (for ablations/debugging)
4. WSD: Warmup-Stable-Decay (recent research, good for long runs)

Warmup:
-------
All schedules support warmup: linearly increase LR from 0 to max over the
first N steps. This prevents early training instability from large gradients.

Usage:
------
    LearningRateScheduler scheduler;
    lr_scheduler_init(&scheduler, "cosine", 3e-4, warmup_steps, total_steps, 0.1);
    for (step = 0; step < total_steps; step++) {
        float lr = get_learning_rate(&scheduler, step);
        // Use lr for optimizer
    }
*/
#ifndef SCHEDULERS_H
#define SCHEDULERS_H

#include <assert.h>
#include <math.h>
#include <string.h>

/**
 * LearningRateScheduler - Configuration for learning rate scheduling
 *
 * @field type: Schedule type ("cosine", "linear", "constant", "wsd")
 * @field learning_rate: Maximum learning rate (reached after warmup)
 * @field warmup_iterations: Number of steps for linear warmup
 * @field train_num_batches: Total number of training steps
 * @field final_learning_rate_frac: Final LR as fraction of max LR (e.g., 0.1 = 10%)
 */
typedef struct {
    const char* type;
    float learning_rate;
    int warmup_iterations;
    int train_num_batches;
    float final_learning_rate_frac;
} LearningRateScheduler;

void lr_scheduler_init(LearningRateScheduler *scheduler, const char* scheduler_type, float learning_rate, int warmup_iterations, int train_num_batches, float final_learning_rate_frac) {
    scheduler->type = scheduler_type;
    scheduler->learning_rate = learning_rate;
    scheduler->warmup_iterations = warmup_iterations;
    scheduler->train_num_batches = train_num_batches;
    scheduler->final_learning_rate_frac = final_learning_rate_frac;
}

/**
 * get_learning_rate_cosine - Cosine annealing schedule with warmup
 *
 * @param scheduler: Scheduler configuration
 * @param step: Current training step
 * @return: Learning rate for this step
 *
 * Schedule shape:
 * 1. Warmup (steps 0 to warmup_iterations):
 *    LR increases linearly from 0 to max_lr
 *
 * 2. Cosine decay (steps warmup_iterations to train_num_batches):
 *    LR follows cosine curve from max_lr down to (max_lr * final_frac)
 *    Formula: lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π * progress))
 *
 * Benefits of cosine:
 * - Smooth, gradual decay (no sudden drops)
 * - Fast decay initially, slower later (good for fine-tuning)
 * - Widely used and well-tested for LLMs
 *
 * Common settings:
 * - warmup_iterations: 2000-5000 steps
 * - final_learning_rate_frac: 0.1 (decay to 10% of max LR)
 */
float get_learning_rate_cosine(LearningRateScheduler *scheduler, int step) {
    float lr = scheduler->learning_rate;
    if (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((float)(step + 1)) / scheduler->warmup_iterations;
    } else {
        float decay_ratio = ((float)(step - scheduler->warmup_iterations)) / (scheduler->train_num_batches - scheduler->warmup_iterations);
        assert(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        float coeff = 0.5f * (1.0f + cosf(M_PI * decay_ratio)); // coeff starts at 1 and goes to 0
        assert(0.0f <= coeff && coeff <= 1.0f);
        float min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = min_lr + coeff * (scheduler->learning_rate - min_lr);
    }
    return lr;
}

// linear: warmup linearly to max LR, then decay linearly to LR * final_learning_rate_frac
float get_learning_rate_linear(LearningRateScheduler *scheduler, int step) {
    float lr = scheduler->learning_rate;
    if (step < scheduler->warmup_iterations) {
        lr = scheduler->learning_rate * ((float)(step + 1)) / scheduler->warmup_iterations;
    } else {
        float decay_ratio = ((float)(step - scheduler->warmup_iterations)) / (scheduler->train_num_batches - scheduler->warmup_iterations);
        assert(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        float min_lr = scheduler->learning_rate * scheduler->final_learning_rate_frac;
        lr = scheduler->learning_rate - decay_ratio * (scheduler->learning_rate - min_lr);
    }
    return lr;
}

// constant
float get_learning_rate_constant(LearningRateScheduler *scheduler, int step) {
    return scheduler->learning_rate;
}

// wsd schedule: warmup linearly, keep constant, last 20% decay using 1 - sqrt decay to final_frac (should be 0.0)
// https://arxiv.org/abs/2405.18392
float get_learning_rate_wsd(LearningRateScheduler *scheduler, int step) {
    int decay_point = (int)(0.8f * scheduler->train_num_batches);
    float max_lr = scheduler->learning_rate;
    float lr = max_lr;
    if (step < scheduler->warmup_iterations) {
        float decay_ratio = ((float)(step + 1)) / scheduler->warmup_iterations;
        lr = max_lr * decay_ratio;
    } else if (step < decay_point) {
        // noop, keep lr constant
    } else {
        float decay_ratio = ((float)(step - decay_point)) / (scheduler->train_num_batches - decay_point);
        assert(0.0f <= decay_ratio && decay_ratio <= 1.0f);
        float min_lr = max_lr * scheduler->final_learning_rate_frac;
        return min_lr + (1.0f - sqrtf(decay_ratio)) * (max_lr - min_lr);
    }
    return lr;
}

// return the learning rate at a given step
float get_learning_rate(LearningRateScheduler *scheduler, int step) {
    float step_learning_rate;
    if (strcmp(scheduler->type, "cosine") == 0) {
        step_learning_rate = get_learning_rate_cosine(scheduler, step);
    } else if (strcmp(scheduler->type, "linear") == 0) {
        step_learning_rate = get_learning_rate_linear(scheduler, step);
    } else if (strcmp(scheduler->type, "constant") == 0) {
        step_learning_rate = get_learning_rate_constant(scheduler, step);
    } else if (strcmp(scheduler->type, "wsd") == 0) {
        step_learning_rate = get_learning_rate_wsd(scheduler, step);
    } else {
        fprintf(stderr, "Unknown learning rate scheduler type: %s\n", scheduler->type);
        exit(EXIT_FAILURE);
    }
    return step_learning_rate;
}

#endif // SCHEDULERS_H
