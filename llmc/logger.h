/*
================================================================================
File: logger.h
Purpose: Simple, lightweight logger for training metrics and progress tracking
================================================================================

Overview:
---------
This file implements a minimal, file-based logging system for recording training
metrics during model training. The logger is designed to be simple, stateless,
and safe for use in distributed training environments.

Design Decisions:
-----------------
1. File-Based: Logs are written to a single text file (main.log) in the output
   directory, making it easy to track training progress over time.

2. Stateless: The logger uses append mode ("a") for all writes, eliminating the
   need to maintain file handle state between logging calls. Each log operation
   opens the file, writes, and closes it.

3. Rank-Aware: Only rank 0 (the master process) performs logging in distributed
   training setups, avoiding race conditions and duplicate log entries.

4. Human-Readable Format: Logs use a simple "key:value" text format that's easy
   to parse and read.

Log Entry Types:
---------------
- Training step: Step number, training loss, learning rate, gradient norm
- Validation step: Step number, validation loss
- Evaluation: Step number, evaluation score

Usage Pattern:
--------------
    Logger logger;
    logger_init(&logger, "output_logs", process_rank, resume_from_checkpoint);
    logger_log_train(&logger, step, loss, lr, grad_norm);
    logger_log_val(&logger, step, val_loss);

Benefits:
---------
- No complex dependencies, just standard C file I/O
- Safe for distributed training (only rank 0 writes)
- Append-only design prevents data loss if training crashes
- Human-readable format for easy debugging
*/
#ifndef LOGGER_H
#define LOGGER_H

#include <assert.h>
#include <stdio.h>
#include <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "utils.h"

/**
 * Logger - Stateless logger for training metrics
 *
 * @field active: 1 if logging is enabled (only true for rank 0), 0 otherwise
 * @field output_log_file: Full path to the log file (main.log)
 */
typedef struct {
    int active;
    char output_log_file[512];
} Logger;

/**
 * logger_init - Initializes the logger
 *
 * @param logger: Pointer to Logger struct to initialize
 * @param log_dir: Directory where log files should be written (can be NULL)
 * @param process_rank: Rank of the current process in distributed training (0 for single-GPU)
 * @param resume: 1 if resuming from checkpoint (append to existing log), 0 for fresh start
 *
 * This function sets up the logger. Key behaviors:
 * - Only activates logging for process_rank 0 (master process)
 * - Creates the log file path as "<log_dir>/main.log"
 * - If resume is 0, wipes any existing log file clean (fresh start)
 * - If resume is 1, existing log file is preserved for appending
 * - If log_dir is NULL, logging is disabled
 *
 * The 500-character limit on log_dir is a safety check to prevent buffer overflow.
 */
void logger_init(Logger *logger, const char *log_dir, int process_rank, int resume) {
    // currently, only rank 0 writes logs
    logger->active = 0;
    if (log_dir != NULL && process_rank == 0) {
        logger->active = 1;
        assert(strlen(log_dir) < 500); // being a bit lazy, could relax later
        snprintf(logger->output_log_file, 512, "%s/main.log", log_dir);
        if (resume == 0) {
            // wipe any existing logfile clean if we're starting fresh
            FILE *logfile = fopenCheck(logger->output_log_file, "w");
            fclose(logfile);
        }
    }
}

/**
 * logger_log_eval - Logs an evaluation metric
 *
 * @param logger: Pointer to Logger struct
 * @param step: Current training step number
 * @param val: Evaluation score/metric value
 *
 * Records an evaluation metric to the log file. Evaluation metrics are typically
 * computed on held-out test sets (e.g., HellaSwag accuracy, MMLU score).
 *
 * Log format: "s:<step> eval:<value>"
 * Example: "s:1000 eval:0.7234"
 *
 * This function does nothing if the logger is inactive (non-rank-0 processes).
 */
void logger_log_eval(Logger *logger, int step, float val) {
    if (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d eval:%.4f\n", step, val);
        fclose(logfile);
    }
}

/**
 * logger_log_val - Logs validation loss
 *
 * @param logger: Pointer to Logger struct
 * @param step: Current training step number
 * @param val_loss: Validation loss value
 *
 * Records the validation loss to the log file. Validation loss is computed
 * periodically on a validation dataset (separate from training data) to
 * monitor model performance and detect overfitting.
 *
 * Log format: "s:<step> tel:<loss>"
 * Example: "s:500 tel:2.3456"
 *
 * Note: "tel" likely stands for "test/evaluation loss"
 *
 * This function does nothing if the logger is inactive (non-rank-0 processes).
 */
void logger_log_val(Logger *logger, int step, float val_loss) {
    if (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d tel:%.4f\n", step, val_loss);
        fclose(logfile);
    }
}

/**
 * logger_log_train - Logs training step metrics
 *
 * @param logger: Pointer to Logger struct
 * @param step: Current training step number
 * @param train_loss: Training loss for this step
 * @param learning_rate: Current learning rate (may vary due to scheduling)
 * @param grad_norm: L2 norm of the gradients (useful for detecting training instabilities)
 *
 * Records comprehensive training metrics for each step. This is the most detailed
 * logging function and provides the core metrics needed to monitor training health.
 *
 * Log format: "s:<step> trl:<loss> lr:<learning_rate> norm:<grad_norm>"
 * Example: "s:100 trl:3.4567 lr:0.000300 norm:12.34"
 *
 * Metrics explained:
 * - trl (training loss): The loss value on the current training batch
 * - lr (learning rate): Current learning rate (may change with schedulers)
 * - norm (gradient norm): L2 norm of gradients, useful for detecting:
 *   - Gradient explosion (very large norms)
 *   - Vanishing gradients (very small norms)
 *   - Training stability issues
 *
 * This function does nothing if the logger is inactive (non-rank-0 processes).
 */
void logger_log_train(Logger *logger, int step, float train_loss, float learning_rate, float grad_norm) {
    if (logger->active == 1) {
        FILE *logfile = fopenCheck(logger->output_log_file, "a");
        fprintf(logfile, "s:%d trl:%.4f lr:%.6f norm:%.2f\n", step, train_loss, learning_rate, grad_norm);
        fclose(logfile);
    }
}

#endif