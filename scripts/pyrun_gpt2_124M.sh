#!/bin/bash
# ==============================================================================
# PyTorch GPT-2 124M Training Script
# ==============================================================================
# This script trains a GPT-2 124M parameter model using PyTorch on the FineWeb
# dataset. It's equivalent to scripts/run_gpt2_124M.sh but uses the PyTorch
# implementation instead of the CUDA C implementation.
#
# Model specs:
#   - 124M parameters (d12 architecture: 12 layers, 768 dim)
#   - Trained on ~10B tokens from FineWeb
#   - Total training: 18,865 iterations at 524,288 tokens/batch
#   - Target compute: ~7e18 FLOPs (Chinchilla scaling)
#
# Hardware requirements:
#   - 8 GPUs (A100 80GB recommended)
#   - Training time: ~94 minutes on 8x A100
#   - Estimated cost: ~$20 on cloud infrastructure
#
# Prerequisites:
#   1. Download data: python dev/data/fineweb.py --version 10B
#   2. Install PyTorch with CUDA support
#   3. Ensure GPUs are available and accessible
#
# Usage:
#   # Multi-GPU (8 GPUs):
#   bash scripts/pyrun_gpt2_124M.sh
#
#   # Single GPU (modify script or run directly):
#   python train_gpt2.py --input_bin "dev/data/fineweb10B/fineweb_train_*.bin" \
#       --input_val_bin "dev/data/fineweb10B/fineweb_val_*.bin" [other args...]
# ==============================================================================

# Launch distributed training across 8 GPUs using PyTorch's torchrun
# --standalone: Single-node multi-GPU training
# --nproc_per_node=8: Use 8 GPUs on this node
torchrun --standalone --nproc_per_node=8 train_gpt2.py \
    --input_bin "dev/data/fineweb10B/fineweb_train_*.bin" \
    --input_val_bin "dev/data/fineweb10B/fineweb_val_*.bin" \
    --val_loss_every 250 \
    --sample_every 0 \
    --output_dir pylog_gpt2_124M \
    --write_tensors 0 \
    --model d12 \
    --batch_size 32 \
    --sequence_length 1024 \
    --total_batch_size 524288 \
    --dtype bfloat16 \
    --compile 1 \
    --tensorcores 1 \
    --flash 1 \
    --num_iterations 18865 \
    --weight_decay 0.1 \
    --zero_stage 1 \
    --learning_rate 0.0006 \
    --warmup_iters 700 \
    --learning_rate_decay_frac 0.0 \
    --overfit_single_batch 0

# ==============================================================================
# Parameter Explanation
# ==============================================================================
# --input_bin: Training data shards (tokenized binary format)
# --input_val_bin: Validation data shards
# --val_loss_every: Compute validation loss every N iterations
# --sample_every: Generate text samples every N iterations (0 = disabled)
# --output_dir: Directory for checkpoints and logs
# --write_tensors: Write tensor dumps for debugging (0 = disabled)
# --model d12: GPT-2 124M architecture (12 layers, 768 hidden dim)
# --batch_size: Per-GPU batch size (32 sequences per GPU)
# --sequence_length: Context length (1024 tokens)
# --total_batch_size: Global batch size across all GPUs (524,288 tokens)
# --dtype: Use bfloat16 precision for training
# --compile: Use PyTorch 2.0 compilation for faster execution
# --tensorcores: Enable tensor core operations for matrix multiplies
# --flash: Use Flash Attention for efficient attention computation
# --num_iterations: Total training iterations (18,865)
# --weight_decay: L2 regularization strength (0.1)
# --zero_stage: ZeRO optimization stage 1 (optimizer state sharding)
# --learning_rate: Peak learning rate (6e-4)
# --warmup_iters: Linear warmup iterations (700)
# --learning_rate_decay_frac: LR decay fraction (0.0 = constant after warmup)
# --overfit_single_batch: Debug mode to overfit single batch (0 = disabled)
