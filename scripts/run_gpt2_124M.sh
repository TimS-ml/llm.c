#!/bin/bash
# ==============================================================================
# GPT-2 124M Training Script on FineWeb
# ==============================================================================
# This script trains a GPT-2 124M parameter model on the FineWeb dataset,
# reproducing GPT-2 Small with modern datasets and techniques.
#
# Model specifications:
#   - Parameters: 124M (d12 architecture: 12 layers, 768 hidden dim)
#   - Training tokens: ~10B from FineWeb
#   - Compute budget: 6 × 124M × 10B = 7.44e18 FLOPs (~7e18)
#   - Training steps: 18,865 iterations
#   - Tokens per step: 524,288
#
# Hardware requirements:
#   - 8× NVIDIA A100 80GB GPUs (or equivalent)
#   - Training time: ~94 minutes (~300ms/iteration)
#   - Estimated cost: ~$20 on cloud infrastructure ($14/hr for 8×A100)
#
# Prerequisites:
#   1. Build trainer: make train_gpt2cu USE_CUDNN=1
#   2. Download FineWeb data: python dev/data/fineweb.py --version 10B
#   3. (Optional) Prepare eval data: python dev/data/hellaswag.py
#   4. Ensure MPI is installed for multi-GPU training
#
# Features:
#   - Automatic resume on crash/interruption
#   - Multi-GPU training via MPI + NCCL
#   - BFloat16 mixed precision training
#   - Gradient checkpointing for memory efficiency
#
# Usage:
#   bash scripts/run_gpt2_124M.sh
#
# The script will run until completion (DONE file created) and automatically
# resume if interrupted.
# ==============================================================================

# Build the CUDA training executable with cuDNN flash attention
make train_gpt2cu USE_CUDNN=1

# Output directory for checkpoints and logs
out_dir="log_gpt2_124M"

# Training completion marker (created when all 18,865 iterations finish)
done_file="$out_dir/DONE_00018865"

# ==============================================================================
# Training Loop with Automatic Resume
# ==============================================================================
# This loop allows training to resume automatically if it stops due to:
# - Hardware failures or preemption
# - Out of memory errors
# - Network issues
# - Manual interruption
#
# The -y 1 flag enables checkpoint resumption
while true; do

    # Check if training is complete
    if [ -f "$done_file" ]; then
        echo "Training complete! File $done_file exists. Exiting."
        break
    fi

    # Launch distributed training across 8 GPUs
    # mpirun -np 8: Run 8 processes (one per GPU)
    mpirun -np 8 ./train_gpt2cu \
                -i "dev/data/fineweb10B/fineweb_train_*.bin" \
                -j "dev/data/fineweb10B/fineweb_val_*.bin" \
                -o $out_dir \
                -v 250 -s 20000 -g 144 \
                -h 1 \
                -b 64 -t 1024 \
                -d 524288 \
                -r 0 \
                -z 1 \
                -c 0.1 \
                -l 0.0006 \
                -q 0.0 \
                -u 700 \
                -n 5000 \
                -y 1 \
                -e "d12"

    # Brief pause before checking completion and potentially restarting
    sleep 1
done

# ==============================================================================
# Training Parameter Explanation
# ==============================================================================
# -i: Training data pattern (tokenized binary shards)
# -j: Validation data pattern
# -o: Output directory for checkpoints and logs
# -v 250: Validate every 250 steps
# -s 20000: Save checkpoint every 20,000 steps
# -g 144: Generate text samples every 144 steps
# -h 1: Enable mixed precision (BFloat16)
# -b 64: Batch size per GPU (64 sequences)
# -t 1024: Sequence length (context window)
# -d 524288: Total batch size across all GPUs (524,288 tokens)
# -r 0: Random seed (0 = default)
# -z 1: Enable ZeRO optimization
# -c 0.1: Weight decay (L2 regularization)
# -l 0.0006: Learning rate (6e-4)
# -q 0.0: Gradient clipping (0.0 = disabled)
# -u 700: Warmup iterations (linear LR warmup)
# -n 5000: Keep checkpoints every 5000 steps
# -y 1: Resume from latest checkpoint if available
# -e "d12": Model architecture (12 layers, 768 dim)
