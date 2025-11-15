#!/bin/bash
# ==============================================================================
# GPT-2 1558M Training Script on FineWeb-EDU
# ==============================================================================
# This script trains a GPT-2 1558M parameter model (GPT-2 XL) on FineWeb-EDU,
# a curated high-quality educational subset of FineWeb.
#
# Model specifications:
#   - Parameters: 1558M (d48 architecture: 48 layers, 1600 hidden dim)
#   - Training tokens: ~32B from FineWeb-EDU
#   - Compute budget: 6 × 1558M × 32B = 2.99e20 FLOPs (~3e20)
#   - Training steps: 32,000 iterations
#   - Tokens per step: 1,048,576 (larger batch than other models)
#
# Hardware requirements:
#   - 8× NVIDIA H100 80GB GPUs (H100 recommended for best performance)
#   - Training time: ~24 hours (~2.7-2.8s/iteration)
#   - Estimated cost: ~$672 on cloud infrastructure ($28/hr for 8×H100)
#
# Prerequisites:
#   1. Build trainer: make train_gpt2cu USE_CUDNN=1
#   2. Download FineWeb-EDU data: python dev/data/edu_fineweb.py --version 100B
#   3. Ensure MPI is installed for multi-GPU training
#
# Note: Uses FineWeb-EDU for higher quality training data and cosine LR schedule
#
# Usage:
#   bash scripts/run_gpt2_1558M.sh
# ==============================================================================

# Build the CUDA training executable with cuDNN flash attention
make train_gpt2cu USE_CUDNN=1

# Output directory for checkpoints and logs
out_dir="log_gpt2_1558M"

# Training completion marker
done_file="$out_dir/DONE_00032000"

# ==============================================================================
# Training Loop with Automatic Resume
# ==============================================================================
while true; do

    # Check if training is complete
    if [ -f "$done_file" ]; then
        echo "Training complete! File $done_file exists. Exiting."
        break
    fi

    # Launch distributed training across 8 GPUs
    mpirun -np 8 ./train_gpt2cu \
                -i "dev/data/edu_fineweb100B/edu_fineweb_train_*.bin" \
                -j "dev/data/edu_fineweb100B/edu_fineweb_val_*.bin" \
                -o $out_dir \
                -v 250 -s 300000 -g 384 \
                -h 1 \
                -b 16 -t 1024 \
                -d 1048576 \
                -r 0 \
                -z 1 \
                -c 0.1 \
                -k "cosine" \
                -l 0.0006 \
                -q 0.1 \
                -u 700 \
                -n 2000 \
                -x 32000 \
                -ge 1 \
                -y 1 \
                -e "d48"

    sleep 1
done

# ==============================================================================
# Training Parameter Explanation
# ==============================================================================
# -i: Training data (FineWeb-EDU 100B tokenized binary shards)
# -j: Validation data
# -o: Output directory for checkpoints and logs
# -v 250: Validate every 250 steps
# -s 300000: Save checkpoint every 300,000 steps
# -g 384: Generate text samples every 384 steps
# -h 1: Enable mixed precision (BFloat16)
# -b 16: Batch size per GPU (reduced to 16 due to very large model)
# -t 1024: Sequence length (context window)
# -d 1048576: Total batch size across all GPUs (1M tokens, 2X other models)
# -r 0: Random seed (0 = default)
# -z 1: Enable ZeRO optimization
# -c 0.1: Weight decay (L2 regularization)
# -k "cosine": Learning rate schedule (cosine decay instead of linear)
# -l 0.0006: Learning rate (6e-4)
# -q 0.1: Gradient clipping (0.1, enabled for stability)
# -u 700: Warmup iterations
# -n 2000: Keep checkpoints every 2000 steps
# -x 32000: Maximum iterations (training stops at this point)
# -ge 1: Gradient accumulation enabled
# -y 1: Resume from latest checkpoint if available
# -e "d48": Model architecture (48 layers, 1600 dim, GPT-2 1558M)
