#!/bin/bash
# ==============================================================================
# GPT-2 774M Training Script on FineWeb
# ==============================================================================
# This script trains a GPT-2 774M parameter model (GPT-2 Large) on FineWeb.
# This is approximately 100X the compute of the 124M model and 10X the 350M.
#
# Model specifications:
#   - Parameters: 774M (d36 architecture: 36 layers, 1280 hidden dim)
#   - Training tokens: ~150B from FineWeb (100B dataset)
#   - Compute budget: 6 × 774M × 150B = 6.966e20 FLOPs (~7e20, 100X 124M)
#   - Training steps: 286,102 iterations
#   - Tokens per step: 524,288
#
# Hardware requirements:
#   - 8× NVIDIA A100 80GB GPUs (or equivalent)
#   - Training time: ~135 hours (~5.6 days, ~1.7s/iteration)
#   - Estimated cost: ~$2,000 on cloud infrastructure (100X cost of 124M)
#
# Prerequisites:
#   1. Build trainer: make train_gpt2cu USE_CUDNN=1
#   2. Download FineWeb 100B data: python dev/data/fineweb.py --version 100B
#   3. (Optional) Prepare eval data: python dev/data/hellaswag.py
#   4. Ensure MPI is installed for multi-GPU training
#
# Usage:
#   bash scripts/run_gpt2_774M.sh
# ==============================================================================

# Build the CUDA training executable with cuDNN flash attention
make train_gpt2cu USE_CUDNN=1

# Output directory for checkpoints and logs
out_dir="log_gpt2_774M"

# Training completion marker
done_file="$out_dir/DONE_00286102"

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
                -i "dev/data/fineweb100B/fineweb_train_*.bin" \
                -j "dev/data/fineweb100B/fineweb_val_*.bin" \
                -o $out_dir \
                -v 250 -s 300000 -g 144 \
                -h 1 \
                -b 32 -t 1024 \
                -d 524288 \
                -r 0 \
                -z 1 \
                -c 0.1 \
                -l 0.00025 \
                -q 0.0 \
                -u 700 \
                -n 4000 \
                -x 286102 \
                -y 1 \
                -e "d36"

    sleep 1
done

# ==============================================================================
# Training Parameter Explanation
# ==============================================================================
# -i: Training data (FineWeb 100B tokenized binary shards)
# -j: Validation data
# -o: Output directory for checkpoints and logs
# -v 250: Validate every 250 steps
# -s 300000: Save checkpoint every 300,000 steps
# -g 144: Generate text samples every 144 steps
# -h 1: Enable mixed precision (BFloat16)
# -b 32: Batch size per GPU (reduced from 64 due to larger model)
# -t 1024: Sequence length (context window)
# -d 524288: Total batch size across all GPUs
# -r 0: Random seed (0 = default)
# -z 1: Enable ZeRO optimization
# -c 0.1: Weight decay (L2 regularization)
# -l 0.00025: Learning rate (2.5e-4, lower than smaller models)
# -q 0.0: Gradient clipping (disabled)
# -u 700: Warmup iterations
# -n 4000: Keep checkpoints every 4000 steps
# -x 286102: Maximum iterations (training stops at this point)
# -y 1: Resume from latest checkpoint if available
# -e "d36": Model architecture (36 layers, 1280 dim, GPT-2 774M)
