#!/bin/bash
# ==============================================================================
# GPT-2 350M Training Script on FineWeb
# ==============================================================================
# This script trains a GPT-2 350M parameter model (GPT-2 Medium) on FineWeb.
# This is approximately 10X the compute of the 124M model.
#
# Model specifications:
#   - Parameters: 350M (d24 architecture: 24 layers, 1024 hidden dim)
#   - Training tokens: ~31.5B from FineWeb (100B dataset)
#   - Compute budget: 6 × 350M × 31.5B = 6.615e19 FLOPs (~7e19, 10X 124M)
#   - Training steps: 60,000 iterations
#   - Tokens per step: 524,288
#
# Hardware requirements:
#   - 8× NVIDIA A100 80GB GPUs (or equivalent)
#   - Training time: ~13.7 hours (~820ms/iteration)
#   - Estimated cost: ~$200 on cloud infrastructure (10X cost of 124M)
#
# Prerequisites:
#   1. Build trainer: make train_gpt2cu USE_CUDNN=1
#   2. Download FineWeb 100B data: python dev/data/fineweb.py --version 100B
#   3. (Optional) Prepare eval data: python dev/data/hellaswag.py
#   4. Ensure MPI is installed for multi-GPU training
#
# Usage:
#   bash scripts/run_gpt2_350M.sh
# ==============================================================================

# Build the CUDA training executable with cuDNN flash attention
make train_gpt2cu USE_CUDNN=1

# Output directory for checkpoints and logs
out_dir="log_gpt2_350M"

# Training completion marker
done_file="$out_dir/DONE_00060000"

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
                -v 250 -s 100000 -g 144 \
                -h 1 \
                -b 64 -t 1024 \
                -d 524288 \
                -r 0 \
                -z 1 \
                -c 0.1 \
                -l 0.0003 \
                -q 0.0 \
                -u 700 \
                -n 2000 \
                -x 60000 \
                -y 1 \
                -e "d24"

    sleep 1
done

# ==============================================================================
# Training Parameter Explanation
# ==============================================================================
# -i: Training data (FineWeb 100B tokenized binary shards)
# -j: Validation data
# -o: Output directory for checkpoints and logs
# -v 250: Validate every 250 steps
# -s 100000: Save checkpoint every 100,000 steps
# -g 144: Generate text samples every 144 steps
# -h 1: Enable mixed precision (BFloat16)
# -b 64: Batch size per GPU
# -t 1024: Sequence length (context window)
# -d 524288: Total batch size across all GPUs
# -r 0: Random seed (0 = default)
# -z 1: Enable ZeRO optimization
# -c 0.1: Weight decay (L2 regularization)
# -l 0.0003: Learning rate (3e-4, lower than 124M due to larger model)
# -q 0.0: Gradient clipping (disabled)
# -u 700: Warmup iterations
# -n 2000: Keep checkpoints every 2000 steps
# -x 60000: Maximum iterations (training stops at this point)
# -y 1: Resume from latest checkpoint if available
# -e "d24": Model architecture (24 layers, 1024 dim, GPT-2 350M)
