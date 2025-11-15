#!/bin/bash
# ==============================================================================
# GPT-3 125M Training Script on FineWeb
# ==============================================================================
# This script trains a GPT-3 style 125M parameter model on the FineWeb dataset.
# It follows GPT-3 architecture with 2048 context length (vs 1024 for GPT-2).
#
# Model specifications:
#   - Parameters: 125M (GPT-3 architecture with 768 hidden dim)
#   - Training tokens: ~300B from FineWeb (100B dataset)
#   - Context length: 2048 (double GPT-2's 1024)
#   - Compute budget: 6 × 125M × 300B = 2.25e20 FLOPs (~2e20)
#   - Training steps: 572,204 iterations
#   - Tokens per step: 524,288
#
# Hardware requirements:
#   - 8× NVIDIA A100 80GB GPUs (or equivalent)
#   - Training time: ~24 hours (~150ms/iteration)
#   - Estimated cost: ~$336 on cloud infrastructure ($14/hr for 8×A100)
#
# Prerequisites:
#   1. Build trainer: make train_gpt2cu USE_CUDNN=1
#   2. Download FineWeb 100B data: python dev/data/fineweb.py --version 100B
#   3. Ensure MPI is installed for multi-GPU training
#
# Key differences from GPT-2 124M:
#   - Longer context (2048 vs 1024)
#   - More training data (300B vs 10B tokens)
#   - GPT-3 architecture with modern improvements
#
# Usage:
#   bash scripts/run_gpt3_125M.sh
# ==============================================================================

# Build the CUDA training executable with cuDNN flash attention
make train_gpt2cu USE_CUDNN=1

# Output directory for checkpoints and logs
out_dir="log_gpt3_125M"

# Training completion marker
done_file="$out_dir/DONE_00572204"

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
                -v 250 -s 20000 -g 144 \
                -h 1 \
                -b 32 -t 2048 \
                -d 524288 \
                -r 0 \
                -z 1 \
                -c 0.1 \
                -l 0.0006 \
                -q 0.1 \
                -u 700 \
                -n 10000 \
                -nk 5 \
                -nm 50000 \
                -ge 1 \
                -sl 7.0 \
                -sg 7.0 \
                -y 1 \
                -x 572204 \
                -e "gpt3:c768"

    sleep 1
done

# ==============================================================================
# Training Parameter Explanation
# ==============================================================================
# -i: Training data (FineWeb 100B tokenized binary shards)
# -j: Validation data
# -o: Output directory for checkpoints and logs
# -v 250: Validate every 250 steps
# -s 20000: Save checkpoint every 20,000 steps
# -g 144: Generate text samples every 144 steps
# -h 1: Enable mixed precision (BFloat16)
# -b 32: Batch size per GPU
# -t 2048: Sequence length (GPT-3 uses 2048 vs GPT-2's 1024)
# -d 524288: Total batch size across all GPUs
# -r 0: Random seed (0 = default)
# -z 1: Enable ZeRO optimization
# -c 0.1: Weight decay (L2 regularization)
# -l 0.0006: Learning rate (6e-4)
# -q 0.1: Gradient clipping (0.1, enabled for stability)
# -u 700: Warmup iterations
# -n 10000: Keep checkpoints every 10,000 steps
# -nk 5: Keep N most recent checkpoints
# -nm 50000: Keep checkpoints at multiples of this number
# -ge 1: Gradient accumulation enabled
# -sl 7.0: Softmax temperature for logits
# -sg 7.0: Softmax temperature for gradients
# -y 1: Resume from latest checkpoint if available
# -x 572204: Maximum iterations (training stops at this point)
# -e "gpt3:c768": Model architecture (GPT-3 with 768 hidden dim)
