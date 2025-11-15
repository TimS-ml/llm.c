#!/bin/bash
# ==============================================================================
# Multi-Node Multi-GPU GPT-2 124M Training Script (MPI)
# ==============================================================================
# This script demonstrates how to run distributed training across multiple nodes
# using MPI (Message Passing Interface) and NCCL for GPU communication.
#
# Hardware setup:
#   - 2 nodes with 8 GPUs each (16 GPUs total)
#   - H100 80GB GPUs recommended
#   - InfiniBand or high-speed network interconnect recommended
#   - Shared or synchronized filesystem across nodes
#
# Prerequisites:
#   1. Build the GPU trainer: make train_gpt2cu USE_CUDNN=1
#   2. Install MPI (OpenMPI recommended)
#   3. Install NCCL for multi-GPU communication
#   4. Set up passwordless SSH between nodes
#   5. Ensure network configuration (InfiniBand, RDMA, etc.)
#   6. Download training data on all nodes or use shared filesystem
#
# Configuration:
#   - Modify binary_path, out_dir, and data paths for your system
#   - Update host1 and host2 with your node hostnames
#   - Adjust NCCL environment variables for your network setup
#
# Usage:
#   bash scripts/multi_node/run_gpt2_124M_mpi.sh
# ==============================================================================

# Build the CUDA training executable with cuDNN support
make train_gpt2cu USE_CUDNN=1

# ==============================================================================
# System Configuration - CUSTOMIZE THESE FOR YOUR SETUP
# ==============================================================================

# NOTE: Change these paths to match your system
binary_path="/home/ubuntu/llm.c/train_gpt2cu"
out_dir="/ephemeral/data/fineweb/log_gpt2_124M_multi"
train_data_path='/ephemeral/data/fineweb/bin_10B/fineweb_train_*.bin'
val_data_path='/ephemeral/data/fineweb/bin_10B/fineweb_val_*.bin'

# Node hostnames
# You can find these in `/etc/hosts` file or terminal prompt (user@host:~$)
host1="h100-node-1-0"  # master and worker node
host2="h100-node-1-1"  # worker node

# ==============================================================================
# Binary Distribution
# ==============================================================================

# Copy the compiled binary to all worker nodes
# If using a shared filesystem (e.g., NFS), this is unnecessary
# Otherwise, ensure the binary exists on all nodes
scp -r $binary_path $USER@$host2:$binary_path

# ==============================================================================
# Environment Configuration
# ==============================================================================

# NCCL Debugging (uncomment if experiencing issues)
# export NCCL_DEBUG=INFO          # Enable verbose NCCL logging
# export NCCL_DEBUG_SUBSYS=ALL    # Log all NCCL subsystems

# GPU Selection
# Specify which GPUs to use on each node (0-7 for 8 GPUs)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL Performance Optimization Flags
# GPUDirect RDMA: Enable direct GPU-to-GPU communication across nodes
# Level 2 = Use GPUDirect RDMA when available (bypasses CPU for inter-node transfers)
export NCCL_NET_GDR_LEVEL=2

# InfiniBand: Enable InfiniBand for high-speed inter-node communication
# 0 = enabled, 1 = disabled
export NCCL_IB_DISABLE=0

# Network Interface Configuration
# NOTE: Customize these for your network setup or comment out if not needed
# ens17 is an example interface name - use `ifconfig` or `ip addr` to find yours
export NCCL_SOCKET_IFNAME=ens17           # NCCL network interface
export OMPI_MCA_btl_tcp_if_include=ens17  # MPI TCP interface
export NCCL_P2P_LEVEL=PXB                 # P2P communication level (PXB = PCIe crossbar)

# ==============================================================================
# Launch Distributed Training
# ==============================================================================

# Run training across 16 processes (8 GPUs × 2 nodes)
# -np 16: Total number of processes (16 GPUs)
# --host $host1:8,$host2:8: 8 processes on each host
mpirun -np 16 --host $host1:8,$host2:8 \
    $binary_path \
    -i "$train_data_path" \
    -j "$val_data_path" \
    -o $out_dir \
    -v 250 -s 20000 -g 144 \
    -h 1 \
    -b 64 -t 1024 \
    -d 2097152 \
    -r 0 \
    -z 1 \
    -c 0.1 \
    -l 0.0006 \
    -q 0.1 \
    -u 700 \
    -n 1000 \
    -y 0 \
    -e d12 \
    -pi "mpi"

# ==============================================================================
# Training Parameter Explanation
# ==============================================================================
# -i: Input training data path (binary tokenized format)
# -j: Input validation data path
# -o: Output directory for checkpoints and logs
# -v: Validation interval (compute validation loss every N steps)
# -s: Checkpoint save interval (save model every N steps)
# -g: Generation interval (generate text samples every N steps)
# -h: Use float16/bfloat16 (1 = enabled)
# -b: Batch size per GPU (64 sequences)
# -t: Sequence length / context window (1024 tokens)
# -d: Total batch size across all GPUs (2,097,152 tokens)
# -r: Random seed (0 = use default)
# -z: Use zero optimization (1 = enabled)
# -c: Weight decay for regularization (0.1)
# -l: Learning rate (6e-4)
# -q: Gradient clipping value (0.1)
# -u: Warmup iterations for learning rate (700)
# -n: Checkpoint keep interval (keep every Nth checkpoint)
# -y: Resume from checkpoint (0 = start fresh, 1 = resume)
# -e: Model architecture (d12 = 12 layers, 768 dim, GPT-2 124M)
# -pi: Process initialization mode ("mpi" for MPI-based multi-node) \
