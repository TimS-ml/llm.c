#!/bin/bash
# ==============================================================================
# FineWeb Dataset Downloader
# ==============================================================================
# This script downloads the FineWeb dataset in pre-tokenized format (.bin files).
# FineWeb is a large-scale, high-quality web corpus for language model training.
#
# Dataset information:
#   - Source: HuggingFace (chrisdryden/FineWebTokenizedGPT2)
#   - Format: Binary tokenized files (GPT-2 tokenizer)
#   - Total shards: 1,028 training shards + 1 validation shard
#   - Shard naming: fineweb_train_000001.bin through fineweb_train_001028.bin
#   - Each shard contains tokenized text data
#
# Usage:
#   ./fineweb.sh            # Download all 1,028 shards (default)
#   ./fineweb.sh 100        # Download first 100 shards only
#   ./fineweb.sh 500        # Download first 500 shards only
#
# Storage requirements:
#   - Each shard: ~100MB
#   - Full dataset (1,028 shards): ~100GB
#   - Partial downloads can save disk space for testing
#
# Prerequisites:
#   - curl (for downloading)
#   - Sufficient disk space
#   - Internet connection
#
# Output:
#   - Creates fineweb100B/ directory
#   - Downloads files in parallel (40 concurrent downloads)
# ==============================================================================

# Parse command line arguments
# If no argument provided, download all shards (1028)
# Otherwise, download the specified number of shards
if [ $# -eq 0 ]; then
    MAX_SHARDS=1028
else
    MAX_SHARDS=$1
fi

# Validate shard count - cap at maximum available shards
if [ $MAX_SHARDS -gt 1028 ]; then
    MAX_SHARDS=1028
fi

# ==============================================================================
# Configuration
# ==============================================================================

# HuggingFace dataset URLs
TRAIN_BASE_URL="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/resolve/main/fineweb_train_"
VAL_URL="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/resolve/main/fineweb_val_000000.bin?download=true"

# Local directory for downloaded files
SAVE_DIR="fineweb100B"

# Create the directory if it doesn't exist
mkdir -p "$SAVE_DIR"

# ==============================================================================
# Download Functions
# ==============================================================================

# Function to download a single file
# Args: $1 = URL to download from
download() {
    local FILE_URL=$1
    # Extract filename from URL (remove query parameters)
    local FILE_NAME=$(basename $FILE_URL | cut -d'?' -f1)
    local FILE_PATH="${SAVE_DIR}/${FILE_NAME}"

    # Download using curl
    # -s: Silent mode (no progress bar)
    # -L: Follow redirects
    # -o: Output file path
    curl -s -L -o "$FILE_PATH" "$FILE_URL"
    echo "Downloaded $FILE_NAME to $SAVE_DIR"
}

# Function to run commands in parallel with job control
# Args: $1 = max concurrent jobs, $@ = array of commands to run
run_in_parallel() {
    local max_jobs=$1
    shift
    local commands=("$@")
    local job_count=0

    # Launch commands in parallel, respecting max_jobs limit
    for cmd in "${commands[@]}"; do
        eval "$cmd" &
        ((job_count++))
        # If we've hit the limit, wait for one job to finish
        if (( job_count >= max_jobs )); then
            wait -n  # Wait for next job to complete
            ((job_count--))
        fi
    done

    # Wait for any remaining background jobs to finish
    wait
}

# Export the download function so it's available in subshells
export -f download

# ==============================================================================
# Download Execution
# ==============================================================================

# Download validation shard in background
download "$VAL_URL" &

# Generate download commands for training shards
# Format: fineweb_train_000001.bin, fineweb_train_000002.bin, etc.
train_commands=()
for i in $(seq -f "%06g" 1 $MAX_SHARDS); do
    FILE_URL="${TRAIN_BASE_URL}${i}.bin?download=true"
    train_commands+=("download \"$FILE_URL\"")
done

# Execute downloads in parallel (40 concurrent downloads)
# Adjust this number based on your bandwidth and system capacity
run_in_parallel 40 "${train_commands[@]}"

# Completion message
echo "=========================================="
echo "Download complete!"
echo "Downloaded validation shard and $MAX_SHARDS training shards"
echo "Location: $SAVE_DIR"
echo "=========================================="
