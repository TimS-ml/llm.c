#!/bin/bash
# ==============================================================================
# llm.c Starter Pack Downloader
# ==============================================================================
# This script downloads essential files to get started with llm.c:
#   - Pre-trained GPT-2 124M model checkpoints
#   - GPT-2 tokenizer
#   - Tiny Shakespeare dataset (for quick testing)
#   - HellaSwag validation dataset (for evaluation)
#
# Purpose:
#   Provides everything needed to quickly test and validate the llm.c
#   implementation without training from scratch.
#
# Downloaded files:
#   1. gpt2_124M.bin            - GPT-2 124M checkpoint (FP32)
#   2. gpt2_124M_bf16.bin       - GPT-2 124M checkpoint (BF16)
#   3. gpt2_124M_debug_state.bin - Debug state for testing
#   4. gpt2_tokenizer.bin        - GPT-2 tokenizer (BPE)
#   5. tiny_shakespeare_train.bin - Tiny Shakespeare training data
#   6. tiny_shakespeare_val.bin   - Tiny Shakespeare validation data
#   7. hellaswag_val.bin          - HellaSwag validation set
#
# Total download size: ~500MB
#
# Usage:
#   bash dev/download_starter_pack.sh
#
# Output directories:
#   dev/                          - Model checkpoints and tokenizer
#   dev/data/tinyshakespeare/     - Tiny Shakespeare dataset
#   dev/data/hellaswag/           - HellaSwag evaluation set
#
# Prerequisites:
#   - curl (for downloading)
#   - Internet connection
# ==============================================================================

# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# ==============================================================================
# Configuration
# ==============================================================================

# HuggingFace dataset base URL
BASE_URL="https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/"

# Output directory paths (relative to script location)
SAVE_DIR_PARENT="$SCRIPT_DIR/.."           # For model files
SAVE_DIR_TINY="$SCRIPT_DIR/data/tinyshakespeare"  # For Shakespeare data
SAVE_DIR_HELLA="$SCRIPT_DIR/data/hellaswag"       # For HellaSwag eval

# Create the directories if they don't exist
mkdir -p "$SAVE_DIR_TINY"
mkdir -p "$SAVE_DIR_HELLA"

# Files to download
FILES=(
    "gpt2_124M.bin"                # GPT-2 124M model (FP32 precision)
    "gpt2_124M_bf16.bin"           # GPT-2 124M model (BF16 precision)
    "gpt2_124M_debug_state.bin"    # Debug state for testing
    "gpt2_tokenizer.bin"           # GPT-2 BPE tokenizer
    "tiny_shakespeare_train.bin"   # Tiny Shakespeare training set
    "tiny_shakespeare_val.bin"     # Tiny Shakespeare validation set
    "hellaswag_val.bin"            # HellaSwag evaluation set
)

# ==============================================================================
# Download Functions
# ==============================================================================

# Function to download a single file to the appropriate directory
# Args: $1 = filename
download_file() {
    local FILE_NAME=$1
    local FILE_URL="${BASE_URL}${FILE_NAME}?download=true"
    local FILE_PATH

    # Route file to appropriate directory based on name
    if [[ "$FILE_NAME" == tiny_shakespeare* ]]; then
        FILE_PATH="${SAVE_DIR_TINY}/${FILE_NAME}"
    elif [[ "$FILE_NAME" == hellaswag* ]]; then
        FILE_PATH="${SAVE_DIR_HELLA}/${FILE_NAME}"
    else
        # Model and tokenizer files go to parent directory (dev/)
        FILE_PATH="${SAVE_DIR_PARENT}/${FILE_NAME}"
    fi

    # Download using curl
    # -s: Silent mode
    # -L: Follow redirects
    # -o: Output file path
    curl -s -L -o "$FILE_PATH" "$FILE_URL"
    echo "Downloaded $FILE_NAME to $FILE_PATH"
}

# Export the function so it's available in subshells
export -f download_file

# Generate download commands for all files
download_commands=()
for FILE in "${FILES[@]}"; do
    download_commands+=("download_file \"$FILE\"")
done

# Function to run commands in parallel batches
# Args: $1 = batch size, $@ = commands to run
run_in_parallel() {
    local batch_size=$1
    shift
    local i=0
    local command

    # Execute commands in batches
    for command; do
        eval "$command" &
        ((i = (i + 1) % batch_size))
        # Wait for batch to complete before starting next batch
        if [ "$i" -eq 0 ]; then
            wait
        fi
    done

    # Wait for any remaining jobs to finish
    wait
}

# ==============================================================================
# Download Execution
# ==============================================================================

# Download all files in parallel (6 at a time)
run_in_parallel 6 "${download_commands[@]}"

# Completion message
echo "=========================================="
echo "Starter pack download complete!"
echo "Files saved to:"
echo "  - Model files: $SAVE_DIR_PARENT"
echo "  - Tiny Shakespeare: $SAVE_DIR_TINY"
echo "  - HellaSwag: $SAVE_DIR_HELLA"
echo "=========================================="