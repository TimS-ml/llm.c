"""
HellaSwag Benchmark Evaluation and Dataset Preparation

HellaSwag is a commonsense reasoning benchmark that tests a model's ability to
complete sentences with the most plausible ending. It contains sentences from
ActivityNet captions and WikiHow articles, with 4 completion choices per example.

Dataset Information:
    - Paper: "HellaSwag: Can a Machine Really Finish Your Sentence?"
    - Source: https://github.com/rowanz/hellaswag
    - Size: 10,042 validation examples, 39,905 training examples
    - Task: Multiple-choice sentence completion (4 choices)
    - Metric: Accuracy and normalized accuracy

This Script:
    1. Downloads HellaSwag data from GitHub (JSONL format)
    2. Evaluates a PyTorch model using completion-style scoring
    3. Exports data to binary format for C/CUDA evaluation

Example Data Structure:
    {
      "ind": 24,
      "activity_label": "Roof shingle removal",
      "ctx_a": "A man is sitting on a roof.",
      "ctx_b": "he",
      "ctx": "A man is sitting on a roof. he",
      "split": "val",
      "split_type": "indomain",
      "label": 3,
      "endings": [
        "is using wrap to wrap a pair of skis.",
        "is ripping level tiles off.",
        "is holding a rubik's cube.",
        "starts pulling up roofing on a roof."
      ],
      "source_id": "activitynet~v_-JhWjGDPHMY"
    }

Field Descriptions:
    - ind: Dataset ID number
    - activity_label: ActivityNet or WikiHow category
    - ctx_a: Context before incomplete noun phrase
    - ctx_b: Incomplete noun phrase (may be empty)
    - ctx: Full context (ctx_a + " " + ctx_b)
    - split: "train", "val", or "test"
    - split_type: "indomain" or "zeroshot"
    - label: Index of correct ending (0-3)
    - endings: List of 4 possible completions
    - source_id: Source video/article identifier

Evaluation Method:
    This script uses "completion style" evaluation:
    - Compute log-likelihood of each completion given the context
    - Two scoring methods:
        * acc: Use sum of log-likelihoods (favors longer completions)
        * acc_norm: Use average log-likelihood (normalized by length)
    - Select completion with lowest loss (highest likelihood)

Benchmark Results:
    GPT-2 (124M):
        - Eleuther harness: acc 28.92%, acc_norm 31.14% (multiple choice)
        - This script: acc 28.59%, acc_norm 29.55% (completion)

    GPT-2-XL (1558M):
        - Eleuther harness: acc 40.04%, acc_norm 50.89% (multiple choice)
        - This script: acc 38.42%, acc_norm 48.93% (completion)

Usage:
    # Evaluate GPT-2 model and export to binary
    $ python dev/data/hellaswag.py -m gpt2 -d cuda

    # Evaluate GPT-2-XL
    $ python dev/data/hellaswag.py -m gpt2-xl -d cuda

Output:
    - hellaswag/hellaswag_val.bin: Binary format for C evaluation
    - Console: Accuracy metrics and example predictions

Dataset Statistics:
    - Validation: 10,042 examples
    - Average context length: ~20 tokens
    - Average completion length: ~10-15 tokens
"""

import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from data_common import download_file, write_evalfile

# -----------------------------------------------------------------------------
# Directory where dataset will be cached (sibling to this script)
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

# URLs for different splits of HellaSwag dataset (JSONL format)
hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

# Initialize GPT-2 tokenizer (used for all evaluations in this script)
enc = tiktoken.get_encoding("gpt2")

def download(split):
    """
    Download HellaSwag dataset split from GitHub.

    Args:
        split: One of "train", "val", or "test"

    Returns:
        None

    Side Effects:
        - Creates DATA_CACHE_DIR if it doesn't exist
        - Downloads hellaswag_{split}.jsonl if not already present

    Example:
        >>> download("val")
        Downloading https://raw.githubusercontent.com/.../hellaswag_val.jsonl...
    """
    # Create cache directory if needed
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # Download the specified split
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

def render_example(example):
    """
    Convert a HellaSwag example into tensors for model evaluation.

    This function prepares data for both PyTorch evaluation and C export.
    It tokenizes the context and all 4 completion choices, creating padded
    tensors and masks to identify which tokens to evaluate.

    Args:
        example: Dictionary with keys "ctx", "label", "endings"
            - ctx: Context string (prompt)
            - label: Index of correct ending (0-3)
            - endings: List of 4 completion strings

    Returns:
        Tuple of (data, tokens, mask, label):
            - data: Dict for C export with ctx_tokens, ending_tokens, label
            - tokens: Tensor [4, max_len] of token IDs (padded with 0s)
            - mask: Tensor [4, max_len] where 1 = evaluate, 0 = ignore
            - label: Integer index of correct completion

    Token Format:
        Each row: [context_tokens... | ending_tokens... | padding...]
        Corresponding mask: [0, 0, ..., 0 | 1, 1, ..., 1 | 0, 0, ...]

    Note:
        - Endings are prepended with " " for proper GPT-2 tokenization
        - Rows are padded to max_len for batched evaluation
        - Mask identifies completion region for loss calculation
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # Data structure for C export (binary format)
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # Tokenize the context (shared across all 4 choices)
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens

    # Build token sequences for each completion choice
    tok_rows = []
    mask_rows = []
    for end in endings:
        # Prepend " " to ending for proper tokenization (GPT-2 treats leading space specially)
        end_tokens = enc.encode(" " + end)
        # Concatenate context + ending tokens
        tok_rows.append(ctx_tokens + end_tokens)
        # Mask: 0 for context (don't evaluate), 1 for ending (do evaluate)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        # Store ending tokens for C export
        data["ending_tokens"].append(end_tokens)

    # Pad all rows to same length for batched processing
    # Rows can have different lengths due to variable ending lengths
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)  # 4 choices
    mask = torch.zeros((4, max_len), dtype=torch.long)

    # Copy token and mask data into padded tensors
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def iterate_examples(split):
    """
    Iterate through HellaSwag examples from a JSONL file.

    Downloads the split if needed, then yields examples one at a time.
    This is memory-efficient for large datasets.

    Args:
        split: One of "train", "val", or "test"

    Yields:
        Dictionary for each example (see module docstring for structure)

    Example:
        >>> for example in iterate_examples("val"):
        ...     print(example["ctx"], "->", example["endings"][example["label"]])

    Note:
        Validation split has 10,042 examples total.
    """
    # Ensure data is downloaded
    download(split)

    # Stream examples from JSONL file (one JSON object per line)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

@torch.no_grad()
def evaluate(model_type, device):
    """
    Evaluate a GPT-2 model on HellaSwag validation set and export to binary.

    This function:
    1. Loads a pretrained GPT-2 model from HuggingFace
    2. Evaluates all validation examples using completion-style scoring
    3. Computes two accuracy metrics (sum and normalized)
    4. Exports tokenized data to binary format for C evaluation
    5. Prints first 10 examples with predictions for debugging

    Args:
        model_type: HuggingFace model name (e.g., "gpt2", "gpt2-xl")
        device: Device to run on (e.g., "cuda", "cpu")

    Returns:
        None

    Side Effects:
        - Prints running accuracy statistics
        - Writes hellaswag_val.bin for C evaluation
        - Shows detailed predictions for first 10 examples

    Evaluation Method:
        For each example with 4 completion choices:
        1. Compute cross-entropy loss for each completion
        2. Two scoring methods:
            - acc: Use sum of losses (favors longer completions)
            - acc_norm: Use average loss (normalized by length)
        3. Select completion with lowest loss
        4. Compare to ground truth label

    Metrics:
        - acc: Accuracy using sum of log-likelihoods
        - acc_norm: Accuracy using average log-likelihood (length-normalized)

    Note:
        Uses @torch.no_grad() for memory efficiency during evaluation.
        TF32 is enabled for faster matmul on Ampere+ GPUs.
    """
    # Use TensorFloat-32 for faster matrix multiplication on modern GPUs
    torch.set_float32_matmul_precision('high')

    # Load pretrained GPT-2 model from HuggingFace
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    # Optional: compile model for faster inference (uncomment if desired)
    # model = torch.compile(model)

    # Storage for C export and accuracy tracking
    datas = []  # Data for binary export
    num_correct_norm = 0  # Count for normalized accuracy
    num_correct = 0  # Count for sum-based accuracy
    num_total = 0

    # Evaluate each example in validation set
    for example in iterate_examples("val"):
        # Convert example to tensors
        data, tokens, mask, label = render_example(example)
        datas.append(data)  # Store for C export
        tokens = tokens.to(device)
        mask = mask.to(device)

        # Get model predictions (logits for next token at each position)
        logits = model(tokens).logits  # Shape: [4, seq_len, vocab_size]

        # Compute autoregressive cross-entropy loss
        # Shift logits and tokens: predict token i+1 from tokens 0..i
        shift_logits = (logits[..., :-1, :]).contiguous()  # Remove last position
        shift_tokens = (tokens[..., 1:]).contiguous()      # Remove first position

        # Flatten for cross_entropy computation
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)

        # Compute per-token cross-entropy loss
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)  # Shape: [4, seq_len-1]

        # Apply mask to compute loss only on completion region
        # Shift mask to align with shifted tokens
        shift_mask = (mask[..., 1:]).contiguous()
        masked_shift_losses = shift_losses * shift_mask

        # Compute two types of scores
        sum_loss = masked_shift_losses.sum(dim=1)  # Sum of losses (favors longer)
        avg_loss = sum_loss / shift_mask.sum(dim=1)  # Average loss (normalized)

        # Select completion with lowest loss (highest likelihood)
        pred = sum_loss.argmin().item()  # Prediction using sum
        pred_norm = avg_loss.argmin().item()  # Prediction using average

        # Update accuracy statistics
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

        # Print running statistics
        print(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # Print detailed info for first 10 examples (debugging)
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

    # Export tokenized data to binary format for C evaluation
    filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_val.bin")
    write_evalfile(filename, datas)

if __name__ == "__main__":
    # Command-line interface for HellaSwag evaluation
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate models on HellaSwag benchmark")
    parser.add_argument("-m", "--model_type", type=str, default="gpt2",
                        help="HuggingFace model name (e.g., gpt2, gpt2-xl)")
    parser.add_argument("-d", "--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    # Run evaluation and export to binary
    evaluate(args.model_type, args.device)
