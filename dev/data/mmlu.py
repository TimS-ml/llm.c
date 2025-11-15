"""
MMLU (Massive Multitask Language Understanding) Benchmark Evaluation

MMLU is a comprehensive benchmark testing knowledge across 57 diverse subjects
including STEM, humanities, social sciences, and more. It evaluates a model's
world knowledge and problem-solving abilities through multiple-choice questions.

Dataset Information:
    - Paper: "Measuring Massive Multitask Language Understanding"
    - Source: https://github.com/hendrycks/test
    - Size: 14,042 test examples across 57 subjects
    - Task: Multiple-choice question answering (4 choices: A, B, C, D)
    - Subjects: Math, science, history, law, medicine, etc.
    - Metric: Accuracy and normalized accuracy

This Script:
    1. Downloads MMLU test data from Berkeley (tar archive)
    2. Evaluates a PyTorch model using completion-style scoring
    3. Reports accuracy across all subjects combined

Data Format:
    CSV files (no header) with columns:
    [question, choice_A, choice_B, choice_C, choice_D, answer]

    Example row:
    "What is 2+2?", "3", "4", "5", "6", "B"

Question Format:
    The script reformats each example as:
    "Question: {question}\n\nAnswer: {choice}"

    This prompt format helps the model understand the task.

Evaluation Method:
    Similar to HellaSwag, uses completion-style evaluation:
    - Compute log-likelihood of each answer choice
    - Two scoring methods:
        * acc: Use sum of log-likelihoods (favors longer answers)
        * acc_norm: Use average log-likelihood (normalized by length)
    - Select choice with lowest loss (highest likelihood)

Benchmark Results:
    GPT-2 (124M):
        - This script: acc 25.57%, acc_norm 27.21%
        - Random chance: 25% (4 choices)

    GPT-2-XL (1558M):
        - This script: acc 29.27%, acc_norm 30.35%

Note:
    MMLU is designed for large models. Small models like GPT-2 perform
    only slightly above random chance. State-of-the-art models achieve
    70-90% accuracy.

Usage:
    # Evaluate GPT-2 model
    $ python dev/data/mmlu.py -m gpt2 -d cuda

    # Evaluate GPT-2-XL
    $ python dev/data/mmlu.py -m gpt2-xl -d cuda

Output:
    - Console: Running accuracy statistics
    - Detailed predictions for first 10 examples
    - No binary export (unlike HellaSwag)

Dataset Statistics:
    - Total: 14,042 test examples
    - 57 subjects with varying numbers of questions
    - Questions range from factual recall to complex reasoning
"""

import os
import requests
import tiktoken
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from data_common import download_file

# -----------------------------------------------------------------------------
# Directory where dataset will be cached (sibling to this script)
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "mmlu")

# Initialize GPT-2 tokenizer (used for all evaluations)
enc = tiktoken.get_encoding("gpt2")

# URL for MMLU dataset (tar archive containing CSV files for all subjects)
data_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

def download():
    """
    Download and extract the MMLU dataset from Berkeley.

    Downloads a tar archive containing CSV files for all 57 MMLU subjects.
    The archive is automatically extracted to create a 'data/test/' directory
    with one CSV file per subject.

    Returns:
        None

    Side Effects:
        - Creates DATA_CACHE_DIR if it doesn't exist
        - Downloads data.tar if not already present
        - Extracts to data/test/ subdirectory (57 CSV files)

    Note:
        The extraction creates a 'data' directory inside DATA_CACHE_DIR,
        containing subdirectories for different splits (train, dev, test).
        Only test split is used in this script.
    """
    # Create cache directory if needed
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # Download and extract the dataset
    data_filename = os.path.join(DATA_CACHE_DIR, f"data.tar")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
        # Extract tar archive (creates data/test/, data/dev/, data/val/ subdirectories)
        os.system(f"tar -xf {data_filename} -C {DATA_CACHE_DIR}")
    else:
        print(f"{data_filename} already exists, skipping download...")

def iterate_examples():
    """
    Iterate through all MMLU test examples across all subjects.

    Reads CSV files for all 57 subjects and yields examples one at a time.
    This is memory-efficient and processes subjects sequentially.

    Yields:
        Dictionary for each example:
            - question: Question text (string)
            - endings: List of 4 answer choices (strings)
            - label: Correct answer as letter "A", "B", "C", or "D"

    Example:
        >>> for example in iterate_examples():
        ...     print(f"{example['question']}: {example['label']}")

    Note:
        Total of 14,042 examples across all subjects.
        Each CSV file represents one subject (e.g., chemistry, history).
    """
    # Ensure dataset is downloaded and extracted
    download()

    # Get all subject CSV files from test directory
    test_dir = os.path.join(DATA_CACHE_DIR, "data", "test")
    csv_files = [f for f in os.listdir(test_dir) if f.endswith(".csv")]

    # Process each subject file
    for csv_file in csv_files:
        csv_path = os.path.join(test_dir, csv_file)
        print(csv_path)  # Print which subject is being processed

        # Load CSV (no header row)
        # Columns: [question, choice_A, choice_B, choice_C, choice_D, answer]
        df = pd.read_csv(csv_path, header=None)
        n = df.shape[0]

        # Yield each question in this subject
        for idx in range(n):
            example = {
                "question": df.iloc[idx, 0],  # Question text
                "endings": [  # Four answer choices
                    df.iloc[idx, 1],  # Choice A
                    df.iloc[idx, 2],  # Choice B
                    df.iloc[idx, 3],  # Choice C
                    df.iloc[idx, 4]   # Choice D
                ],
                "label": df.iloc[idx, 5],  # Correct answer ("A", "B", "C", or "D")
            }
            yield example

def render_example(example):
    """
    Convert an MMLU example into tensors for model evaluation.

    Formats the question and answer choices into a prompt, tokenizes,
    and creates padded tensors with masks for evaluation.

    Args:
        example: Dictionary with keys "question", "endings", "label"
            - question: Question text (string)
            - endings: List of 4 answer choices (strings)
            - label: Correct answer as letter ("A", "B", "C", or "D")

    Returns:
        Tuple of (tokens, mask, label):
            - tokens: Tensor [4, max_len] of token IDs (padded with 0s)
            - mask: Tensor [4, max_len] where 1 = evaluate, 0 = ignore
            - label: Integer index of correct answer (0-3)

    Prompt Format:
        Each example is formatted as:
        "Question: {question}\n\nAnswer: {choice}"

        This format helps the model understand it should answer a question.

    Token Format:
        Each row: [question_tokens... "Answer:" | choice_tokens... | padding...]
        Corresponding mask: [0, 0, ..., 0 | 1, 1, ..., 1 | 0, 0, ...]

    Note:
        - Answer choices are prepended with " " for proper GPT-2 tokenization
        - Rows are padded to max_len for batched evaluation
        - Label is converted from letter ("A"-"D") to index (0-3)
    """
    # Format the prompt with question
    ctx = f"Question: {example['question']}\n\nAnswer:"
    ctx_tokens = enc.encode(ctx)

    # Build token sequences for each answer choice
    tok_rows = []
    mask_rows = []
    for end in example["endings"]:
        # Prepend " " to answer for proper tokenization
        # Convert to string in case answer is numeric
        end_tokens = enc.encode(" " + str(end))
        # Concatenate question + answer tokens
        tok_rows.append(ctx_tokens + end_tokens)
        # Mask: 0 for question (don't evaluate), 1 for answer (do evaluate)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))

    # Pad all rows to same length for batched processing
    # Rows can have different lengths due to variable answer lengths
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)  # 4 answer choices
    mask = torch.zeros((4, max_len), dtype=torch.long)

    # Copy token and mask data into padded tensors
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    # Convert label from letter ("A", "B", "C", "D") to index (0, 1, 2, 3)
    label = "ABCD".index(example["label"])

    return tokens, mask, label

@torch.no_grad()
def evaluate(model_type, device):
    """
    Evaluate a GPT-2 model on the MMLU test set.

    This function:
    1. Loads a pretrained GPT-2 model from HuggingFace
    2. Evaluates all test examples using completion-style scoring
    3. Computes two accuracy metrics (sum and normalized)
    4. Prints first 10 examples with predictions for debugging

    Args:
        model_type: HuggingFace model name (e.g., "gpt2", "gpt2-xl")
        device: Device to run on (e.g., "cuda", "cpu")

    Returns:
        None

    Side Effects:
        - Prints running accuracy statistics after each example
        - Shows detailed predictions for first 10 examples

    Evaluation Method:
        For each question with 4 answer choices:
        1. Format as "Question: {question}\n\nAnswer: {choice}"
        2. Compute cross-entropy loss for each answer choice
        3. Two scoring methods:
            - acc: Use sum of losses (favors longer answers)
            - acc_norm: Use average loss (normalized by length)
        4. Select answer with lowest loss
        5. Compare to ground truth

    Metrics:
        - acc: Accuracy using sum of log-likelihoods
        - acc_norm: Accuracy using average log-likelihood (length-normalized)

    Expected Performance:
        GPT-2 (124M): ~25-27% (slightly above random 25%)
        GPT-2-XL (1558M): ~29-30%
        Random baseline: 25% (4 choices)

    Note:
        MMLU is designed for large models. Small models perform poorly.
        State-of-the-art models (GPT-4, etc.) achieve 70-90%.
        Uses @torch.no_grad() for memory efficiency.
    """
    # Use TensorFloat-32 for faster matrix multiplication on modern GPUs
    torch.set_float32_matmul_precision('high')

    # Load pretrained GPT-2 model from HuggingFace
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    # Optional: compile model for faster inference (uncomment if desired)
    # model = torch.compile(model)

    # Accuracy tracking
    num_correct_norm = 0  # Count for normalized accuracy
    num_correct = 0  # Count for sum-based accuracy
    num_total = 0

    # Evaluate each example across all subjects
    for example in iterate_examples():
        # Convert example to tensors
        tokens, mask, label = render_example(example)
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

        # Apply mask to compute loss only on answer region (not question)
        # Shift mask to align with shifted tokens
        shift_mask = (mask[..., 1:]).contiguous()
        masked_shift_losses = shift_losses * shift_mask

        # Compute two types of scores
        sum_loss = masked_shift_losses.sum(dim=1)  # Sum of losses
        avg_loss = sum_loss / shift_mask.sum(dim=1)  # Average loss (normalized)

        # Select answer with lowest loss (highest likelihood)
        pred = sum_loss.argmin().item()  # Prediction using sum
        pred_norm = avg_loss.argmin().item()  # Prediction using average

        # Update accuracy statistics
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

        # Print running statistics
        print(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm/num_total:.4f}")

        # Print detailed info for first 10 examples (debugging)
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['question']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred}, actual: {label}")

if __name__ == "__main__":
    # Command-line interface for MMLU evaluation
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate models on MMLU benchmark")
    parser.add_argument("-m", "--model_type", type=str, default="gpt2",
                        help="HuggingFace model name (e.g., gpt2, gpt2-xl)")
    parser.add_argument("-d", "--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    args = parser.parse_args()

    # Run evaluation on all MMLU subjects
    evaluate(args.model_type, args.device)
