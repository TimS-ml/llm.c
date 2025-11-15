"""
TinyShakespeare Dataset Preparation for Language Model Training

This script downloads and tokenizes the TinyShakespeare dataset, which contains
~1MB of Shakespeare's plays. It's a popular small dataset for testing language
models and training character/word-level models.

Dataset Source:
    - Original: Andrej Karpathy's char-rnn repository on GitHub
    - Size: ~1.1 million characters, ~338,025 tokens (GPT-2)
    - Content: Concatenated Shakespeare plays

Processing Steps:
    1. Downloads raw text from GitHub (if not cached)
    2. Splits text into sections separated by blank lines
    3. Tokenizes using either GPT-2 or LLaMA-3 tokenizer
    4. Splits into train/validation (90/10 split at 32,768 tokens)
    5. Saves as binary files for efficient C/CUDA loading

Output Files:
    - tinyshakespeare/tiny_shakespeare.txt: Raw text
    - tinyshakespeare/tiny_shakespeare_val.bin: Validation tokens (first 32K)
    - tinyshakespeare/tiny_shakespeare_train.bin: Training tokens (rest)

Usage Examples:
    # Process with GPT-2 tokenizer (default)
    $ python dev/data/tinyshakespeare.py --model=gpt-2
    writing 32,768 tokens to .../tiny_shakespeare_val.bin (66,560 bytes) in the gpt-2 format
    writing 305,260 tokens to .../tiny_shakespeare_train.bin (611,544 bytes) in the gpt-2 format

    # Process with LLaMA-3 tokenizer
    $ python dev/data/tinyshakespeare.py --model=llama-3
    writing 32,768 tokens to .../tiny_shakespeare_val.bin (132,096 bytes) in the llama-3 format
    writing 276,224 tokens to .../tiny_shakespeare_train.bin (1,105,920 bytes) in the llama-3 format

Performance:
    Runs in a few seconds depending on internet connection and computer speed.

Data Format:
    The .bin files contain:
    - Header: 256 int32 values (metadata)
    - Data: uint16 (GPT-2) or uint32 (LLaMA-3) token IDs

Note:
    Token counts differ between tokenizers due to vocabulary differences.
    GPT-2 has ~50K vocab, LLaMA-3 has ~128K vocab.
"""

import argparse
import os

import tiktoken
from transformers import AutoTokenizer

from data_common import download_file, write_datafile

# -----------------------------------------------------------------------------
# Directory where dataset will be cached (sibling to this script)
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tinyshakespeare")

def download():
    """
    Download the TinyShakespeare dataset from GitHub.

    Downloads the raw text file from Andrej Karpathy's char-rnn repository if it
    doesn't already exist locally. The file is cached to avoid repeated downloads.

    The dataset contains ~1MB of Shakespeare's plays concatenated together.

    Returns:
        None

    Side Effects:
        - Creates DATA_CACHE_DIR if it doesn't exist
        - Downloads tiny_shakespeare.txt if not already present
    """
    # Create cache directory if it doesn't exist
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # URL to the raw TinyShakespeare text file
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare.txt")

    # Download only if not already cached
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

def tokenize(model_desc):
    """
    Tokenize the TinyShakespeare text and split into train/validation sets.

    This function loads the raw text, tokenizes it using the specified model's tokenizer,
    and saves the tokens as binary files. The text is split into sections (separated by
    blank lines) and each section is treated as a separate document with an end-of-text
    token delimiter.

    Args:
        model_desc: Model type, either "gpt-2" or "llama-3"

    Returns:
        None

    Side Effects:
        - Creates tiny_shakespeare_val.bin (first 32,768 tokens)
        - Creates tiny_shakespeare_train.bin (remaining tokens)

    Raises:
        ValueError: If model_desc is not "gpt-2" or "llama-3"

    Token Counts:
        GPT-2:    ~338K total tokens (32K val, 305K train)
        LLaMA-3:  ~309K total tokens (32K val, 276K train)

    Note:
        The first 32,768 tokens (~10%) are used for validation.
        This fixed split ensures consistent evaluation across runs.
    """
    # Initialize the appropriate tokenizer based on model type
    if model_desc == "gpt-2":
        # Use tiktoken for fast GPT-2 tokenization
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode_ordinary(s)
        eot = enc._special_tokens['<|endoftext|>']  # End-of-text token ID (50256)
    elif model_desc == "llama-3":
        # Use HuggingFace transformers for LLaMA-3 tokenization
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        # Disable special token insertion to have manual control
        encode = lambda s: tokenizer.encode(s, add_special_tokens=False, verbose=False, split_special_tokens=True)
        # Get the EOT token (LLaMA-3 uses token ID 128000)
        eot = tokenizer.encode('')[0]
    else:
        raise ValueError(f"unknown model descriptor {model_desc}")

    # Load the raw text file
    data_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare.txt")
    text = open(data_filename, 'r').read()

    # Split text into sections (separated by blank lines)
    # Each section represents a logical document boundary
    sections = text.split("\n\n")

    # Tokenize all sections with EOT delimiters
    tokens = []
    for i, s in enumerate(sections):
        # Add end-of-text token before each section to mark document boundaries
        tokens.append(eot)

        # Backward compatibility note: Original implementation had a bug where \n\n
        # was not removed but EOT was added before it. We preserve this behavior
        # by adding \n\n back to all sections except the last one
        spad = s + "\n\n" if i != len(sections) - 1 else s

        # Tokenize the section and add to token list
        tokens.extend(encode(spad))

    # Split into validation and training sets
    # First 32,768 tokens (~10%) for validation, rest for training
    val_tokens = tokens[:32768]
    train_tokens = tokens[32768:]

    # Write binary files for efficient loading in C/CUDA code
    val_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_val.bin")
    train_filename = os.path.join(DATA_CACHE_DIR, "tiny_shakespeare_train.bin")
    write_datafile(val_filename, val_tokens, model_desc)
    write_datafile(train_filename, train_tokens, model_desc)

if __name__ == "__main__":
    # Command-line interface for dataset preparation
    parser = argparse.ArgumentParser(description="Tiny Shakespeare dataset preprocessing")
    parser.add_argument("-m", "--model_desc", type=str, default="gpt-2",
                        choices=["gpt-2", "llama-3"],
                        help="Model type, gpt-2|llama-3")
    args = parser.parse_args()

    # Execute the pipeline: download then tokenize
    download()
    tokenize(args.model_desc)
