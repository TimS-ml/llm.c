"""
FineWeb Dataset Preparation for Large-Scale Language Model Pretraining

This script downloads and tokenizes the FineWeb dataset, a massive web-crawled
corpus curated by HuggingFace for language model pretraining. FineWeb is derived
from CommonCrawl and filtered for quality.

Dataset Information:
    - Source: HuggingFace (HuggingFaceFW/fineweb and fineweb-edu)
    - Variants:
        * Classic FineWeb: General web text
        * FineWeb-Edu: Educational content (higher quality)
    - Sizes: 10B token sample, 100B token sample, or full dataset
    - Purpose: Large-scale pretraining of language models

Document Structure:
    Each document is a JSON object with fields:
    {
      "text": "Posted by mattsmith on 20th April 2012\nStraight from...",
      "id": "<urn:uuid:d853d453-196e-4488-a411-efc2b26c40d2>",
      "dump": "CC-MAIN-2013-20",
      "url": "http://example.com/article",
      "date": "2013-05-18T07:24:47Z",
      "file_path": "s3://commoncrawl/crawl-data/...",
      "language": "en",
      "language_score": 0.9185474514961243,
      "token_count": 594
    }

Processing Strategy:
    1. Streams dataset from HuggingFace (no full download needed)
    2. Tokenizes documents in parallel using multiprocessing
    3. Writes tokens to sharded binary files (default: 100M tokens per shard)
    4. First shard is validation, remaining shards are training
    5. Each document is prefixed with an end-of-text token

Output Files:
    - fineweb{size}/fineweb_val_000000.bin: Validation shard
    - fineweb{size}/fineweb_train_000001.bin: Training shard 1
    - fineweb{size}/fineweb_train_000002.bin: Training shard 2
    - ... (multiple shards)

Usage Examples:
    # Download FineWeb-Edu 10B sample with GPT-2 tokenizer
    $ python dev/data/fineweb.py -t edu -v 10B -m gpt-2

    # Download FineWeb-Edu 100B sample with LLaMA-3 tokenizer
    $ python dev/data/fineweb.py -t edu -v 100B -m llama-3

    # Custom shard size (50M tokens per shard)
    $ python dev/data/fineweb.py -t edu -v 10B -s 50000000

Performance:
    - 10B sample: ~30 minutes (depends on internet and CPU)
    - 100B sample: 3-5 hours (depends on internet and CPU)
    - Uses multiprocessing with (CPU_count - 2) workers

Data Format:
    Binary files with:
    - Header: 256 int32 values (metadata)
    - Data: uint16 (GPT-2) or uint32 (LLaMA-3) token IDs
    - Each document starts with an EOT token delimiter

References:
    Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb
    Paper: https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
"""
import os
import argparse
import multiprocessing as mp

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoTokenizer


from data_common import write_datafile
# ------------------------------------------

# Parse command-line arguments
parser = argparse.ArgumentParser(description="FineWeb and Edu-FineWeb dataset preprocessing")
parser.add_argument("-t", "--type", type=str, default="classic",
                    help="Fineweb type, edu|classic")
parser.add_argument("-v", "--version", type=str, default="10B",
                    help="Fineweb data sample size, 10B|100B")
parser.add_argument("-m", "--model_desc", type=str, default="gpt-2",
                    help="Model descriptor, gpt-2|llama-3")
parser.add_argument("-s", "--shard_size", type=int, default=10**8,
                    help="Size of each data shard in the output .bin files, in tokens")
args = parser.parse_args()

# Validate arguments
assert args.version in {"10B", "100B"}, "version must be one of: 10B, 100B"
assert args.type in {"edu", "classic"}, "type must be one of: edu, classic"

# Map (type, version) combinations to (local_directory, remote_dataset_name)
# The local directory stores cached data, remote name specifies HuggingFace dataset variant
directories = {
    ("classic", "10B"): ("fineweb10B", "sample-10BT"),
    ("classic", "100B"): ("fineweb100B", "sample-100BT"),
    ("edu", "10B"): ("edu_fineweb10B", "sample-10BT"),
    ("edu", "100B"): ("edu_fineweb100B", "sample-100BT")
}
local_dir, remote_name = directories[(args.type, args.version)]

# Create cache directory for this dataset variant
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Load the dataset from HuggingFace
# This streams the data, so it doesn't download everything at once
if args.type == "classic":
    fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")
    name = "fineweb"
elif args.type =="edu":
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
    name = "edu_fineweb"

def tokenize_llama(doc):
    """
    Tokenize a single document using LLaMA-3 tokenizer.

    This function is designed to be called by multiprocessing workers.
    Each call processes one document independently.

    Args:
        doc: Dictionary with "text" key containing the document text

    Returns:
        numpy array of uint32 token IDs, starting with EOT token

    Note:
        - EOT (end-of-text) token is prepended to mark document boundaries
        - LLaMA-3 uses uint32 tokens (vocab size ~128K)
        - Tokenizer is loaded per-call (multiprocessing requirement)
    """
    # Load LLaMA-3 tokenizer (each process needs its own instance)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False, verbose=False, split_special_tokens=True)
    eot = tokenizer.encode('')[0]  # EOT token ID (128000)

    # Tokenize: EOT token + document tokens
    tokens = [eot]  # Document delimiter
    tokens.extend(encode(doc["text"]))

    # Convert to numpy array with validation
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**32).all(), "token dictionary too large for uint32"
    tokens_np_uint = tokens_np.astype(np.uint32)

    return tokens_np_uint

def tokenize_gpt2(doc):
    """
    Tokenize a single document using GPT-2 tokenizer.

    This function is designed to be called by multiprocessing workers.
    Each call processes one document independently.

    Args:
        doc: Dictionary with "text" key containing the document text

    Returns:
        numpy array of uint16 token IDs, starting with EOT token

    Note:
        - EOT (end-of-text) token is prepended to mark document boundaries
        - GPT-2 uses uint16 tokens (vocab size ~50K)
        - Uses tiktoken for fast tokenization
    """
    # Load GPT-2 tokenizer (lightweight, fast to initialize)
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode_ordinary(s)
    eot = enc._special_tokens['<|endoftext|>']  # EOT token ID (50256)

    # Tokenize: EOT token + document tokens
    tokens = [eot]  # Document delimiter
    tokens.extend(encode(doc["text"]))

    # Convert to numpy array with validation
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint = tokens_np.astype(np.uint16)

    return tokens_np_uint

# Determine token data type based on model (uint16 for GPT-2, uint32 for LLaMA-3)
token_dtype = {
    "gpt-2": np.uint16,
    "llama-3": np.uint32
}[args.model_desc]

# Main processing loop: tokenize documents and write sharded binary files
# Each shard contains args.shard_size tokens (default: 100M tokens)
nprocs = max(1, os.cpu_count() - 2)  # Use most CPUs but leave 2 for system
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # Preallocate buffer for current shard (avoids repeated allocations)
    all_tokens_np = np.empty((args.shard_size,), dtype=token_dtype)
    token_count = 0  # Number of tokens in current shard
    progress_bar = None

    # Select tokenization function based on model type
    tokenize = lambda x: None
    if args.model_desc == "gpt-2":
        tokenize = tokenize_gpt2
    elif args.model_desc == "llama-3":
        tokenize = tokenize_llama
    else:
        raise ValueError(f"unknown model {args.model_desc}")

    # Process documents in parallel, streaming from HuggingFace
    # pool.imap yields tokenized documents as they complete
    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # Check if current document fits in current shard
        if token_count + len(tokens) < args.shard_size:
            # Document fits: append to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)

            # Create/update progress bar for current shard
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # Document doesn't fit: split across current and next shard
            split = "val" if shard_index == 0 else "train"  # First shard is validation
            filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")

            # Fill remainder of current shard
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]

            # Write completed shard to disk
            write_datafile(filename, all_tokens_np.tolist(), args.model_desc)
            shard_index += 1
            progress_bar = None

            # Start next shard with leftover tokens from current document
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # Write any remaining tokens as the final shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, (all_tokens_np[:token_count]).tolist(), args.model_desc)
