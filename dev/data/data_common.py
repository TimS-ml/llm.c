"""
Common Utilities for Dataset Processing in llm.c

This module provides shared utility functions for downloading, tokenizing, and saving
datasets used in the llm.c project. It includes functions for:
- Downloading files from URLs with progress bars
- Writing tokenized data in binary formats compatible with C code
- Writing evaluation datasets for multiple-choice benchmarks

The binary file formats are designed for efficient loading in C/CUDA training code.

Usage Example:
    from data_common import download_file, write_datafile

    # Download a dataset
    download_file("https://example.com/data.txt", "local_data.txt")

    # Save tokenized data
    tokens = [123, 456, 789]
    write_datafile("output.bin", tokens, model_desc="gpt-2")
"""

import requests
from tqdm import tqdm
import numpy as np


def download_file(url: str, fname: str, chunk_size=1024):
    """
    Download a file from a URL with a progress bar.

    This function streams the download to avoid loading large files entirely into memory.
    It displays a progress bar using tqdm to show download progress.

    Args:
        url: The URL to download from
        fname: The local filename to save to
        chunk_size: Number of bytes to download at a time (default: 1024)

    Returns:
        None

    Example:
        >>> download_file("https://example.com/data.txt", "data.txt")
        data.txt: 100%|██████████| 1.5M/1.5M [00:02<00:00, 512kB/s]
    """
    # Stream the response to avoid loading entire file into memory
    resp = requests.get(url, stream=True)

    # Get the total file size from headers (if available)
    total = int(resp.headers.get("content-length", 0))

    # Open file and progress bar simultaneously using context managers
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",        # Display as bytes
        unit_scale=True,  # Auto-scale to KB, MB, GB
        unit_divisor=1024,
    ) as bar:
        # Download in chunks and update progress bar
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


# Binary file format specifications for different model types
# Each model type has a unique magic number for file validation,
# a version number for format tracking, and a token data type
HEADERS_INFO = {
    "gpt-2": {
        "magic": 20240520,      # Magic number to identify GPT-2 format files
        "version": 1,           # Format version number
        "token_dtype": np.uint16,  # GPT-2 uses 16-bit tokens (vocab size < 65536)
    },
    "llama-3": {
        "magic": 20240801,      # Magic number to identify LLaMA-3 format files
        "version": 7,           # Format version number
        "token_dtype": np.uint32,  # LLaMA-3 uses 32-bit tokens (larger vocab)
    },
}

def write_datafile(filename, toks, model_desc="gpt-2"):
    """
    Save tokenized data as a binary file for efficient C/CUDA loading.

    This function writes tokens to a binary file with a standardized header format
    that can be read efficiently by C code. The file structure is:
    - Header: 256 int32 values (1024 bytes total)
        - header[0]: magic number (identifies file format)
        - header[1]: version number
        - header[2]: number of tokens
        - header[3-255]: reserved for future use (zeros)
    - Data: Token IDs as uint16 (GPT-2) or uint32 (LLaMA-3)

    Args:
        filename: Path where the binary file will be written
        toks: List or array of token IDs to save
        model_desc: Model type, either "gpt-2" or "llama-3" (default: "gpt-2")

    Returns:
        None

    Raises:
        AssertionError: If token count exceeds 2^31 or model_desc is invalid

    Example:
        >>> tokens = [123, 456, 789, 101112]
        >>> write_datafile("train.bin", tokens, model_desc="gpt-2")
        writing 4 tokens to train.bin (1,032 bytes) in the gpt-2 format
    """
    # Validate token count (max ~2.1 billion tokens)
    assert len(toks) < 2**31, "token count too large"
    assert model_desc in ["gpt-2", "llama-3"], f"unknown model descriptor {model_desc}"

    # Get format specifications for this model type
    info = HEADERS_INFO[model_desc]

    # Construct the header (256 int32 values = 1024 bytes)
    header = np.zeros(256, dtype=np.int32)
    header[0] = info["magic"]      # Magic number for format identification
    header[1] = info["version"]    # Version for backward compatibility
    header[2] = len(toks)          # Number of tokens following header

    # Convert tokens to appropriate numpy dtype (uint16 for GPT-2, uint32 for LLaMA-3)
    toks_np = np.array(toks, dtype=info["token_dtype"])

    # Calculate total file size: header (256*4 bytes) + tokens
    num_bytes = (256 * 4) + (len(toks) * toks_np.itemsize)
    print(f"writing {len(toks):,} tokens to {filename} ({num_bytes:,} bytes) in the {model_desc} format")

    # Write header and tokens to binary file
    with open(filename, "wb") as f:
        f.write(header.tobytes())   # Write 1024-byte header
        f.write(toks_np.tobytes())  # Write token data

def write_evalfile(filename, datas):
    """
    Save evaluation data for multiple-choice benchmarks as a binary file.

    This function is used for multiple-choice evaluation datasets like HellaSwag and MMLU.
    Each example consists of a context (question/prompt) and multiple completion choices,
    with one correct answer. The binary format enables efficient C/CUDA evaluation.

    File Structure:
        - Header: 256 int32 values (1024 bytes)
            - header[0]: magic number (20240522)
            - header[1]: version (1)
            - header[2]: number of examples
            - header[3]: longest example size in bytes
            - header[4-255]: reserved (zeros)
        - Examples: Stream of uint16 values, each example contains:
            - START_EXAMPLE: delimiter (65535 = 2^16-1)
            - EXAMPLE_BYTES: size of this example in bytes
            - EXAMPLE_INDEX: index of this example
            - LABEL: index of correct completion (0-3)
            - NUM_COMPLETIONS: number of choices (typically 4)
            - Context: NUM followed by NUM token IDs
            - Completions: NUM followed by NUM token IDs (repeated for each choice)

    Args:
        filename: Path where the binary file will be written
        datas: List of dictionaries, each containing:
            - "label": Index of correct completion (int)
            - "ctx_tokens": List of context token IDs
            - "ending_tokens": List of lists, each containing completion token IDs

    Returns:
        None

    Raises:
        AssertionError: If validation fails (too many examples, bad tokens, etc.)

    Example:
        >>> eval_data = [{
        ...     "label": 2,
        ...     "ctx_tokens": [123, 456],
        ...     "ending_tokens": [[789], [101], [112], [131]]
        ... }]
        >>> write_evalfile("hellaswag_val.bin", eval_data)
        writing 1 examples to hellaswag_val.bin

    Note:
        Token IDs must be < 65535 (2^16-1) since the delimiter uses that value.
        For HellaSwag validation: 10,042 examples result in ~3.6MB file.
    """
    # Construct the header (256 int32 values)
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240522      # Magic number for eval file format
    header[1] = 1             # Version number
    header[2] = len(datas)    # Total number of examples
    header[3] = 0             # Longest example size (filled in later)

    # Build the data stream for all examples
    longest_example_bytes = 0  # Track the longest example for C buffer allocation
    full_stream = []           # Complete stream of uint16 values

    # Validate we don't have too many examples (max 65535)
    assert len(datas) < 2**16, "too many examples?"

    # Process each example
    for idx, data in enumerate(datas):
        stream = []

        # Example header fields
        stream.append(2**16-1)        # START_EXAMPLE delimiter (65535)
        stream.append(0)              # EXAMPLE_BYTES (filled in later)
        stream.append(idx)            # EXAMPLE_INDEX
        stream.append(data["label"])  # LABEL (correct answer index)

        # Get completion choices
        ending_tokens = data["ending_tokens"]
        assert len(ending_tokens) == 4, "expected 4 completions for now? can relax later"
        stream.append(len(ending_tokens))  # NUM_COMPLETIONS

        # Add context (shared across all completions)
        ctx_tokens = data["ctx_tokens"]
        # Ensure tokens don't conflict with delimiter value (65535)
        assert all(0 <= t < 2**16-1 for t in ctx_tokens), "bad context token"
        stream.append(len(ctx_tokens))  # Number of context tokens
        stream.extend(ctx_tokens)        # The actual context tokens

        # Add each completion option
        for end_tokens in ending_tokens:
            # Ensure tokens don't conflict with delimiter value
            assert all(0 <= t < 2**16-1 for t in end_tokens), "bad completion token"
            stream.append(len(end_tokens))  # Number of tokens in this completion
            stream.extend(end_tokens)        # The actual completion tokens

        # Calculate example size and update the EXAMPLE_BYTES field
        nbytes = len(stream) * 2  # 2 bytes per uint16
        assert nbytes < 2**16, "example too large?"
        stream[1] = nbytes  # Fill in the EXAMPLE_BYTES field (index 1)

        # Track longest example for header
        longest_example_bytes = max(longest_example_bytes, nbytes)

        # Add this example to the full stream
        full_stream.extend(stream)

    # Convert to numpy array for efficient writing
    stream_np = np.array(full_stream, dtype=np.uint16)

    # Update header with longest example size (useful for C buffer allocation)
    assert 0 < longest_example_bytes < 2**16, f"bad longest_example"
    header[3] = longest_example_bytes

    # Write to file (e.g., HellaSwag val: 10,042 examples, ~3.6MB)
    print(f"writing {len(datas):,} examples to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())    # Write header (1024 bytes)
        f.write(stream_np.tobytes()) # Write all examples as uint16 stream
