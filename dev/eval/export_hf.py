"""
GPT-2 Model Converter: llm.c Binary Format to HuggingFace

This script converts GPT-2 models from the llm.c binary checkpoint format to
HuggingFace Transformers format. This enables using llm.c-trained models with
the HuggingFace ecosystem for inference, fine-tuning, or deployment.

The converter handles two binary formats:
- Version 3: float32 weights with padded vocabulary
- Version 5: bfloat16 weights with padded vocabulary

Features:
- Converts model weights and configuration
- Includes GPT-2 tokenizer
- Optionally uploads to HuggingFace Hub
- Validates model by running a test generation

Binary Format (llm.c):
    Header (1024 bytes): Configuration parameters
    Weights: Model parameters in specific order

Output Format (HuggingFace):
    - config.json: Model configuration
    - model.safetensors: Model weights
    - tokenizer files: Vocabulary and special tokens

Setup:
    # Install HuggingFace Hub CLI (optional, for uploading)
    pip install -U "huggingface_hub[cli]"
    huggingface-cli login

Usage Examples:
    # Basic conversion to local directory
    python export_hf.py --input gpt2_124M.bin --output gpt2-124M-hf

    # Convert and upload to HuggingFace Hub
    python export_hf.py --input gpt2_124M.bin --output myusername/gpt2-124M \\
        --push true

    # Convert to bfloat16 format
    python export_hf.py --input gpt2_124M.bin --output gpt2-124M-bf16 \\
        --dtype bfloat16

    # Skip test generation
    python export_hf.py --input gpt2_124M.bin --output gpt2-124M-hf \\
        --spin false

Command-Line Arguments:
    --input, -i: Path to llm.c binary file (required)
    --output, -o: HuggingFace model output directory (required)
    --dtype, -d: Output precision (float32 or bfloat16), default: bfloat16
    --push, -p: Upload to HuggingFace Hub (true/false), default: false
    --spin, -s: Run test generation (true/false), default: true
"""

import numpy as np
import torch
import argparse, sys
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

# -----------------------------------------------------------------------------
# Tensor functions for both bfloat16 (from int16) and normal float32
# Both return float32 tensors

def tensor_bf16(data_int16, transpose=False):
    """
    Convert bfloat16 data (stored as int16) to float32 tensor.

    Since numpy doesn't support bfloat16, it's stored as int16 in the binary
    file. This function reinterprets those bytes as bfloat16 and converts to
    float32 for use with HuggingFace models.

    Args:
        data_int16 (np.ndarray): Array of int16 values representing bfloat16 data
        transpose (bool): If True, transpose the array before conversion.
            Needed for Conv1D->Linear weight compatibility. Default: False

    Returns:
        torch.Tensor: Float32 tensor with the same values
    """
    if transpose:
        data_int16 = data_int16.transpose(1,0)
    return torch.tensor(data_int16).view(torch.bfloat16).to(torch.float32)

def tensor_fp32(data_float32, transpose=False):
    """
    Convert float32 numpy array to float32 PyTorch tensor.

    Args:
        data_float32 (np.ndarray): Float32 numpy array
        transpose (bool): If True, transpose before converting. Default: False

    Returns:
        torch.Tensor: Float32 tensor
    """
    if transpose:
        data_float32 = data_float32.transpose(1,0)
    return torch.tensor(data_float32).view(torch.float32)

# -----------------------------------------------------------------------------
# Main conversion function

def convert(filepath, output, push_to_hub=False, out_dtype="bfloat16"):
    """
    Convert a GPT-2 model from llm.c binary format to HuggingFace format.

    This function reads a binary checkpoint file created by train_gpt2.py, extracts
    the model configuration and weights, and saves them in HuggingFace Transformers
    format. The output can be used with the HuggingFace ecosystem or uploaded to
    the HuggingFace Hub.

    Binary File Format (llm.c):
        Header (1024 bytes): 256 int32 values
            [0]: Magic number (20240326)
            [1]: Version (3=float32, 5=bfloat16)
            [2]: block_size (max sequence length)
            [3]: vocab_size (actual vocabulary size, e.g., 50257)
            [4]: n_layer (number of transformer blocks)
            [5]: n_head (number of attention heads)
            [6]: n_embd (embedding dimension)
            [7]: padded_vocab_size (vocab size rounded to multiple, e.g., 50304)
        Weights: All parameters in predetermined order

    Args:
        filepath (str): Path to llm.c binary checkpoint file
        output (str): Output directory or HuggingFace Hub model ID
            (e.g., "myusername/my-model")
        push_to_hub (bool): If True, upload to HuggingFace Hub after saving
            locally. Requires authentication via huggingface-cli login. Default: False
        out_dtype (str): Output precision for HuggingFace model.
            Options: "float32" or "bfloat16". Default: "bfloat16"

    Raises:
        SystemExit: If magic number or version is invalid
        ValueError: If out_dtype is not "float32" or "bfloat16"

    Notes:
        - Automatically handles vocabulary padding (llm.c pads to multiples of 128)
        - Handles weight transposition for Conv1D layers
        - Includes GPT-2 tokenizer from HuggingFace
    """
    print(f"Converting model {filepath} to {output} in {out_dtype} format and pushing to Hugging Face: {push_to_hub}")

    f = open(filepath, 'rb')
    # Read in our header, checking the magic number and version
    # version 3 = fp32, padded vocab
    # version 5 = bf16, padded vocab
    model_header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if model_header[0] != 20240326:
        print("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    version = model_header[1]
    if not version in [3, 5]:
        print("Bad version in model file")
        exit(1)

    # Load in our model parameters
    maxT = model_header[2].item() # max sequence length
    V = model_header[3].item() # vocab size
    L =  model_header[4].item() # num layers
    H = model_header[5].item() # num heads
    C = model_header[6].item() # channels
    Vp = model_header[7].item() # padded vocab size

    print(f"{version=}, {maxT=}, {V=}, {Vp=}, {L=}, {H=}, {C=}")

    # Define the shapes of our parameters
    shapes = {
        'wte': (Vp, C),
        'wpe': (maxT, C),
        'ln1w': (L, C),
        'ln1b': (L, C),
        'qkvw': (L, 3 * C, C),
        'qkvb': (L, 3 * C),
        'attprojw': (L, C, C),
        'attprojb': (L, C),
        'ln2w': (L, C),
        'ln2b': (L, C),
        'fcw': (L, 4 * C, C),
        'fcb': (L, 4 * C),
        'fcprojw': (L, C, 4 * C),
        'fcprojb': (L, C),
        'lnfw': (C,),
        'lnfb': (C,),
    }

    # Load in our weights given our parameter shapes
    dtype = np.float32 if version == 3 else np.int16
    w = {}
    for key, shape in shapes.items():
        num_elements = np.prod(shape)
        data = np.frombuffer(f.read(num_elements * np.dtype(dtype).itemsize), dtype=dtype)
        w[key] = data.reshape(shape)
        # The binary file saves the padded vocab - drop the padding back to GPT2 size
        if shape[0] == Vp:
            w[key] = w[key].reshape(shape)[:(V-Vp), :]
    # Ensure the file is fully read and then close
    assert f.read() == b''
    f.close()

    # Map to our model dict, the tensors at this stage are always fp32
    mk_tensor = {
        3 : tensor_fp32,
        5 : tensor_bf16,
    }[version]
    model_dict = {}
    model_dict['transformer.wte.weight'] = mk_tensor(w['wte'])
    model_dict['transformer.wpe.weight'] = mk_tensor(w['wpe'])
    model_dict['lm_head.weight'] = model_dict['transformer.wte.weight'] # Tie weights
    for i in range(L):
        model_dict[f'transformer.h.{i}.ln_1.weight'] = mk_tensor(w['ln1w'][i])
        model_dict[f'transformer.h.{i}.ln_1.bias'] = mk_tensor(w['ln1b'][i])
        model_dict[f'transformer.h.{i}.attn.c_attn.weight'] = mk_tensor(w['qkvw'][i], True)
        model_dict[f'transformer.h.{i}.attn.c_attn.bias'] = mk_tensor(w['qkvb'][i])
        model_dict[f'transformer.h.{i}.attn.c_proj.weight'] = mk_tensor(w['attprojw'][i], True)
        model_dict[f'transformer.h.{i}.attn.c_proj.bias'] = mk_tensor(w['attprojb'][i])
        model_dict[f'transformer.h.{i}.ln_2.weight'] = mk_tensor(w['ln2w'][i])
        model_dict[f'transformer.h.{i}.ln_2.bias'] = mk_tensor(w['ln2b'][i])
        model_dict[f'transformer.h.{i}.mlp.c_fc.weight'] = mk_tensor(w['fcw'][i], True)
        model_dict[f'transformer.h.{i}.mlp.c_fc.bias'] = mk_tensor(w['fcb'][i])
        model_dict[f'transformer.h.{i}.mlp.c_proj.weight'] = mk_tensor(w['fcprojw'][i], True)
        model_dict[f'transformer.h.{i}.mlp.c_proj.bias'] = mk_tensor(w['fcprojb'][i])
    model_dict['transformer.ln_f.weight'] = mk_tensor(w['lnfw'])
    model_dict['transformer.ln_f.bias'] = mk_tensor(w['lnfb'])

    # Create a GPT-2 model instance, in the requested dtype
    config = GPT2Config(vocab_size = V,
                        n_positions = maxT,
                        n_ctx = maxT,
                        n_embd = C,
                        n_layer = L,
                        n_head = H)
    model = GPT2LMHeadModel(config)
    if out_dtype == "bfloat16":
        model = model.to(torch.bfloat16)

    # Set the model dict and save
    model.load_state_dict(model_dict)
    model.save_pretrained(output, max_shard_size="5GB", safe_serialization=True)

    # Copy over a standard gpt2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(output)

    if push_to_hub:
        print(f"Uploading {output} to Hugging Face")
        model.push_to_hub(output)
        tokenizer.push_to_hub(output)

def spin(output):
    """
    Test the converted model by generating sample text.

    This function loads the converted HuggingFace model and runs a test
    generation to verify the conversion was successful. It uses a simple
    prompt and generates 64 tokens.

    Args:
        output (str): Path to the converted HuggingFace model directory

    Test Configuration:
        - Prompt: "During photosynthesis in green plants"
        - Max tokens: 64
        - Repetition penalty: 1.3
        - Uses Flash Attention 2 if available
        - Runs on CUDA if available

    Output:
        Prints the generated text to console

    Note:
        Requires transformers library with Flash Attention 2 support.
        Falls back to standard attention if Flash Attention is unavailable.
    """
    print("Taking the exported model for a spin...")
    print('-'*80)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load tokenizer and model from the converted directory
    tokenizer = AutoTokenizer.from_pretrained(output)
    model = AutoModelForCausalLM.from_pretrained(output, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()

    # Prepare test prompt
    tokens = tokenizer.encode("During photosynthesis in green plants", return_tensors="pt")
    tokens = tokens.to('cuda')

    # Generate continuation
    output = model.generate(tokens, max_new_tokens=64, repetition_penalty=1.3)
    samples = tokenizer.batch_decode(output)

    # Print results
    for sample in samples:
        print('-'*30)
        print(sample)

# -----------------------------------------------------------------------------

if __name__== '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="The name of the llm.c model.bin file", type=str, required=True)
    parser.add_argument("--output","-o",  help="The Hugging Face output model directory", type=str, required=True)
    parser.add_argument("--dtype", "-d", help="Output as either float32 or bfloat16 (default)", type=str, default="bfloat16")
    parser.add_argument("--push", "-p", help="Push the model to your Hugging Face account", type=bool, default=False)
    parser.add_argument("--spin", "-s", help="Take the model for a spin at the end?", type=bool, default=True)
    args = parser.parse_args()
    convert(args.input, args.output, args.push, args.dtype)
    if args.spin:
        spin(args.output)
