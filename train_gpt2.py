"""
GPT-2 Training and Inference Reference Implementation

This module provides a complete PyTorch implementation of the GPT-2 language model,
including training, inference, and model export functionality. It's designed as a
reference implementation for the llm.c project, which provides a C implementation
of GPT-2 training and inference.

The implementation closely follows the original GPT-2 architecture with some
optimizations for modern hardware:
- Support for mixed precision training (float32, float16, bfloat16)
- Flash Attention for efficient self-attention computation
- Distributed Data Parallel (DDP) training across multiple GPUs
- Model compilation with torch.compile for improved performance
- Gradient accumulation for large effective batch sizes

Architecture Overview:
    The GPT-2 model consists of:
    - Token embeddings (wte): Maps input tokens to embedding vectors
    - Position embeddings (wpe): Adds positional information to embeddings
    - Transformer blocks: Stack of L identical blocks, each containing:
        * Layer normalization (ln_1)
        * Multi-head causal self-attention (CausalSelfAttention)
        * Layer normalization (ln_2)
        * Feed-forward MLP with GELU activation
    - Final layer normalization (ln_f)
    - Language model head (lm_head): Projects to vocabulary size for next-token prediction

References:
    1) The official GPT-2 TensorFlow implementation released by OpenAI:
       https://github.com/openai/gpt-2/blob/master/src/model.py
    2) HuggingFace Transformers PyTorch implementation:
       https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

Example Usage:
    # Train on a single GPU with bfloat16, TensorCores, and Flash Attention:
    python train_gpt2.py --write_tensors=0 --num_iterations=50 --sequence_length=1024 \\
        --compile=1 --tensorcores=1 --dtype=bfloat16 --flash=1

    # Distributed training on 4 GPUs:
    torchrun --standalone --nproc_per_node=4 train_gpt2.py --write_tensors=0 \\
        --num_iterations=50 --sequence_length=1024 --compile=1 --tensorcores=1 \\
        --dtype=bfloat16

    # Train from pretrained GPT-2 checkpoint:
    python train_gpt2.py --model=gpt2 --input_bin=data/train.bin \\
        --input_val_bin=data/val.bin --batch_size=8 --sequence_length=512

    # Export model weights to binary format for C:
    python train_gpt2.py --write_tensors=1 --num_iterations=1 --model=gpt2

Key Command-Line Arguments:
    --model: Model size (gpt2, gpt2-medium, gpt2-large, gpt2-xl, d12, d24, d36, d48)
    --input_bin: Path to training data in binary format
    --batch_size: Batch size per GPU (default: 4)
    --sequence_length: Sequence length for training (default: 64, max: 1024)
    --total_batch_size: Total batch size across all GPUs and accumulation steps
    --num_iterations: Number of training iterations
    --learning_rate: Learning rate (default: 1e-4)
    --dtype: Data type for training (float32, float16, bfloat16)
    --compile: Enable torch.compile for faster training (0 or 1)
    --flash: Enable Flash Attention (0 or 1)
    --tensorcores: Enable TensorCore operations (0 or 1)
    --write_tensors: Write model weights and debug state to disk (0 or 1)
"""

import os
import math
import glob
import struct
import inspect
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.distributed as dist

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class NewGELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    This is the exact implementation used by OpenAI in GPT-2. Note that there are
    multiple approximations of GELU in the literature. This version uses a tanh
    approximation rather than the exact erf-based formulation for better performance.

    The GELU activation is defined as:
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    This is a smooth, non-monotonic function that weights inputs by their value
    rather than their sign like ReLU. It has shown strong performance in transformer
    models.

    Reference:
        Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)", 2016
        https://arxiv.org/abs/1606.08415
    """
    def forward(self, input):
        """
        Apply GELU activation function.

        Args:
            input (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU activation applied, same shape as input.
        """
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

# Global flag to toggle flash-attention (set at runtime via command-line args)
FLASH = 0

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention mechanism for GPT-2.

    This implements the core attention mechanism that allows the model to attend to
    previous positions in the sequence. The "causal" aspect means that each position
    can only attend to itself and previous positions, not future positions, which is
    enforced through an attention mask.

    The attention mechanism works as follows:
    1. Project input to Query (Q), Key (K), and Value (V) matrices
    2. Split into multiple attention heads for parallel processing
    3. Compute attention scores: softmax(Q @ K^T / sqrt(d_k))
    4. Apply causal mask to prevent attending to future positions
    5. Compute weighted sum: attention_scores @ V
    6. Concatenate heads and project back to embedding dimension

    Attributes:
        c_attn (nn.Linear): Combined linear projection for Q, K, V (input -> 3 * n_embd)
        c_proj (nn.Linear): Output projection (n_embd -> n_embd)
        n_head (int): Number of attention heads
        n_embd (int): Embedding dimension
        bias (torch.Tensor): Lower triangular causal mask (registered as buffer)

    Implementation Details:
        - Uses a single linear layer (c_attn) to compute Q, K, V for efficiency
        - Supports both manual attention computation and Flash Attention
        - The output projection (c_proj) uses scaled initialization (GPT-2 paper)
    """

    def __init__(self, config):
        """
        Initialize the causal self-attention layer.

        Args:
            config (GPTConfig): Model configuration object containing:
                - n_embd: Embedding dimension
                - n_head: Number of attention heads
                - block_size: Maximum sequence length

        Raises:
            AssertionError: If n_embd is not divisible by n_head
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dim must be divisible by number of heads"

        # Combined projection for Query, Key, Value for all heads
        # This is more efficient than three separate projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection - projects concatenated head outputs back to n_embd
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Flag for special initialization (scaled by 1/sqrt(2*n_layer) per GPT-2 paper)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

        # Store dimensions for reshaping
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Create causal mask: lower triangular matrix ensures position i can only attend to positions <= i
        # Note: Named 'bias' to match OpenAI/HuggingFace naming, but it's actually an attention mask
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        """
        Apply multi-head causal self-attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C) where:
                B = batch size
                T = sequence length
                C = embedding dimension (n_embd)

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C) after attention and projection

        Implementation Notes:
            When FLASH=0 (manual attention):
                - Computes full (T, T) attention matrix
                - Uses O(T^2) memory and computation
                - Good for debugging and small sequences

            When FLASH=1 (Flash Attention):
                - Uses PyTorch's optimized scaled_dot_product_attention
                - More memory efficient with O(T) memory usage
                - Faster on modern GPUs with appropriate hardware support
        """
        B, T, C = x.size()  # Batch size, sequence length, embedding dimensionality (n_embd)

        # Calculate query, key, values for all heads in a single matrix multiplication
        # Input: (B, T, C) -> Output: (B, T, 3*C)
        qkv = self.c_attn(x)

        # Split into separate Q, K, V tensors, each of shape (B, T, C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape and transpose to split across attention heads
        # (B, T, C) -> (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        # where head_size = C // n_head
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        if FLASH:
            # Flash Attention: memory-efficient attention using PyTorch's optimized implementation
            # Automatically handles causal masking when is_causal=True
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Manual attention implementation - useful for debugging and understanding
            # Step 1: Compute attention scores Q @ K^T, scaled by sqrt(head_size)
            # Shape: (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # Step 2: Apply causal mask - positions can only attend to themselves and earlier positions
            # Set masked positions to -inf so they become 0 after softmax
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

            # Step 3: Normalize attention scores with softmax
            att = F.softmax(att, dim=-1)

            # Step 4: Compute weighted sum of values
            # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
            y = att @ v

        # Re-assemble all head outputs side by side
        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network) for GPT-2.

    This is the position-wise feed-forward network that appears after the attention
    layer in each transformer block. It applies two linear transformations with a
    GELU activation in between:

        MLP(x) = proj(GELU(fc(x)))

    The intermediate dimension is 4x the embedding dimension, which allows the
    model to learn more complex transformations. This expansion and contraction
    pattern is standard in transformer architectures.

    Attributes:
        c_fc (nn.Linear): First linear layer, expands from n_embd to 4*n_embd
        gelu (NewGELU): GELU activation function
        c_proj (nn.Linear): Second linear layer, projects from 4*n_embd back to n_embd

    Architecture:
        Input: (B, T, C)
        -> Linear: (B, T, 4*C)
        -> GELU: (B, T, 4*C)
        -> Linear: (B, T, C)
        Output: (B, T, C)
    """

    def __init__(self, config):
        """
        Initialize the MLP layer.

        Args:
            config (GPTConfig): Model configuration object containing n_embd
        """
        super().__init__()
        # Expand to 4x the embedding dimension (standard transformer ratio)
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = NewGELU()
        # Project back to original embedding dimension
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        # Flag for scaled initialization (per GPT-2 paper)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x):
        """
        Apply feed-forward transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C)
        """
        x = self.c_fc(x)      # Expand: (B, T, C) -> (B, T, 4*C)
        x = self.gelu(x)      # Non-linearity
        x = self.c_proj(x)    # Project back: (B, T, 4*C) -> (B, T, C)
        return x

class Block(nn.Module):
    """
    A single Transformer block for GPT-2.

    This implements one layer of the GPT-2 transformer, which consists of:
    1. Layer normalization + Multi-head self-attention + Residual connection
    2. Layer normalization + Feed-forward MLP + Residual connection

    This follows the "Pre-LN" architecture where layer normalization is applied
    before the attention and MLP sublayers, rather than after. This has been
    shown to improve training stability.

    The computation flow is:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    Attributes:
        ln_1 (nn.LayerNorm): Layer normalization before attention
        attn (CausalSelfAttention): Multi-head self-attention layer
        ln_2 (nn.LayerNorm): Layer normalization before MLP
        mlp (MLP): Feed-forward network

    Note:
        The residual connections (x + sublayer(x)) allow gradients to flow
        directly through the network, improving training of deep models.
    """

    def __init__(self, config):
        """
        Initialize a transformer block.

        Args:
            config (GPTConfig): Model configuration object
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        """
        Apply transformer block computation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)

        Returns:
            torch.Tensor: Output tensor of shape (B, T, C)
        """
        # First sublayer: Self-attention with residual connection
        # Pre-LN: normalize first, then apply attention, then add residual
        x = x + self.attn(self.ln_1(x))

        # Second sublayer: Feed-forward with residual connection
        x = x + self.mlp(self.ln_2(x))

        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    """
    Configuration class for GPT-2 model architecture.

    This dataclass holds all the hyperparameters that define the model architecture.
    Default values correspond to GPT-2 small (124M parameters).

    Attributes:
        block_size (int): Maximum sequence length the model can handle. Default: 1024
        vocab_size (int): Size of the vocabulary (number of unique tokens). Default: 50257
            This is the standard GPT-2 vocabulary size from the BPE tokenizer.
        n_layer (int): Number of transformer blocks. Default: 12
        n_head (int): Number of attention heads in each layer. Default: 12
        n_embd (int): Embedding dimension (also called d_model or hidden size). Default: 768
            This must be divisible by n_head.

    Model Size Presets:
        - GPT-2 Small (124M params): n_layer=12, n_head=12, n_embd=768
        - GPT-2 Medium (350M params): n_layer=24, n_head=16, n_embd=1024
        - GPT-2 Large (774M params): n_layer=36, n_head=20, n_embd=1280
        - GPT-2 XL (1558M params): n_layer=48, n_head=25, n_embd=1600
    """
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    """
    GPT-2 Language Model Implementation.

    This is the complete GPT-2 model, implementing the autoregressive language modeling
    architecture introduced in "Language Models are Unsupervised Multitask Learners"
    (Radford et al., 2019).

    The model architecture consists of:
        1. Token embeddings (wte): Converts token IDs to embedding vectors
        2. Position embeddings (wpe): Adds positional information to embeddings
        3. Transformer blocks (h): Stack of L transformer layers
        4. Final layer normalization (ln_f): Normalizes before output projection
        5. Language model head (lm_head): Projects to vocabulary for next-token prediction

    Key Features:
        - Causal (autoregressive) attention masks ensure proper left-to-right generation
        - Weight tying between token embeddings and output projection
        - Residual connections and layer normalization for training stability
        - Scaled initialization for deep networks (GPT-2 paper methodology)

    Attributes:
        config (GPTConfig): Model configuration
        transformer (nn.ModuleDict): Container for the core transformer components
            - wte: Token embedding table
            - wpe: Position embedding table
            - h: List of transformer blocks
            - ln_f: Final layer normalization
        lm_head (nn.Linear): Output projection to vocabulary (shares weights with wte)
        init_rng (torch.Generator): RNG for reproducible weight initialization

    Example:
        >>> config = GPTConfig(n_layer=12, n_head=12, n_embd=768)
        >>> model = GPT(config)
        >>> x = torch.randint(0, config.vocab_size, (2, 512))  # (batch, seq_len)
        >>> logits, loss = model(x, targets=x)
    """

    def __init__(self, config):
        """
        Initialize the GPT-2 model.

        Args:
            config (GPTConfig): Model configuration specifying architecture parameters
        """
        super().__init__()
        self.config = config

        # Build the transformer architecture
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),  # Position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer blocks
            ln_f = nn.LayerNorm(config.n_embd),  # Final layer norm
        ))

        # Language model head: projects hidden states to vocabulary logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Don't initialize lm_head separately - we'll tie weights with wte
        self.lm_head.LLMC_SKIP_INIT = 1

        # Weight tying: share parameters between input and output embeddings
        # This reduces parameters and often improves performance
        # Reference: https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights using a fixed random seed for reproducibility
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights for model parameters.

        This implements the initialization scheme from the GPT-2 paper:
        - Normal distribution with std=0.02 for most weights
        - Scaled initialization for residual projections: std=0.02/sqrt(2*n_layer)
          This scaling prevents activation magnitudes from growing with depth

        Args:
            module (nn.Module): PyTorch module to initialize

        Note:
            This is called via self.apply() during __init__, which recursively
            applies this function to all submodules.
        """
        if isinstance(module, nn.Linear):
            # Determine standard deviation for initialization
            # Residual projections get scaled down to account for accumulation across layers
            std = 0.02 if not hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') else 0.02/math.sqrt(2 * self.config.n_layer)

            # Skip initializing lm_head since it shares weights with wte
            # (wte is initialized below as an Embedding)
            if not hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)

            # Initialize biases to zero if they exist
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            # Initialize embeddings from normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(self, idx, targets=None, return_logits=True):
        """
        Forward pass through the GPT-2 model.

        Args:
            idx (torch.Tensor): Input token indices of shape (B, T) where:
                B = batch size, T = sequence length
                Values should be in range [0, vocab_size)
            targets (torch.Tensor, optional): Target token indices for computing loss,
                shape (B, T). If provided, cross-entropy loss is computed. If None,
                only logits are computed (inference mode). Default: None
            return_logits (bool): Whether to return logits. Setting to False saves
                memory during training when only loss is needed. Default: True

        Returns:
            tuple: (logits, loss) where:
                - logits (torch.Tensor or None): Predicted next-token logits
                    Training mode (targets provided): shape (B, T, vocab_size)
                    Inference mode (no targets): shape (B, 1, vocab_size) - only last position
                    If return_logits=False: None
                - loss (torch.Tensor or None): Cross-entropy loss if targets provided,
                    else None

        Raises:
            AssertionError: If sequence length exceeds model's block_size

        Example:
            >>> model = GPT(config)
            >>> input_ids = torch.randint(0, config.vocab_size, (4, 128))
            >>> logits, loss = model(input_ids, targets=input_ids)  # Training
            >>> logits, _ = model(input_ids[:, :10])  # Inference
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Create position indices for the sequence
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t,)

        # Forward through the transformer
        # 1. Get token embeddings: (B, T) -> (B, T, C)
        tok_emb = self.transformer.wte(idx)

        # 2. Get position embeddings: (T,) -> (T, C)
        #    These are broadcasted and added to token embeddings
        pos_emb = self.transformer.wpe(pos)

        # 3. Combine token and position embeddings
        x = tok_emb + pos_emb

        # 4. Pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # 5. Apply final layer normalization
        x = self.transformer.ln_f(x)

        # 6. Compute logits and loss
        if targets is not None:
            # Training mode: compute logits for all positions and calculate loss
            logits = self.lm_head(x)  # (B, T, vocab_size)
            # Flatten for cross-entropy: (B*T, vocab_size) and (B*T,)
            # ignore_index=-1 allows masking certain positions
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference mode: only compute logits for the last position
            # This is a minor optimization - we only need the next token prediction
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            # Note: using list [-1] to preserve the time dimension
            loss = None

        # Optionally skip returning logits to save memory (e.g., when using DDP)
        if not return_logits:
            logits = None

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Load pretrained GPT-2 model weights from HuggingFace.

        This class method creates a GPT model instance and initializes it with
        pretrained weights from OpenAI's GPT-2 models, downloaded via HuggingFace
        Transformers library.

        The method handles weight format differences between HuggingFace's implementation
        and this implementation (e.g., Conv1D vs Linear layers) by transposing weights
        where necessary.

        Args:
            model_type (str): Size of GPT-2 model to load. Must be one of:
                - 'gpt2': 124M parameters (12 layers, 12 heads, 768 dim)
                - 'gpt2-medium': 350M parameters (24 layers, 16 heads, 1024 dim)
                - 'gpt2-large': 774M parameters (36 layers, 20 heads, 1280 dim)
                - 'gpt2-xl': 1558M parameters (48 layers, 25 heads, 1600 dim)

        Returns:
            GPT: Model instance with pretrained weights loaded

        Raises:
            AssertionError: If model_type is not one of the supported sizes
            AssertionError: If state dict keys don't match between HF and our implementation

        Example:
            >>> model = GPT.from_pretrained('gpt2')
            >>> # Model is now ready for inference or fine-tuning

        Note:
            This requires the `transformers` package to be installed.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # Map model_type to architecture configuration parameters
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        # All GPT-2 models use the same vocab size and context length
        config_args['vocab_size'] = 50257  # BPE vocabulary size
        config_args['block_size'] = 1024   # Maximum sequence length

        # Create our GPT model with the specified configuration
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # Filter out the attention bias (causal mask) - it's a buffer, not a parameter
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # Download and load the HuggingFace pretrained model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Prepare HuggingFace state dict for comparison
        # Filter out attention masks/biases which are buffers, not parameters
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # List of weights that need to be transposed
        # HuggingFace uses Conv1D layers, we use Linear layers
        # Conv1D stores weights as (in_features, out_features)
        # Linear stores weights as (out_features, in_features)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Verify that the number of parameters matches
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        # Copy weights from HuggingFace model to our model
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Transpose Conv1D weights to match Linear layer format
                # Verify shapes are compatible (reversed dimensions)
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"Shape mismatch for {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Direct copy for non-transposed parameters (embeddings, biases, layer norms)
                assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch for {k}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
        """
        Configure the optimizer for training.

        This creates an AdamW optimizer with weight decay applied selectively:
        - 2D+ parameters (weights in Linear/Embedding layers) get weight decay
        - 1D parameters (biases, layer norm parameters) don't get weight decay

        This selective decay is a best practice in transformer training, as applying
        weight decay to biases and normalization parameters can hurt performance.

        Args:
            weight_decay (float): Weight decay (L2 penalty) coefficient for applicable parameters
            learning_rate (float): Initial learning rate for the optimizer
            betas (tuple): Coefficients for computing running averages of gradient and
                its square in Adam (typically (0.9, 0.95) or (0.9, 0.999))
            device_type (str): Type of device ('cuda' or 'cpu'), affects fused optimizer availability
            zero_stage (int): ZeRO optimization stage (0=disabled, 1=optimizer state partitioning,
                2=optimizer+gradients, 3=optimizer+gradients+parameters). Stage 1 uses
                ZeroRedundancyOptimizer for distributed training.

        Returns:
            torch.optim.Optimizer: Configured optimizer (AdamW or ZeroRedundancyOptimizer)

        Implementation Notes:
            - Uses fused AdamW kernel when available on CUDA for better performance
            - Separates parameters into decay/no-decay groups based on dimensionality
            - ZeRO stage 1 partitions optimizer states across GPUs to reduce memory
        """
        # Collect all parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optimizer parameter groups with selective weight decay
        # Rule: 2D+ tensors (weights) get decay, 1D tensors (biases, norms) don't
        # This is standard practice in transformers as biases/norms are already regularized
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Print parameter counts for diagnostic purposes
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Use fused AdamW if available (faster on CUDA)
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")

        # Create optimizer: ZeRO stage 1 for distributed training, regular AdamW otherwise
        if zero_stage == 1:
            # ZeroRedundancyOptimizer partitions optimizer states across GPUs
            # This reduces memory usage in multi-GPU training
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_rate, betas=betas, fused=use_fused)
            optimizer.add_param_group(optim_groups[1])
        else:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively given a conditioning sequence.

        This method performs autoregressive text generation by repeatedly:
        1. Computing logits for the next token given the current sequence
        2. Sampling from the predicted distribution
        3. Appending the sampled token to the sequence
        4. Repeating until max_new_tokens are generated

        The generation uses sampling strategies (temperature, top-k) to control
        randomness and quality of outputs.

        Args:
            idx (torch.LongTensor): Conditioning sequence of token indices, shape (B, T)
                where B is batch size and T is current sequence length
            max_new_tokens (int): Number of new tokens to generate
            temperature (float, optional): Temperature for sampling. Higher values
                (>1.0) make output more random, lower values (<1.0) make it more
                deterministic. Default: 1.0
            top_k (int, optional): If specified, only sample from the top k most
                likely tokens. Helps avoid sampling very unlikely tokens. Default: None
                (consider all tokens)

        Returns:
            torch.LongTensor: Extended sequence with generated tokens appended,
                shape (B, T + max_new_tokens)

        Example:
            >>> model.eval()
            >>> context = torch.tensor([[1, 2, 3]])  # Starting tokens
            >>> generated = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=40)
            >>> # generated contains original context + 100 new tokens

        Note:
            - This method should be called with model.eval() mode
            - Uses torch.no_grad() decorator for memory efficiency
            - Automatically crops context if it exceeds block_size
        """
        for _ in range(max_new_tokens):
            # Crop context if it's too long (keep only the most recent block_size tokens)
            # This is necessary because positional embeddings only go up to block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass to get logits for next token prediction
            logits, _ = self(idx_cond)

            # Extract logits for the last position (next token prediction)
            # Apply temperature scaling: higher temp = more uniform distribution
            logits = logits[:, -1, :] / temperature

            # Optionally apply top-k filtering to prevent sampling very unlikely tokens
            if top_k is not None:
                # Find the k-th largest logit value
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Set all logits below the k-th largest to -inf (will become 0 after softmax)
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Convert logits to probabilities via softmax
            probs = F.softmax(logits, dim=-1)

            # Sample one token from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    """
    Read only the header of a data shard to extract metadata.

    This function performs a quick read of just the file header without
    loading the full dataset, which is useful for validation and determining
    dataset size before loading.

    Args:
        filename (str): Path to the binary data shard file

    Returns:
        int: Number of tokens in the shard (ntok)

    Raises:
        SystemExit: If magic number doesn't match expected value (20240520)
        AssertionError: If file format version is not supported

    Binary Format:
        Header: 256 int32 values (1024 bytes total)
            [0]: Magic number (20240520 for current format)
            [1]: Version number (currently 1)
            [2]: Number of tokens in the file
            [3-255]: Reserved for future use
        Data: Follows header (not read by this function)
    """
    with open(filename, "rb") as f:
        # Read 256 int32 integers (4 bytes each = 1024 bytes total)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)

    # Validate file format
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)

    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # Number of tokens in this shard
    return ntok

def _load_data_shard(filename):
    """
    Load a complete data shard from disk.

    Reads the full binary file containing tokenized text data. The file
    consists of a header followed by token data stored as uint16 values.

    Args:
        filename (str): Path to the binary data shard file

    Returns:
        np.ndarray: Array of token indices (dtype=uint16)

    Raises:
        AssertionError: If magic number, version, or token count is incorrect

    Binary Format:
        Header: 256 int32 values (1024 bytes)
            [0]: Magic number (20240520)
            [1]: Version (1)
            [2]: Number of tokens (ntok)
        Data: ntok uint16 values (2 bytes each)
            Each value is a token index in range [0, vocab_size)
    """
    with open(filename, "rb") as f:
        # Read header
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # Number of tokens claimed in header

        # Read token data: stored as uint16 (2 bytes per token)
        # This limits vocab size to 65536, which is sufficient for GPT-2 (50257)
        tokens = np.frombuffer(f.read(), dtype=np.uint16)

    # Verify that the number of tokens matches the header
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    """
    Efficient data loader for distributed training across multiple GPUs.

    This data loader is designed for multi-GPU training with the following features:
    - Distributes data loading across multiple processes (one per GPU)
    - Supports datasets split into multiple shards (files)
    - Provides deterministic iteration for reproducibility
    - Automatically handles shard transitions
    - Memory-efficient: loads one shard at a time

    Each process reads from the same shard but at different positions to ensure
    no data overlap. When a shard is exhausted, all processes move to the next shard.

    Attributes:
        process_rank (int): Rank of current process (0 to num_processes-1)
        num_processes (int): Total number of processes (GPUs)
        B (int): Batch size per process
        T (int): Sequence length
        files (list): Sorted list of data shard file paths
        ntok_total (int): Total number of tokens across all shards
        current_shard (int): Index of currently loaded shard
        current_position (int): Current reading position in the shard
        tokens (np.ndarray): Currently loaded shard data

    Example:
        >>> # Single GPU
        >>> loader = DistributedDataLoader("data/train_*.bin", B=4, T=1024,
        ...                                process_rank=0, num_processes=1)
        >>> x, y = loader.next_batch()

        >>> # Multi-GPU with DDP (automatically uses environment variables)
        >>> ddp_rank = int(os.environ['RANK'])
        >>> ddp_world_size = int(os.environ['WORLD_SIZE'])
        >>> loader = DistributedDataLoader("data/train_*.bin", B=4, T=1024,
        ...                                process_rank=ddp_rank,
        ...                                num_processes=ddp_world_size)
    """

    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        """
        Initialize the distributed data loader.

        Args:
            filename_pattern (str): Glob pattern for data shard files
                (e.g., "data/train_*.bin")
            B (int): Batch size per process
            T (int): Sequence length for each example
            process_rank (int): Rank of this process (0-indexed)
            num_processes (int): Total number of processes in distributed setup

        Raises:
            AssertionError: If no files match the pattern
            AssertionError: If any shard is too small for the configuration
        """
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # Find all data shard files matching the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # Validate all shards and count total tokens
        # Each shard must have enough data for all processes to read one batch
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            # Need B*T tokens per process, plus 1 for the target
            # Times num_processes for all processes
            assert shard_ntok >= num_processes * B * T + 1, \
                f"Shard {fname} too small: has {shard_ntok}, needs {num_processes * B * T + 1}"
            ntok_total += shard_ntok

        self.ntok_total = ntok_total
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # Initialize loader state
        self.current_shard = None
        self.reset()

    def reset(self):
        """
        Reset the data loader to the beginning of the first shard.

        This is useful for re-starting iteration over the dataset, e.g., when
        overfitting a single batch or for validation loops.

        Optimization: If shard 0 is already loaded, just resets the position
        without reloading the file from disk.
        """
        # Optimization: avoid reloading if we're already on shard 0
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])

        # Set position for this process
        # Each process starts at a different offset to avoid reading the same data
        self.current_position = self.process_rank * self.B * self.T

    def advance(self):
        """
        Advance to the next data shard in the sequence.

        Moves to the next shard file, wrapping around to the first shard if
        we've reached the end. Loads the new shard and resets the position.
        """
        # Move to next shard (circular: wraps to 0 after last shard)
        self.current_shard = (self.current_shard + 1) % len(self.files)

        # Reset position for this process in the new shard
        self.current_position = self.process_rank * self.B * self.T

        # Load the new shard into memory
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        """
        Get the next batch of training data.

        Returns input and target sequences for language modeling. The target
        sequence is the input shifted by one position (next token prediction).

        Returns:
            tuple: (x, y) where:
                - x (torch.LongTensor): Input token indices, shape (B, T)
                - y (torch.LongTensor): Target token indices, shape (B, T)
                  Targets are inputs shifted by 1: y[i] = x[i+1]

        Implementation Details:
            - Reads B*T+1 consecutive tokens from the current position
            - First B*T tokens become inputs, last B*T tokens become targets
            - After reading, advances position by B*T*num_processes to ensure
              each process reads non-overlapping data
            - Automatically advances to next shard when current shard is exhausted
        """
        B = self.B
        T = self.T

        # Read B*T+1 tokens: we need the extra token for the shifted targets
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)

        # Split into inputs and targets (shifted by 1)
        x = (buf[:-1]).view(B, T)  # First B*T tokens as inputs
        y = (buf[1:]).view(B, T)   # Last B*T tokens as targets (shifted)

        # Advance position for this process
        # Skip ahead by B*T*num_processes so each process reads different data
        self.current_position += B * T * self.num_processes

        # Check if we need to move to the next shard
        # If the next batch would go out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()

        return x, y

# -----------------------------------------------------------------------------
# Python -> C bridge utilities for saving params/grads/activations to .bin files

def write_fp32(tensor, file):
    """
    Write a tensor to a binary file in float32 format.

    Converts the tensor to float32, moves to CPU, and writes raw bytes to file.
    This ensures consistent precision for the C implementation to read.

    Args:
        tensor (torch.Tensor): Tensor to write (any dtype, any device)
        file (BinaryIO): Open binary file handle for writing
    """
    t = tensor.detach().cpu().to(torch.float32)
    b = t.numpy().tobytes()
    file.write(b)

def write_bf16(tensor, file):
    """
    Write a tensor to a binary file in bfloat16 format.

    Converts tensor to bfloat16 and writes to file. Since numpy doesn't support
    bfloat16, we reinterpret the bytes as int16 for the conversion.

    Args:
        tensor (torch.Tensor): Tensor to write (any dtype, any device)
        file (BinaryIO): Open binary file handle for writing

    Note:
        Bfloat16 uses the same exponent range as float32 but with reduced
        precision (8 bits vs 23 bits for mantissa). This makes it suitable
        for deep learning where wide dynamic range is more important than precision.
    """
    t = tensor.detach().cpu().to(torch.bfloat16)
    # NumPy doesn't have bf16 datatype, so reinterpret as int16
    t = t.view(torch.int16)
    b = t.numpy().tobytes()
    file.write(b)

def write_tensors(model_tensors, L, file, dtype):
    """
    Write all GPT-2 model weights to a binary file in a specific order.

    The weights are written in a predetermined order that matches the C
    implementation's expectations. This allows the C code to read weights
    sequentially without seeking.

    Args:
        model_tensors (dict): State dict containing model parameters
        L (int): Number of layers in the model
        file (BinaryIO): Open binary file for writing
        dtype (str): Data type for writing ("float32" or "bfloat16")
    """
    assert dtype in {"float32", "bfloat16"}
    write_fun = write_fp32 if dtype == "float32" else write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    write_fun(model_tensors["transformer.wpe.weight"], file) # (T, C)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.bias"], file)
    for i in range(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    for i in range(L): # (L, 3C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
    for i in range(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.bias"], file)
    for i in range(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    for i in range(L): # (L, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
    for i in range(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    for i in range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.bias"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["transformer.ln_f.bias"], file) # (C, )

@torch.no_grad()
def pad_vocab(tensor, multiple=128, value=0):
    """
    Pad vocabulary size to a GPU-friendly multiple for efficient computation.

    GPT-2's vocabulary size is 50,257, which is not ideal for GPU operations.
    This function pads it to the nearest multiple (e.g., 50,304 for multiple=128),
    which significantly improves matrix multiplication performance on GPUs.

    This is algorithmically a no-op (padded entries are never used), but provides
    substantial performance benefits in the C implementation.

    Args:
        tensor (torch.Tensor): Vocabulary weight tensor, shape (V, C)
        multiple (int): Padding multiple, default 128 (good for GPU memory alignment)
        value (float): Value to use for padding rows, default 0

    Returns:
        torch.Tensor: Padded tensor of shape (Vp, C) where Vp is V rounded up
            to nearest multiple

    Example:
        >>> wte = torch.randn(50257, 768)  # GPT-2 vocab
        >>> wte_padded = pad_vocab(wte, multiple=128)
        >>> wte_padded.shape
        torch.Size([50304, 768])  # 50257 -> 50304
    """
    assert tensor.ndim == 2
    V, C = tensor.shape
    assert V == 50257, "just being defensive here"
    # calculate padded vocab size by rounding up to nearest multiple
    Vp = ((V + multiple - 1) // multiple) * multiple
    # pad the tensor
    pad_rows = Vp - V
    padded = tensor if pad_rows == 0 else F.pad(tensor, (0, 0, 0, pad_rows), value=value)
    assert padded.shape == (Vp, C)
    return padded

def write_model(model, filename, dtype):
    """
    Export GPT-2 model weights to a binary file for C implementation.

    Writes a complete model checkpoint including configuration and weights in a
    format that can be read by the C implementation.

    File Format:
        Header (1024 bytes): 256 int32 values containing:
            [0]: Magic number (20240326)
            [1]: Version (3=float32, 5=bfloat16)
            [2]: block_size
            [3]: vocab_size
            [4]: n_layer
            [5]: n_head
            [6]: n_embd
            [7]: padded_vocab_size
        Weights: All model parameters in predetermined order

    Args:
        model (GPT): GPT model to export
        filename (str): Output file path
        dtype (str): Data type for weights ("float32" or "bfloat16")
    """
    assert dtype in {"float32", "bfloat16"}  # float16 todo maybe later
    version = {
        "float32": 3, # 3: all tensors are fp32, padded vocab
        "bfloat16": 5, # 5: all tensors are bf16, padded vocab
    }[dtype]
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240326 # magic
    header[1] = version # checkpoint version
    header[2] = model.config.block_size
    header[3] = model.config.vocab_size
    header[4] = model.config.n_layer
    header[5] = model.config.n_head
    header[6] = model.config.n_embd
    # 2) the parameters follow the header
    params = {name: param.cpu() for name, param in model.named_parameters()}
    # pad the vocab to a multiple of 128 here at export, for efficiency in C
    wte = params["transformer.wte.weight"] # (V, C)
    wte_padded = pad_vocab(wte) # (Vp, C)
    params["transformer.wte.weight"] = wte_padded # (Vp, C)
    print(f"padded vocab size from {wte.size(0)} to {wte_padded.size(0)}")
    header[7] = wte_padded.size(0) # padded vocab size store in header
    # now write to file
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes()) # header
        write_tensors(params, model.config.n_layer, file, dtype) # params
    print(f"wrote {filename}")

def write_state(model, x, y, logits, loss, filename):
    """
    Export training state for debugging the C implementation.

    Writes inputs, outputs, loss, and gradients from one training step to a
    binary file. The C implementation can use this to verify its computations
    match the Python reference.

    File Format:
        Header (1024 bytes): 256 int32 values
        Input tokens (x): B*T int32 values
        Target tokens (y): B*T int32 values
        Logits: B*T*vocab_size float32 values
        Loss: 1 float32 value
        Gradients: All parameter gradients in same order as write_tensors

    Args:
        model (GPT): Model (must have gradients computed)
        x (torch.Tensor): Input tokens, shape (B, T)
        y (torch.Tensor): Target tokens, shape (B, T)
        logits (torch.Tensor): Model outputs, shape (B, T, vocab_size)
        loss (torch.Tensor): Scalar loss value
        filename (str): Output file path
    """
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240327 # magic
    header[1] = 2 # run state version = 2 (1 -> 2 for padded vocab changes)
    header[2] = x.size(0) # batch size of the batch, B
    header[3] = x.size(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() for name, param in model.named_parameters()}
    # pad the vocab grads here as well, to mirror write_model
    wte_grad = grads["transformer.wte.weight"] # (V, C)
    wte_grad_padded = pad_vocab(wte_grad, value=0) # (Vp, C) # TODO later maybe pad with nan?
    grads["transformer.wte.weight"] = wte_grad_padded # (Vp, C)
    print(f"padded vocab size in reference grads from {wte_grad.size(0)} to {wte_grad_padded.size(0)}")
    with open(filename, "wb") as file:
        # header
        file.write(header.numpy().tobytes())
        # input x
        file.write(x.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # targets y
        file.write(y.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # logits (result of the model forward pass)
        write_fp32(logits.cpu(), file)
        # loss (single float, result of the cross entropy loss)
        write_fp32(loss.cpu(), file)
        # gradients
        write_tensors(grads, model.config.n_layer, file, "float32")
    print(f"wrote {filename}")

def write_tokenizer(enc, filename):
    """
    Export tokenizer vocabulary to binary file for C implementation.

    Writes the tokenizer's vocabulary in a format the C code can read. Each
    token is stored as a length-prefixed byte sequence.

    File Format:
        Header (1024 bytes): 256 int32 values
            [0]: Magic number (20240328)
            [1]: Version (2)
            [2]: Number of tokens
            [3]: EOT token ID
        Token data: For each token:
            - 1 byte: length of token bytes
            - N bytes: UTF-8 encoded token bytes

    Args:
        enc: Tokenizer object with decode_bytes, max_token_value, and eot_token
        filename (str): Output file path
    """
    n = enc.max_token_value + 1
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240328 # magic
    header[1] = 2 # tokenizer version = 2 (1 -> 2: includes EOT token)
    header[2] = n # number of tokens
    header[3] = enc.eot_token # EOT token
    with open(filename, "wb") as file:
        file.write(header.numpy().tobytes())
        for i in range(n):
            b = enc.decode_bytes([i])
            length = len(b)
            assert length < 256, f"Token length exceeds 255: {length}"
            file.write(struct.pack("<B", length))  # Write the length as a 1-byte unsigned integer
            file.write(b)  # Write the actual bytes
    print(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    """
    Print only from the master process in distributed training.

    In multi-GPU training, this prevents duplicate log messages by only printing
    from rank 0. In single-GPU mode, behaves like regular print().

    Args:
        *args: Positional arguments passed to print()
        **kwargs: Keyword arguments passed to print()

    Note:
        Uses the RANK environment variable set by torchrun. If RANK is not set
        (single-GPU mode), defaults to 0 and prints normally.
    """
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

if __name__ == "__main__":
    import time
    import argparse
    import tiktoken
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("--input_bin", type=str, default="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, default="", help="input .bin to eval validation loss on")
    parser.add_argument("--output_dir", type=str, default="", help="output directory to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, default="gpt2", help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=int, default=4, help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=int, default=64, help="sequence length")
    parser.add_argument("--total_batch_size", type=int, default=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=int, default=10, help="number of iterations to run")
    parser.add_argument("--inference_only", type=int, default=0, help="only run inference")
    # optimization
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=int, default=0, help="learning rate warmup iterations")
    parser.add_argument("--learning_rate_decay_frac", type=float, default=1.0, help="learning rate warmup iterations")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument("--val_loss_every", type=int, default=0, help="every how mant steps to evaluate val loss?")
    parser.add_argument("--val_max_steps", type=int, default=20, help="how many batches of val to average?")
    parser.add_argument("--sample_every", type=int, default=0, help="how often to sample from the model?")
    # debugging
    parser.add_argument("--overfit_single_batch", type=int, default=1, help="overfit just one batch of data")
    # numerics
    parser.add_argument("--tensorcores", type=int, default=0, help="use tensorcores")
    # memory management
    parser.add_argument("--device", type=str, default="", help="by default we autodetect, or set it here")
    parser.add_argument("--compile", type=int, default=0, help="torch.compile the model")
    parser.add_argument("--flash", type=int, default=0, help="use flash attention")
    parser.add_argument("--dtype", type=str, default="float32", help="float32|float16|bfloat16")
    parser.add_argument("--zero_stage", type=int, default=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.add_argument("--write_tensors", type=int, default=1, help="write tensors to disk")
    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 1024
    assert args.dtype in {"float32", "float16", "bfloat16"}
    assert args.model in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "d12", "d24", "d36", "d48"}

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = 0 # each process gets the exact same seed
        zero_stage = args.zero_stage
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = True
        seed_offset = 0
        # select the device
        if args.device:
            # provided explicitly by the user
            device = args.device
        else:
            # attempt to autodetect the device
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
    print(f"using device: {device}")
    device_type = 'cuda' if 'cuda' in device else 'cpu'

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    tokens_per_fwdbwd = B * T * ddp_world_size
    assert args.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # rng / reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    if args.tensorcores:
        torch.set_float32_matmul_precision('high')

    # turn on/off flash attention
    assert args.flash in {0, 1}
    FLASH = args.flash

    # init (and write) the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    if master_process and args.write_tensors: # tokenizer is technically not tensors but ok
        write_tokenizer(enc, "gpt2_tokenizer.bin")

    # init the model, either from scratch or from OpenAI pretrained checkpoint
    if args.model[0] == "d":
        # from scratch (random weights)
        model_config = {
            "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
            "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
            "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
            "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
        }[args.model]
        model = GPT(model_config)
    else:
        # load the GPT-2 model weights
        model = GPT.from_pretrained(args.model)
    model.train()
    model.to(device)
    if args.compile:
        if hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = True # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    if args.input_val_bin:
        val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    if master_process and args.write_tensors and (not args.inference_only):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss.backward()
        # save model params, in both float32 and bfloat16
        model_to_size = {"gpt2": "124M", "gpt2-medium": "355M", "gpt2-large": "774M", "gpt2-xl": "1558M"}
        model_to_size.update({f"d{d}": f"d{d}" for d in [12, 24, 36, 48]})
        model_size_str = model_to_size[args.model] # e.g. "124M", or "d12"
        write_model(model, f"gpt2_{model_size_str}.bin", dtype="float32")
        write_model(model, f"gpt2_{model_size_str}_bf16.bin", dtype="bfloat16")
        # save x, y, logits, loss, and parameter gradients, for debugging C
        # always store these in fp32 to have an accurate reference (?)
        write_state(model, x, y, logits, loss, f"gpt2_{model_size_str}_debug_state.bin")
        # reset the train_loader for the optimization below
        train_loader.reset()

    # -------------------------------------------------------------------------
    # main training loop

    # here we wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
                                               learning_rate=args.learning_rate, betas=(0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it+1) / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.num_iterations:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.num_iterations - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (args.learning_rate - min_lr)

    # create the logging directory if it does not exist
    logfile = None
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        logfile = os.path.join(args.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") as f:
            pass

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings = []
    norm = -1.0   # dummy value to print in inference-only mode
    for step in range(args.num_iterations + 1):
        t0 = time.time()
        last_step = (step == args.num_iterations)

        # once in a while evaluate the validation dataset
        if (args.val_loss_every > 0 \
            and (step % args.val_loss_every == 0 or last_step)) \
            and (val_loader is not None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(args.val_max_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits=False)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            # log to console and to file
            print0(f"val loss {val_loss}")
            if master_process and logfile is not None:
                with open(logfile, "a") as f:
                    f.write("s:%d tel:%f\n" % (step, val_loss))

        # once in a while perform model inference on the master process
        if (args.sample_every > 0 \
            and (step % args.sample_every == 0 or last_step)) \
            and master_process:
            model.eval()
            # before we end, let's also do one round of inference
            # we'll kick off the generation with "<|endoftext|>", which designates the start of a new sequence
            start_ids = [enc.eot_token]
            xg = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            max_new_tokens = 32
            temperature = 1.0
            top_k = 40
            yg = raw_model.generate(xg, max_new_tokens, temperature=temperature, top_k=top_k)
            print0('---------------')
            print0(enc.decode(yg[0].tolist()))
            print0('---------------')

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # if we are trying to overfit a single batch, we reset the loader here
        if args.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        for micro_step in range(grad_accum_steps):
            # fetch a batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            if ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=False)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_steps
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            if not args.inference_only:
                loss.backward()
        if ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t1-t0)
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log to logile
        if master_process and logfile is not None:
            with open(logfile, "a") as f:
                f.write("s:%d trl:%f\n" % (step, lossf))

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1-t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    if ddp:
        destroy_process_group()
