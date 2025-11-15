"""
Layer Normalization Reference Implementation
==============================================

This script provides a PyTorch reference implementation of Layer Normalization
for educational purposes and to generate test data for the C implementation.

PURPOSE:
1. Demonstrate LayerNorm forward and backward passes using basic PyTorch operations
2. Validate our manual backward pass against PyTorch's autograd
3. Generate reference binary data (ln.bin) for testing the C implementation

BACKGROUND:
Instead of using PyTorch's built-in nn.LayerNorm (which is highly optimized
but difficult to understand), we implement LayerNorm manually using simple
tensor operations. This helps us understand the algorithm before implementing
it in C.

USAGE:
Run this script to generate ln.bin:
    python layernorm.py

The generated file will contain test inputs and expected outputs for both
forward and backward passes, which the C program will use for validation.
"""

import torch

# Small constant to prevent division by zero in normalization
eps = 1e-5

class LayerNorm:
    """
    Manual implementation of Layer Normalization using basic PyTorch operations.

    This class demonstrates the mathematical operations behind LayerNorm without
    using PyTorch's optimized nn.LayerNorm module. It implements both forward
    and backward passes manually.
    """

    @staticmethod
    def forward(x, w, b):
        """
        Forward pass of Layer Normalization.

        Args:
            x: Input tensor of shape (B, T, C)
               B = batch size, T = sequence length, C = channels/features
            w: Weight tensor of shape (C,) - learnable scale parameters
            b: Bias tensor of shape (C,) - learnable shift parameters

        Returns:
            out: Normalized output of shape (B, T, C)
            cache: Tuple of values needed for backward pass (x, w, mean, rstd)

        Algorithm:
            1. Compute mean across channel dimension (C) for each (B, T) position
            2. Compute variance across channel dimension
            3. Normalize: (x - mean) / sqrt(variance + epsilon)
            4. Scale and shift: normalized * weight + bias
        """
        B, T, C = x.size()

        # Step 1: Calculate mean across the channel dimension (dim=-1)
        # Sum over C and divide by C to get mean for each (B, T) position
        # keepdim=True maintains shape as (B, T, 1) for broadcasting
        mean = x.sum(-1, keepdim=True) / C  # Shape: (B, T, 1)

        # Step 2: Center the input by subtracting mean
        # Broadcasting: (B, T, C) - (B, T, 1) = (B, T, C)
        xshift = x - mean  # Shape: (B, T, C)

        # Step 3: Calculate variance (mean of squared deviations)
        # Note: This is population variance, not sample variance (no Bessel's correction)
        var = (xshift**2).sum(-1, keepdim=True) / C  # Shape: (B, T, 1)

        # Step 4: Calculate reciprocal standard deviation
        # Adding eps prevents division by zero
        # Using **-0.5 is equivalent to 1/sqrt(var + eps)
        rstd = (var + eps) ** -0.5  # Shape: (B, T, 1)

        # Step 5: Normalize the centered input
        # Multiply by reciprocal std dev (equivalent to dividing by std dev)
        norm = xshift * rstd  # Shape: (B, T, C)

        # Step 6: Apply learnable affine transformation (scale and shift)
        # Element-wise multiply by weight and add bias
        out = norm * w + b  # Shape: (B, T, C)

        # Cache values needed for backward pass
        # We save x, w, mean, and rstd (not norm, to save memory)
        cache = (x, w, mean, rstd)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass of Layer Normalization.

        Args:
            dout: Gradient of loss w.r.t. output, shape (B, T, C)
                  This is the upstream gradient from the next layer
            cache: Tuple (x, w, mean, rstd) saved from forward pass

        Returns:
            dx: Gradient of loss w.r.t. input x, shape (B, T, C)
            dw: Gradient of loss w.r.t. weights w, shape (C,)
            db: Gradient of loss w.r.t. bias b, shape (C,)

        Mathematical Derivation:
            Given forward pass: out = w * ((x - mean) / sqrt(var + eps)) + b
            We apply the chain rule to compute gradients.

            The input gradient is complex because each input element affects:
            1. Its own normalized value directly
            2. The mean (which affects all C elements)
            3. The variance (which affects all C elements)

            The final simplified formula accounts for all three effects.
        """
        x, w, mean, rstd = cache

        # Recompute normalized values (memory vs. compute tradeoff)
        # We could have cached this in forward, but it would use B*T*C memory
        # Recomputing uses only the cached mean & rstd (B*T memory)
        norm = (x - mean) * rstd  # Shape: (B, T, C)

        # Gradient w.r.t. bias: db/dbias = 1, so db = sum of dout
        # Sum over batch (dim=0) and time (dim=1) dimensions
        db = dout.sum((0, 1))  # Shape: (C,)

        # Gradient w.r.t. weights: dout/dw = norm, so dw = sum of (dout * norm)
        # Sum over batch and time dimensions
        dw = (dout * norm).sum((0, 1))  # Shape: (C,)

        # Gradient w.r.t. input (most complex part)
        # First, compute gradient w.r.t. normalized values
        # From: out = norm * w + b, we get: dnorm = dout * w
        dnorm = dout * w  # Shape: (B, T, C)

        # Now compute gradient w.r.t. input
        # This formula accounts for three ways input affects output:
        # 1. dnorm: Direct effect through normalization
        # 2. dnorm.mean(...): Effect through the mean
        # 3. norm * (dnorm * norm).mean(...): Effect through variance
        # The formula is derived by applying chain rule to:
        # norm = (x - mean) * rstd, where mean and rstd depend on x
        dx = dnorm - dnorm.mean(-1, keepdim=True) - norm * (dnorm * norm).mean(-1, keepdim=True)

        # Final scaling by reciprocal standard deviation
        dx *= rstd  # Shape: (B, T, C)

        return dx, dw, db

# ============================================================================
# VALIDATION: Test our manual backward pass against PyTorch autograd
# ============================================================================

# Create small test tensors with random values
B = 2  # Batch size
T = 3  # Sequence length
C = 4  # Number of channels/features

# Create input tensors with gradient tracking enabled
# requires_grad=True tells PyTorch to track operations for autograd
x = torch.randn(B, T, C, requires_grad=True)
w = torch.randn(C, requires_grad=True)
b = torch.randn(C, requires_grad=True)

# Run our manual forward pass
out, cache = LayerNorm.forward(x, w, b)

# Create random upstream gradients (simulating gradients from next layer)
dout = torch.randn(B, T, C)

# Run our manual backward pass
dx, dw, db = LayerNorm.backward(dout, cache)

# ============================================================================
# Compare our manual gradients to PyTorch's autograd
# ============================================================================

# Create a fake loss by taking weighted sum of outputs
# This is just to have a scalar to call .backward() on
# The weights are the random dout values we created
fakeloss = (out * dout).sum()

# Use PyTorch's autograd to compute gradients
# This will populate the .grad attributes of x, w, and b
fakeloss.backward()

# Compare our manual gradients to PyTorch's autograd gradients
# If our implementation is correct, these errors should be very small (< 1e-6)
print("dx error:", (x.grad - dx).abs().max().item())
print("dw error:", (w.grad - dw).abs().max().item())
print("db error:", (b.grad - db).abs().max().item())

# ============================================================================
# GENERATE REFERENCE DATA: Write test data to binary file for C validation
# ============================================================================

# Extract cached values from forward pass
x, w, mean, rstd = cache

def write(tensor, handle):
    """
    Write a PyTorch tensor to a binary file in float32 format.

    Args:
        tensor: PyTorch tensor to write
        handle: File handle opened in binary write mode

    The tensor is:
    1. Detached from the computation graph (no gradient tracking)
    2. Converted to NumPy array
    3. Cast to float32 (standard C float type)
    4. Written as raw bytes
    """
    handle.write(tensor.detach().numpy().astype("float32").tobytes())

# Write all test data to ln.bin
# This file will be read by the C program (layernorm.c) for validation
# IMPORTANT: The order of writes must match the order of reads in the C code!
with open('ln.bin', 'wb') as file:
    # Write forward pass inputs
    write(x, file)     # Input tensor (B, T, C)
    write(w, file)     # Weights (C,)
    write(b, file)     # Biases (C,)

    # Write forward pass expected outputs
    write(out, file)   # Normalized output (B, T, C)
    write(mean, file)  # Computed means (B, T)
    write(rstd, file)  # Reciprocal std devs (B, T)

    # Write backward pass inputs and expected outputs
    write(dout, file)  # Upstream gradients (B, T, C)
    write(dx, file)    # Expected input gradients (B, T, C)
    write(dw, file)    # Expected weight gradients (C,)
    write(db, file)    # Expected bias gradients (C,)

print("\nSuccessfully wrote reference data to ln.bin")
print("Now run: gcc layernorm.c -o layernorm -lm && ./layernorm")
