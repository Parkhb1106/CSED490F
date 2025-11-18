"""
Triton kernels for batch normalization.

This module provides pure Triton kernel implementations for batch normalization.
The kernels use Welford's online algorithm for numerically stable mean and
variance computation.

For PyTorch integration, use the custom operators registered in ops.py.

Note: For production use, the CUDA implementation is recommended for better
performance and stability.
"""

import triton
import triton.language as tl


@triton.jit
def batchnorm_forward_training_kernel(
    # Pointers
    input_ptr,          # Input tensor pointer
    gamma_ptr,          # Scale parameter pointer
    beta_ptr,           # Shift parameter pointer
    output_ptr,         # Output tensor pointer
    mean_ptr,           # Saved mean pointer
    invstd_ptr,         # Saved inverse std pointer
    running_mean_ptr,   # Running mean pointer (updated in-place)
    running_var_ptr,    # Running variance pointer (updated in-place)
    # Dimensions
    N, C, spatial_dim,  # Batch size, channels, spatial dimension (H*W)
    # Parameters
    momentum,           # Momentum for running stats
    eps,                # Epsilon for numerical stability
    # Strides
    stride_n, stride_c, stride_s,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # TODO: Implement this kernel
    c = tl.program_id(0)
    if c >= C:
        return

    M = N * spatial_dim
    offs = tl.arange(0, BLOCK_SIZE)

    # ----- Pass 1: mean/var (fp32 누적) -----
    sum_x  = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)

    base = 0
    while base < M:
        i = base + offs
        mask = i < M
        n = i // spatial_dim
        s = i - n * spatial_dim
        xptr = input_ptr + n * stride_n + c * stride_c + s * stride_s
        x = tl.load(xptr, mask=mask, other=0.0).to(tl.float32)
        sum_x  += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)
        base += BLOCK_SIZE

    m = tl.full((), M, tl.float32)
    mean = sum_x / m
    var  = sum_x2 / m - mean * mean
    var  = tl.maximum(var, 0.0)
    invstd = 1.0 / tl.sqrt(var + eps)

    # Save batch stats
    tl.store(mean_ptr + c, mean)
    tl.store(invstd_ptr + c, invstd)

    # Update running stats: (1-m)*old + m*batch
    run_mean = tl.load(running_mean_ptr + c).to(tl.float32)
    run_var  = tl.load(running_var_ptr  + c).to(tl.float32)
    mm = momentum.to(tl.float32)
    run_mean = (1.0 - mm) * run_mean + mm * mean
    run_var  = (1.0 - mm) * run_var  + mm * var
    tl.store(running_mean_ptr + c, run_mean)
    tl.store(running_var_ptr  + c, run_var)

    g = tl.load(gamma_ptr + c).to(tl.float32)
    b = tl.load(beta_ptr  + c).to(tl.float32)

    # ----- Pass 2: normalize & affine (저장 전 dtype 맞춤) -----
    base = 0
    while base < M:
        i = base + offs
        mask = i < M
        n = i // spatial_dim
        s = i - n * spatial_dim

        iptr = input_ptr  + n * stride_n + c * stride_c + s * stride_s
        optr = output_ptr + n * stride_n + c * stride_c + s * stride_s

        x = tl.load(iptr, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        y = (x32 - mean) * invstd * g + b
        tl.store(optr, y.to(x.dtype), mask=mask)
        base += BLOCK_SIZE


@triton.jit
def batchnorm_forward_inference_kernel(
    # Pointers
    input_ptr,          # Input tensor pointer
    gamma_ptr,          # Scale parameter pointer
    beta_ptr,           # Shift parameter pointer
    mean_ptr,           # Running mean pointer
    var_ptr,            # Running variance pointer
    output_ptr,         # Output tensor pointer
    invstd_ptr,         # Saved inverse std pointer (for backward)
    # Dimensions
    N, C, spatial_dim,  # Batch size, channels, spatial dimension
    # Parameters
    eps,                # Epsilon for numerical stability
    # Strides
    stride_n, stride_c, stride_s,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # TODO: Implement this kernel
    c = tl.program_id(0)
    if c >= C:
        return

    M = N * spatial_dim
    offs = tl.arange(0, BLOCK_SIZE)

    mean   = tl.load(mean_ptr + c).to(tl.float32)
    var    = tl.load(var_ptr  + c).to(tl.float32)
    var    = tl.maximum(var, 0.0)
    invstd = 1.0 / tl.sqrt(var + eps)
    tl.store(invstd_ptr + c, invstd)

    g = tl.load(gamma_ptr + c).to(tl.float32)
    b = tl.load(beta_ptr  + c).to(tl.float32)

    base = 0
    while base < M:
        i = base + offs
        mask = i < M
        n = i // spatial_dim
        s = i - n * spatial_dim

        iptr = input_ptr  + n * stride_n + c * stride_c + s * stride_s
        optr = output_ptr + n * stride_n + c * stride_c + s * stride_s

        x = tl.load(iptr, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        y = (x32 - mean) * invstd * g + b
        tl.store(optr, y.to(x.dtype), mask=mask)
        base += BLOCK_SIZE



@triton.jit
def batchnorm_backward_kernel(
    # Pointers
    grad_output_ptr,    # Gradient from next layer
    input_ptr,          # Original input
    gamma_ptr,          # Scale parameter
    mean_ptr,           # Saved mean from forward
    invstd_ptr,         # Saved inverse std from forward
    grad_input_ptr,     # Gradient w.r.t. input (output)
    grad_gamma_ptr,     # Gradient w.r.t. gamma (output)
    grad_beta_ptr,      # Gradient w.r.t. beta (output)
    # Dimensions
    N, C, spatial_dim,
    # Strides
    stride_n, stride_c, stride_s,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
):
    # TODO: Implement this kernel
    c = tl.program_id(0)
    if c >= C:
        return

    M = N * spatial_dim
    offs = tl.arange(0, BLOCK_SIZE)

    mean   = tl.load(mean_ptr   + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    g      = tl.load(gamma_ptr  + c).to(tl.float32)

    # ----- Pass 1: dβ, dγ 누적 (fp32) -----
    sum_dy      = tl.zeros((), dtype=tl.float32)
    sum_dy_xhat = tl.zeros((), dtype=tl.float32)

    base = 0
    while base < M:
        i = base + offs
        mask = i < M
        n = i // spatial_dim
        s = i - n * spatial_dim

        goptr = grad_output_ptr + n * stride_n + c * stride_c + s * stride_s
        iptr  = input_ptr        + n * stride_n + c * stride_c + s * stride_s

        dy = tl.load(goptr, mask=mask, other=0.0).to(tl.float32)
        x  = tl.load(iptr,  mask=mask, other=0.0).to(tl.float32)
        xhat = (x - mean) * invstd

        sum_dy      += tl.sum(dy, axis=0)
        sum_dy_xhat += tl.sum(dy * xhat, axis=0)
        base += BLOCK_SIZE

    tl.store(grad_beta_ptr  + c, sum_dy)
    tl.store(grad_gamma_ptr + c, sum_dy_xhat)

    # ----- Pass 2: grad_input (저장 전 원 dtype으로 캐스팅) -----
    m = tl.full((), M, tl.float32)
    base = 0
    while base < M:
        i = base + offs
        mask = i < M
        n = i // spatial_dim
        s = i - n * spatial_dim

        goptr = grad_output_ptr + n * stride_n + c * stride_c + s * stride_s
        iptr  = input_ptr        + n * stride_n + c * stride_c + s * stride_s
        giptr = grad_input_ptr   + n * stride_n + c * stride_c + s * stride_s

        dy = tl.load(goptr, mask=mask, other=0.0).to(tl.float32)
        x  = tl.load(iptr,  mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        xhat = (x32 - mean) * invstd

        dx = (g * invstd / m) * (m * dy - sum_dy - xhat * sum_dy_xhat)
        tl.store(giptr, dx.to(x.dtype), mask=mask)
        base += BLOCK_SIZE

