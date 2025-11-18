import torch
from torch import Tensor
from typing import Tuple

# Step 1: Define custom operators using torch.library API
@torch.library.custom_op("my_ops::batchnorm_forward", mutates_args=("running_mean", "running_var"))
def batchnorm_forward(
    input: Tensor,           # [N, C, H, W]
    gamma: Tensor,           # [C]
    beta: Tensor,            # [C]
    running_mean: Tensor,    # [C]
    running_var: Tensor,     # [C]
    training: bool,
    momentum: float,
    eps: float
) -> Tuple[Tensor, Tensor, Tensor]:
    """forward pass of BatchNorm for 4D input [N, C, H, W]."""

    # Implement Here
    N, C, H, W = input.shape
    dims = (0, 2, 3)  # reduce over N,H,W
    g = gamma.view(1, -1, 1, 1)
    b = beta.view(1, -1, 1, 1)

    if training:
        mean = input.mean(dims)                                # [C]
        var = input.var(dims, unbiased=False)                  # [C]
        invstd = torch.rsqrt(var + eps)                        # [C]

        x_hat = (input - mean.view(1, -1, 1, 1)) * invstd.view(1, -1, 1, 1)
        output = x_hat * g + b

        # In-place running stats update
        running_mean.mul_(1.0 - momentum).add_(momentum * mean)
        running_var.mul_(1.0 - momentum).add_(momentum * var)

        save_mean = mean
        save_invstd = invstd
    else:
        invstd = torch.rsqrt(running_var + eps)                # [C]
        x_hat = (input - running_mean.view(1, -1, 1, 1)) * invstd.view(1, -1, 1, 1)
        output = x_hat * g + b

        # Save the exact stats used for normalization for backward
        save_mean = running_mean.detach().clone()
        save_invstd = invstd  # freshly computed; no need to clone
    
    return output, save_mean, save_invstd


@torch.library.custom_op("my_ops::batchnorm_backward", mutates_args=())
def batchnorm_backward(
    grad_output: Tensor,     # [N, C, H, W]
    input: Tensor,           # [N, C, H, W]
    gamma: Tensor,           # [C]
    save_mean: Tensor,       # [C]
    save_invstd: Tensor      # [C]
) -> Tuple[Tensor, Tensor, Tensor]:
    """backward pass of BatchNorm for 4D input."""

    # Implement Here
    N, C, H, W = input.shape
    m = float(N * H * W)
    dims = (0, 2, 3)

    g = gamma.view(1, -1, 1, 1)
    mean = save_mean.view(1, -1, 1, 1)
    invstd = save_invstd.view(1, -1, 1, 1)

    x_hat = (input - mean) * invstd

    # Gradients for scale (gamma) and shift (beta)
    grad_beta = grad_output.sum(dims)                  # [C]
    grad_gamma = (grad_output * x_hat).sum(dims)       # [C]

    # Determine if stats came from current batch (training) or running stats (eval)
    # If they match (within tolerance), treat as training; otherwise eval-path.
    sample_mean = input.mean(dims)
    is_training_like = torch.allclose(sample_mean, save_mean, rtol=1e-5, atol=1e-5)

    dy = grad_output
    dy_g = dy * g                                      # [N,C,H,W]

    if is_training_like:
        # Training-mode backward:
        # dX = (1/m) * invstd * ( m*dy_g - sum(dy_g) - x_hat*sum(dy_g*x_hat) )
        sum_dy = dy_g.sum(dims).view(1, -1, 1, 1)                 # [1,C,1,1]
        sum_dy_xhat = (dy_g * x_hat).sum(dims).view(1, -1, 1, 1)  # [1,C,1,1]
        grad_input = (invstd / m) * (m * dy_g - sum_dy - x_hat * sum_dy_xhat)
    else:
        # Eval-mode backward (normalization constants are treated as constants)
        grad_input = dy_g * invstd

    return grad_input, grad_gamma, grad_beta


# Step 2: Connect forward and backward with autograd
# This connects our custom forward/backward operators to PyTorch's 
# autograd system, allowing gradients to flow during backpropagation
class BatchNormCustom(torch.autograd.Function):
    """
    Custom Batch Normalization for 4D inputs [N, C, H, W].
    
    Bridges custom operators with PyTorch's autograd engine.
    - forward(): calls custom forward operator and saves context
    - backward(): calls custom backward operator using saved context

    Usage:
        output = BatchNormCustom.apply(input, gamma, beta, running_mean, running_var, training, momentum, eps)
    """
    @staticmethod
    def forward(ctx, input, gamma, beta, running_mean, running_var, training, momentum, eps):
        output, save_mean, save_invstd = torch.ops.my_ops.batchnorm_forward(
            input, gamma, beta, running_mean, running_var, training, momentum, eps
        )
        ctx.save_for_backward(input, gamma, save_mean, save_invstd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, gamma, save_mean, save_invstd = ctx.saved_tensors
        grad_input, grad_gamma, grad_beta = torch.ops.my_ops.batchnorm_backward(
            grad_output, input, gamma, save_mean, save_invstd
        )
        # Return gradients for all forward inputs (None for non-tensor args)
        return grad_input, grad_gamma, grad_beta, None, None, None, None, None
