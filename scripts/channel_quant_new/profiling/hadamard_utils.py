"""Block-wise Hadamard transform utilities for NVFP4 quantization."""
import torch
import torch.nn.functional as F


def hadamard_matrix(n: int) -> torch.Tensor:
    if n == 1:
        return torch.tensor([[1.0]])
    half = hadamard_matrix(n // 2)
    return torch.cat([
        torch.cat([half, half], dim=1),
        torch.cat([half, -half], dim=1),
    ], dim=0) / (2 ** 0.5)


# Block size = 16 matches NVFP4's per-16-element scaling group
H16 = hadamard_matrix(16)


def build_block_diagonal_hadamard(input_dim: int, block_size: int = 16) -> torch.Tensor:
    h = hadamard_matrix(block_size)
    n_blocks = input_dim // block_size
    remainder = input_dim % block_size
    blocks = [h] * n_blocks
    if remainder > 0:
        blocks.append(torch.eye(remainder))
    return torch.block_diag(*blocks)


def rotate_weight_offline(weight: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Rotate weight along input dimension: W_rot = W @ H_block.
    
    weight shape: [out_channels, in_dim]
    H_block: block-diagonal Hadamard, [in_dim, in_dim]
    """
    in_dim = weight.shape[1]
    h = hadamard_matrix(block_size).to(device=weight.device, dtype=weight.dtype)
    n_blocks = in_dim // block_size
    out = weight.clone()
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size
        out[:, start:end] = weight[:, start:end] @ h
    return out


def rotate_input_online(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Rotate input along feature dimension: x_rot = x @ H_block.
    
    x shape: [..., in_dim]
    """
    in_dim = x.shape[-1]
    h = hadamard_matrix(block_size).to(device=x.device, dtype=x.dtype)
    n_blocks = in_dim // block_size
    out = x.clone()
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size
        out[..., start:end] = x[..., start:end] @ h
    return out
