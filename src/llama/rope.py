from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    # todo: rotate each (even, odd) pair by its position-dependent angle
    q_real, q_imag = query.float().reshape(*query.shape[:-1], -1, 2).unbind(-1)   # (B, T, H, d/2)
    k_real, k_imag = key.float().reshape(*key.shape[:-1], -1, 2).unbind(-1)       # (B, T, Hkv, d/2)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=query.device).float() / head_dim))   # (d/2,)
    seqlen = query.shape[1]
    t = torch.arange(seqlen, device=query.device).float()   # m = 0 … T−1, (T,)
    angles = torch.outer(t, freqs)                           # angles[m, i] = m·θᵢ, (T, d/2)
    cos = reshape_for_broadcast(angles.cos(), q_real)   # (T, d/2) → (1, T, 1, d/2)
    sin = reshape_for_broadcast(angles.sin(), q_real)
    q_out_real = q_real * cos - q_imag * sin
    q_out_imag = q_real * sin + q_imag * cos
    k_out_real = k_real * cos - k_imag * sin
    k_out_imag = k_real * sin + k_imag * cos
    query_out = torch.stack([q_out_real, q_out_imag], dim=-1).flatten(-2)   # (B, T, H, d)
    key_out   = torch.stack([k_out_real, k_out_imag], dim=-1).flatten(-2)

    return query_out.type_as(query), key_out.type_as(key)

