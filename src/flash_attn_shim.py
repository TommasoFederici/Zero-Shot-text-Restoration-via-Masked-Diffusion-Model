"""Pure-PyTorch stand-in for the `flash_attn` package.

`kuleshov-group/mdlm-owt`'s custom `trust_remote_code=True` modeling code
(`modeling_mdlm.py`) hard-imports `flash_attn`/`flash_attn.layers.rotary`/
`flash_attn.flash_attn_interface` with no eager/sdpa fallback path. The
real `flash-attn` PyPI package (FlashAttention-2) only supports Ampere,
Ada, or Hopper GPUs -- it does not run on Colab's free-tier T4 (Turing).

`install()` registers functionally-equivalent pure-PyTorch
implementations of the two functions that model actually calls
(rotary embedding application and packed variable-length attention)
under the same module names in `sys.modules`, so both `transformers`'
`trust_remote_code` import check and the model's own forward pass
succeed on any GPU -- at some performance cost vs. real flash-attn
kernels, since this falls back to `scaled_dot_product_attention`.
"""

from __future__ import annotations

import sys
import types

import torch
import torch.nn.functional as F


def apply_rotary_emb_qkv_(
    qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary position embeddings to the q/k slices of packed qkv.

    Stand-in for `flash_attn.layers.rotary.apply_rotary_emb_qkv_`. Only
    the query and key slices (indices 0 and 1 along the packed "three"
    dimension) are rotated; the value slice (index 2) is left untouched,
    matching flash_attn's packed-qkv convention.

    Args:
        qkv: packed query/key/value tensor, shape
            `[batch, seq_len, 3, n_heads, head_dim]`.
        cos: rotary cosine table, shape `[seq_len, head_dim // 2]`.
        sin: rotary sine table, shape `[seq_len, head_dim // 2]`.

    Returns:
        A new tensor of the same shape as `qkv` with rotary embeddings
        applied to its q and k slices.
    """
    cos_full = torch.cat([cos, cos], dim=-1)[None, :, None, :]
    sin_full = torch.cat([sin, sin], dim=-1)[None, :, None, :]

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    q_rotated = q * cos_full + _rotate_half(q) * sin_full
    k_rotated = k * cos_full + _rotate_half(k) * sin_full
    return torch.stack([q_rotated, k_rotated, v], dim=2)


def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    causal: bool = False,
) -> torch.Tensor:
    """Compute packed variable-length self-attention via plain PyTorch.

    Stand-in for
    `flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func`,
    implemented with `torch.nn.functional.scaled_dot_product_attention`
    over each `cu_seqlens` segment independently.

    Args:
        qkv: packed query/key/value tensor, shape
            `[total_tokens, 3, n_heads, head_dim]`.
        cu_seqlens: cumulative sequence-length boundaries, shape
            `[num_segments + 1]`.
        max_seqlen: unused; kept for interface parity with flash_attn.
        dropout_p: attention dropout probability.
        causal: whether to apply a causal attention mask.

    Returns:
        Attention output, shape `[total_tokens, n_heads, head_dim]`.
    """
    del max_seqlen
    boundaries = cu_seqlens.tolist()
    outputs = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        segment = qkv[start:end]
        q, k, v = segment[:, 0], segment[:, 1], segment[:, 2]
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=causal
        )
        outputs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def install() -> None:
    """Register the pure-PyTorch `flash_attn` stand-in in `sys.modules`.

    Idempotent: safe to call multiple times, including before any real
    `flash_attn` package might already be importable (this shim always
    takes precedence once installed).
    """
    if getattr(sys.modules.get("flash_attn"), "_is_pytorch_shim", False):
        return

    flash_attn_module = types.ModuleType("flash_attn")
    flash_attn_module._is_pytorch_shim = True

    rotary_module = types.ModuleType("flash_attn.layers.rotary")
    rotary_module.apply_rotary_emb_qkv_ = apply_rotary_emb_qkv_

    layers_module = types.ModuleType("flash_attn.layers")
    layers_module.rotary = rotary_module

    interface_module = types.ModuleType("flash_attn.flash_attn_interface")
    interface_module.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func

    flash_attn_module.layers = layers_module
    flash_attn_module.flash_attn_interface = interface_module

    sys.modules["flash_attn"] = flash_attn_module
    sys.modules["flash_attn.layers"] = layers_module
    sys.modules["flash_attn.layers.rotary"] = rotary_module
    sys.modules["flash_attn.flash_attn_interface"] = interface_module
