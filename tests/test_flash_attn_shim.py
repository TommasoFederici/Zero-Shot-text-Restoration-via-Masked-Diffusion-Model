"""Local correctness checks for the pure-PyTorch flash_attn shim.

Uses only small synthetic CPU tensors -- no real model, no GPU, per the
project's local-testing rules.
"""

from __future__ import annotations

import sys

import torch

from src.flash_attn_shim import (
    apply_rotary_emb_qkv_,
    flash_attn_varlen_qkvpacked_func,
    install,
)


def test_apply_rotary_emb_qkv_leaves_value_slice_unchanged() -> None:
    batch, seq_len, n_heads, head_dim = 1, 4, 2, 8
    qkv = torch.randn(batch, seq_len, 3, n_heads, head_dim)
    cos = torch.randn(seq_len, head_dim // 2)
    sin = torch.randn(seq_len, head_dim // 2)

    result = apply_rotary_emb_qkv_(qkv, cos, sin)

    assert torch.equal(result[:, :, 2], qkv[:, :, 2])


def test_apply_rotary_emb_qkv_is_identity_when_cos_one_sin_zero() -> None:
    batch, seq_len, n_heads, head_dim = 1, 3, 2, 4
    qkv = torch.randn(batch, seq_len, 3, n_heads, head_dim)
    cos = torch.ones(seq_len, head_dim // 2)
    sin = torch.zeros(seq_len, head_dim // 2)

    result = apply_rotary_emb_qkv_(qkv, cos, sin)

    assert torch.allclose(result, qkv, atol=1e-6)


def test_flash_attn_varlen_qkvpacked_func_output_shape() -> None:
    total_tokens, n_heads, head_dim = 7, 2, 4
    qkv = torch.randn(total_tokens, 3, n_heads, head_dim)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)

    output = flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, max_seqlen=4, dropout_p=0.0, causal=False
    )

    assert output.shape == (total_tokens, n_heads, head_dim)


def test_flash_attn_varlen_qkvpacked_func_segments_do_not_attend_across() -> None:
    n_heads, head_dim = 1, 4
    seg_a = torch.randn(3, 3, n_heads, head_dim)
    seg_b_v1 = torch.randn(2, 3, n_heads, head_dim)
    seg_b_v2 = torch.randn(2, 3, n_heads, head_dim)
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)

    out_v1 = flash_attn_varlen_qkvpacked_func(
        torch.cat([seg_a, seg_b_v1], dim=0), cu_seqlens, max_seqlen=3
    )
    out_v2 = flash_attn_varlen_qkvpacked_func(
        torch.cat([seg_a, seg_b_v2], dim=0), cu_seqlens, max_seqlen=3
    )

    # Segment A's output must be unaffected by changing segment B's content.
    assert torch.allclose(out_v1[:3], out_v2[:3], atol=1e-6)


def test_install_registers_modules_and_is_idempotent() -> None:
    install()
    install()

    assert "flash_attn" in sys.modules
    assert "flash_attn.layers.rotary" in sys.modules
    assert "flash_attn.flash_attn_interface" in sys.modules
    assert (
        sys.modules["flash_attn.layers.rotary"].apply_rotary_emb_qkv_
        is apply_rotary_emb_qkv_
    )
    assert (
        sys.modules["flash_attn.flash_attn_interface"].flash_attn_varlen_qkvpacked_func
        is flash_attn_varlen_qkvpacked_func
    )
