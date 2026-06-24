"""
Paged KV Cache — GPU tensor management for PagedAttention.

The physical KV cache is a pair of pre-allocated tensors per layer:

  k_cache[layer]: [num_blocks, block_size, num_kv_heads, head_dim]  fp16
  v_cache[layer]: [num_blocks, block_size, num_kv_heads, head_dim]  fp16

Layout is (num_blocks, block_size, ...) so that
  kc.view(-1, num_kv_heads, head_dim)
produces a contiguous flat view indexed by (block * block_size + slot).
This enables vectorised scatter/gather via torch.gather without copying.

During a forward pass, PagedKVCache is passed as past_key_value to each
attention layer. The attention layer calls:

  key_out, val_out = cache.update(layer_idx, new_k, new_v,
                                  block_tables, seq_lens, is_decode)

Which does:
  1. Write new_k / new_v into the appropriate block slots (scatter)
  2. Gather the full K/V history for each sequence from its block table
  3. Return gathered K/V for attention computation

Memory layout note:
  block_tables: List[List[int]] — one list of physical block indices per sequence.
  For sequence i with block_table = [5, 12, 3]:
    - block 5 holds tokens 0..block_size-1
    - block 12 holds tokens block_size..2*block_size-1
    - block 3 holds the most recent tokens (partially filled during decode)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class PagedKVCache:
    """
    Pre-allocated GPU KV cache in paged block format.

    Owns two tensor lists (k_cache, v_cache), one entry per layer.
    Shape per entry: [num_blocks, num_kv_heads, block_size, head_dim].

    The cache does NOT track which blocks are allocated — that is the
    BlockSpaceManager's job.  PagedKVCache only reads/writes tensor data
    at the physical block indices it is told to use.
    """

    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        num_kv_heads: int,
        block_size: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.num_blocks  = num_blocks
        self.num_layers  = num_layers
        self.num_kv_heads = num_kv_heads
        self.block_size  = block_size
        self.head_dim    = head_dim
        self.device      = device
        self.dtype       = dtype

        # Allocate both caches in one shot to fail fast if VRAM is tight.
        # Shape: [num_blocks, num_kv_heads, block_size, head_dim]
        block_shape = (num_blocks, num_kv_heads, block_size, head_dim)
        self.k_cache: List[torch.Tensor] = [
            torch.zeros(block_shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v_cache: List[torch.Tensor] = [
            torch.zeros(block_shape, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

    # ── Core method ───────────────────────────────────────────────────────────

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,            # [batch, num_kv_heads, q_len, head_dim]
        new_v: torch.Tensor,            # [batch, num_kv_heads, q_len, head_dim]
        block_tables: List[List[int]],  # [batch] → physical block indices
        seq_lens: List[int],            # number of tokens already stored (before this step)
        is_decode: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Write new_k/new_v to cache, then gather and return the full K/V
        for each sequence (prompt + all previously generated tokens + new).

        Returns:
          k_full: [batch, num_kv_heads, max_seq_len, head_dim]
          v_full: [batch, num_kv_heads, max_seq_len, head_dim]
        """
        kc = self.k_cache[layer_idx]   # [num_blocks, H, BS, D]
        vc = self.v_cache[layer_idx]

        batch_size = len(block_tables)
        bs = self.block_size

        if is_decode:
            # new_k/new_v: [batch, H, 1, D] — one new token per sequence
            for i in range(batch_size):
                total_len = seq_lens[i] + 1        # after appending this token
                last_block_idx = (total_len - 1) // bs
                slot_in_block  = (total_len - 1) % bs
                phys_block = block_tables[i][last_block_idx]
                kc[phys_block, :, slot_in_block, :] = new_k[i, :, 0, :]
                vc[phys_block, :, slot_in_block, :] = new_v[i, :, 0, :]
        else:
            # Prefill: new_k/new_v: [batch, H, prompt_len, D]
            # Each sequence is processed separately (may have different lengths)
            for i in range(batch_size):
                prompt_len = new_k.shape[2]
                for t in range(prompt_len):
                    block_idx = t // bs
                    slot      = t % bs
                    phys_block = block_tables[i][block_idx]
                    kc[phys_block, :, slot, :] = new_k[i, :, t, :]
                    vc[phys_block, :, slot, :] = new_v[i, :, t, :]

        # ── Gather full K/V for all sequences ─────────────────────────────
        new_seq_lens = [sl + (1 if is_decode else new_k.shape[2])
                        for sl in seq_lens]
        max_len = max(new_seq_lens)

        k_full = torch.zeros(
            (batch_size, self.num_kv_heads, max_len, self.head_dim),
            dtype=self.dtype, device=self.device,
        )
        v_full = torch.zeros_like(k_full)

        for i, (bt, total_len) in enumerate(zip(block_tables, new_seq_lens)):
            for t in range(total_len):
                block_idx  = t // bs
                slot       = t % bs
                phys_block = bt[block_idx]
                k_full[i, :, t, :] = kc[phys_block, :, slot, :]
                v_full[i, :, t, :] = vc[phys_block, :, slot, :]

        return k_full, v_full

    # ── Utility ───────────────────────────────────────────────────────────────

    def memory_bytes(self) -> int:
        """Total GPU bytes consumed by this cache."""
        per_layer = self.num_blocks * self.num_kv_heads * self.block_size * self.head_dim
        bytes_per_elem = 2 if self.dtype == torch.float16 else 4
        return per_layer * self.num_layers * 2 * bytes_per_elem   # ×2 for K and V

    def __repr__(self) -> str:
        mb = self.memory_bytes() / 1e6
        return (f"PagedKVCache(blocks={self.num_blocks}, layers={self.num_layers}, "
                f"kv_heads={self.num_kv_heads}, block_size={self.block_size}, "
                f"head_dim={self.head_dim}, mem={mb:.0f}MB)")
