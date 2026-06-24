"""
ModelRunner — loads Qwen2.5-7B and executes prefill / decode steps
using the PagedKVCache.

Design:
  We monkey-patch each Qwen2Attention layer's forward to intercept the
  K/V computation and redirect it through PagedKVCache.update().
  The rest of the model (embeddings, MLP, layernorm) runs unchanged.

Why monkey-patch instead of subclassing?
  HuggingFace's Qwen2 model calls its own attention class directly;
  injecting via a custom model subclass would require copying large
  amounts of boilerplate.  A forward-hook patch is surgical and explicit.

The patched forward:
  1. Computes Q, K, V projections and applies RoPE (unchanged).
  2. Calls paged_kv_cache.update() to write new K/V to blocks and
     gather the full history.
  3. Runs standard SDPA attention on the gathered tensors.
  4. Projects output and returns — compatible with the original signature.

Thread-safety: not thread-safe (single-threaded engine loop).
"""

from __future__ import annotations

import math
import types
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from nano_serving.cache import PagedKVCache
from nano_serving.config import EngineConfig, ModelConfig


# Context object threaded through one forward pass so the patched layers
# know which sequences they're operating on.
class _ForwardCtx:
    def __init__(
        self,
        kv_cache: PagedKVCache,
        block_tables: List[List[int]],
        seq_lens: List[int],
        is_decode: bool,
    ):
        self.kv_cache     = kv_cache
        self.block_tables = block_tables
        self.seq_lens     = seq_lens
        self.is_decode    = is_decode


# Module-level slot for the current forward context.
# Set by ModelRunner before each forward pass, cleared after.
_current_ctx: Optional[_ForwardCtx] = None


def _make_patched_forward(original_forward, layer_idx: int):
    """
    Return a new forward function compatible with the transformers ≥4.45
    Qwen2Attention API:
      - position_embeddings=(cos, sin) is passed in by the decoder layer
      - attributes live on self_attn.config, not directly on self_attn
      - returns (attn_output, attn_weights) — 2 values
    """

    def patched_forward(
        self_attn,
        hidden_states: torch.Tensor,
        position_embeddings,            # (cos, sin) — required in new API
        attention_mask=None,
        past_key_values=None,           # new name (was past_key_value)
        cache_position=None,
        **kwargs,
    ):
        global _current_ctx
        ctx = _current_ctx

        bsz, q_len, _ = hidden_states.size()

        num_q_heads  = self_attn.config.num_attention_heads
        num_kv_heads = self_attn.config.num_key_value_heads
        head_dim     = self_attn.head_dim

        # ── Q / K / V projections ─────────────────────────────────────────
        query = self_attn.q_proj(hidden_states).view(bsz, q_len, num_q_heads,  head_dim).transpose(1, 2)
        key   = self_attn.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value = self_attn.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # ── RoPE (cos/sin already computed by the model's rotary emb layer) ─
        cos, sin = position_embeddings
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # ── PagedKVCache update ───────────────────────────────────────────
        if ctx is not None:
            key_full, value_full = ctx.kv_cache.update(
                layer_idx,
                key, value,
                ctx.block_tables,
                ctx.seq_lens,
                ctx.is_decode,
            )
        else:
            key_full, value_full = key, value

        # ── Repeat KV heads for GQA ───────────────────────────────────────
        if num_q_heads != num_kv_heads:
            n_rep = num_q_heads // num_kv_heads
            key_full   = key_full.repeat_interleave(n_rep, dim=1)
            value_full = value_full.repeat_interleave(n_rep, dim=1)

        # ── Scaled dot-product attention ──────────────────────────────────
        attn_out = F.scaled_dot_product_attention(
            query, key_full, value_full,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=(q_len > 1),   # causal for prefill; no mask needed for single decode token
        )

        # ── Output projection ─────────────────────────────────────────────
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_out = self_attn.o_proj(attn_out)

        return attn_out, None   # (hidden_states, attn_weights)  — 2 values per new API

    return patched_forward


class ModelRunner:
    """
    Loads Qwen2.5-7B and runs prefill / decode steps with PagedKVCache.

    After loading, the attention layers are patched once.  Every forward
    pass sets _current_ctx with the current batch's paging information
    and clears it when done.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        engine_cfg: EngineConfig,
        kv_cache: PagedKVCache,
        device: torch.device,
    ) -> None:
        self.model_cfg  = model_cfg
        self.engine_cfg = engine_cfg
        self.kv_cache   = kv_cache
        self.device     = device

        print(f"Loading {model_cfg.model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_name,
            dtype=torch.float16,
            device_map={"": device},
            trust_remote_code=True,
        )
        self.model.eval()
        print("Model loaded.")

        self._patch_attention_layers()

    # ── Patching ──────────────────────────────────────────────────────────────

    def _patch_attention_layers(self) -> None:
        """Replace each attention layer's forward with the paged version."""
        layers = self.model.model.layers
        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            original = attn.forward
            attn.forward = types.MethodType(
                _make_patched_forward(original, layer_idx), attn
            )
        print(f"Patched {len(layers)} attention layers for PagedKVCache.")

    # ── Prefill ───────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def prefill(
        self,
        token_ids_list: List[List[int]],    # one prompt per sequence
        block_tables:   List[List[int]],
        seq_lens:       List[int],           # all zeros for brand-new seqs
    ) -> List[int]:
        """
        Run one prefill step for a batch of sequences.
        Returns the first generated token id for each sequence.
        """
        global _current_ctx

        # For simplicity in v1: process one sequence at a time during prefill.
        # (Batched prefill with variable-length sequences requires padding or
        # flash-attention with varlen interface — a v2 addition.)
        first_tokens = []
        for i, token_ids in enumerate(token_ids_list):
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            pos_ids   = torch.arange(len(token_ids), device=self.device).unsqueeze(0)

            _current_ctx = _ForwardCtx(
                kv_cache    = self.kv_cache,
                block_tables= [block_tables[i]],
                seq_lens    = [0],               # nothing in cache yet
                is_decode   = False,
            )
            try:
                out = self.model(
                    input_ids=input_ids,
                    position_ids=pos_ids,
                    use_cache=False,   # we handle caching ourselves
                )
            finally:
                _current_ctx = None

            # Greedy sample from last token logits
            logits  = out.logits[0, -1, :]       # [vocab]
            next_id = int(logits.argmax().item())
            first_tokens.append(next_id)

        return first_tokens

    # ── Decode ────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def decode(
        self,
        last_token_ids: List[int],       # most recent token for each sequence
        block_tables:   List[List[int]],
        seq_lens:       List[int],       # number of tokens already in KV cache
    ) -> List[int]:
        """
        Run one decode step for a batch of sequences.
        Each sequence provides exactly one new token.
        Returns the next token id for each sequence.
        """
        global _current_ctx

        batch_size = len(last_token_ids)
        input_ids = torch.tensor(
            [[t] for t in last_token_ids], dtype=torch.long, device=self.device
        )   # [batch, 1]

        # Position id = current length (0-indexed position of the new token)
        pos_ids = torch.tensor(
            [[sl] for sl in seq_lens], dtype=torch.long, device=self.device
        )   # [batch, 1]

        _current_ctx = _ForwardCtx(
            kv_cache    = self.kv_cache,
            block_tables= block_tables,
            seq_lens    = seq_lens,
            is_decode   = True,
        )
        try:
            out = self.model(
                input_ids=input_ids,
                position_ids=pos_ids,
                use_cache=False,
            )
        finally:
            _current_ctx = None

        # Greedy sample
        logits  = out.logits[:, -1, :]           # [batch, vocab]
        next_ids = logits.argmax(dim=-1).tolist()
        return next_ids
