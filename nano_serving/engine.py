"""
LLMEngine — the top-level orchestrator.

Ties together:
  BlockAllocator + BlockSpaceManager  (memory)
  Scheduler                           (batching)
  PagedKVCache                        (GPU tensors)
  ModelRunner                         (forward passes)

Public API:
  engine = LLMEngine.from_config(model_cfg, engine_cfg)
  seq    = engine.add_request(prompt_text, sampling_params)
  engine.step()   # one iteration: prefill + decode
  engine.run_until_done()   # loop until all requests finish

Each call to step():
  1. Ask the scheduler which sequences to prefill and which to decode.
  2. Run prefill for new sequences (ModelRunner.prefill).
  3. Run decode for running sequences (ModelRunner.decode).
  4. Notify the scheduler of completed tokens.
"""

from __future__ import annotations

import time
from typing import List, Optional

import torch

from nano_serving.block_manager import BlockAllocator, BlockSpaceManager
from nano_serving.cache import PagedKVCache
from nano_serving.config import EngineConfig, ModelConfig
from nano_serving.model_runner import ModelRunner
from nano_serving.scheduler import Scheduler
from nano_serving.sequence import Sequence, SamplingParams


class LLMEngine:
    def __init__(
        self,
        model_runner: ModelRunner,
        scheduler:    Scheduler,
        kv_cache:     PagedKVCache,
        tokenizer,
    ) -> None:
        self.model_runner = model_runner
        self.scheduler    = scheduler
        self.kv_cache     = kv_cache
        self.tokenizer    = tokenizer
        self._eos_id      = tokenizer.eos_token_id

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        model_cfg:  ModelConfig,
        engine_cfg: EngineConfig,
        device:     Optional[torch.device] = None,
    ) -> "LLMEngine":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Compute number of KV cache blocks from available VRAM ─────────
        num_blocks = _compute_num_blocks(model_cfg, engine_cfg, device)
        print(f"KV cache: {num_blocks} blocks "
              f"({num_blocks * engine_cfg.block_size} max tokens)")

        kv_cache = PagedKVCache(
            num_blocks   = num_blocks,
            num_layers   = model_cfg.num_layers,
            num_kv_heads = model_cfg.num_kv_heads,
            block_size   = engine_cfg.block_size,
            head_dim     = model_cfg.head_dim,
            device       = device,
            dtype        = torch.float16,
        )

        allocator     = BlockAllocator(num_blocks, engine_cfg.block_size)
        block_manager = BlockSpaceManager(allocator)
        scheduler     = Scheduler(
            block_manager,
            max_num_seqs           = engine_cfg.max_num_seqs,
            max_num_batched_tokens = engine_cfg.max_num_batched_tokens,
        )

        model_runner = ModelRunner(model_cfg, engine_cfg, kv_cache, device)

        return cls(model_runner, scheduler, kv_cache, model_runner.tokenizer)

    # ── Request interface ─────────────────────────────────────────────────────

    def add_request(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> Sequence:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if self._eos_id is not None:
            sampling_params.stop_token_ids = list(
                set(sampling_params.stop_token_ids) | {self._eos_id}
            )
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        seq = Sequence(token_ids, sampling_params)
        self.scheduler.add(seq)
        return seq

    # ── Stepping ──────────────────────────────────────────────────────────────

    def step(self) -> bool:
        """
        Execute one engine iteration (prefill + decode).
        Returns True if there is still work to do.
        """
        batch = self.scheduler.schedule()
        if batch.is_empty:
            return not self.scheduler.is_idle

        # ── Prefill ───────────────────────────────────────────────────────
        if batch.prefill_seqs:
            token_ids_list = [s.prompt_token_ids for s in batch.prefill_seqs]
            block_tables   = [s.block_table       for s in batch.prefill_seqs]
            seq_lens       = [0] * len(batch.prefill_seqs)

            first_tokens = self.model_runner.prefill(
                token_ids_list, block_tables, seq_lens
            )
            self.scheduler.on_prefill_complete(batch.prefill_seqs, first_tokens)

        # ── Decode ────────────────────────────────────────────────────────
        if batch.decode_seqs:
            last_tokens  = [s.last_token_id         for s in batch.decode_seqs]
            block_tables = [s.block_table            for s in batch.decode_seqs]
            # num_total_tokens includes the last generated token whose KV hasn't
            # been written to cache yet — cache holds (total - 1) tokens.
            seq_lens     = [s.num_total_tokens - 1   for s in batch.decode_seqs]

            next_tokens = self.model_runner.decode(
                last_tokens, block_tables, seq_lens
            )
            self.scheduler.on_step_complete(batch.decode_seqs, next_tokens)

        return not self.scheduler.is_idle

    def run_until_done(self) -> None:
        """Drive the engine until all pending requests are complete."""
        while self.step():
            pass

    # ── Decoding helper ───────────────────────────────────────────────────────

    def decode_tokens(self, seq: Sequence) -> str:
        return self.tokenizer.decode(
            seq.output_token_ids, skip_special_tokens=True
        )


# ── Helper: compute num_blocks from free VRAM ─────────────────────────────────

def _compute_num_blocks(
    model_cfg:  ModelConfig,
    engine_cfg: EngineConfig,
    device:     torch.device,
) -> int:
    if engine_cfg.max_num_blocks is not None:
        return engine_cfg.max_num_blocks

    # Measure free VRAM after a dummy forward pass to account for model weights
    # and activation memory.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        usable = int(free_bytes * engine_cfg.gpu_memory_utilization)
    else:
        usable = 4 * 1024**3   # 4 GB fallback for CPU testing

    # Bytes per physical block:
    # block_size × num_kv_heads × head_dim × 2 (K+V) × num_layers × dtype_bytes
    dtype_bytes = 2  # fp16
    bytes_per_block = (
        engine_cfg.block_size
        * model_cfg.num_kv_heads
        * model_cfg.head_dim
        * 2              # K and V
        * model_cfg.num_layers
        * dtype_bytes
    )
    num_blocks = max(1, usable // bytes_per_block)
    return num_blocks
