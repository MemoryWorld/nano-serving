"""
Engine and model configuration.

Two config objects keep concerns separate:
  ModelConfig  — what the model looks like (architecture params)
  EngineConfig — how we serve it (block size, memory budget, batch limits)
"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    num_layers: int = 28
    num_kv_heads: int = 4
    head_dim: int = 128
    dtype: str = "float16"         # KV cache dtype
    max_model_len: int = 8192      # max sequence length the model supports


@dataclass
class EngineConfig:
    # PagedAttention block parameters
    block_size: int = 16           # tokens per KV cache block

    # Memory budget: fraction of GPU VRAM reserved for KV cache blocks.
    # The engine computes num_blocks from free VRAM × gpu_memory_utilization.
    gpu_memory_utilization: float = 0.85

    # Hard upper bound on number of blocks regardless of VRAM.
    # Set to None to compute dynamically at engine init.
    max_num_blocks: int | None = None

    # Scheduler limits
    max_num_seqs: int = 256        # max sequences in flight at once
    max_num_batched_tokens: int = 4096   # max tokens in one prefill+decode step

    @property
    def bytes_per_block(self) -> int:
        """KV cache bytes consumed by one physical block (all layers, K+V)."""
        from nano_serving.config import ModelConfig
        # This will be patched at runtime; provided here for documentation.
        # bytes = block_size × num_kv_heads × head_dim × 2 (K+V) × num_layers × dtype_bytes
        raise NotImplementedError("call engine.block_bytes_per_block() instead")
