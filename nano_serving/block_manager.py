"""
PagedAttention Block Allocator.

The core insight of PagedAttention (Kwon et al., 2023):
  Instead of pre-allocating a contiguous KV cache per request (which
  wastes memory due to internal fragmentation when requests finish early),
  divide the KV cache into fixed-size *physical blocks* and assign them
  to sequences on demand via a *block table* (logical → physical mapping).

Benefits:
  1. Near-zero internal fragmentation  (waste ≤ block_size - 1 tokens per seq)
  2. Near-zero external fragmentation  (free blocks go back to a pool)
  3. Enables prefix caching and copy-on-write for beam search

This file implements the memory management layer — it never touches the GPU.
The GPU KV cache tensors are allocated in ModelRunner; BlockSpaceManager
just tracks which physical block indices are free and which belong to whom.

Key data structures:
  BlockAllocator    — a free-list of physical block indices
  BlockSpaceManager — per-sequence block tables built on top of the allocator
"""

from __future__ import annotations

from typing import Dict, List, Optional


class OutOfMemoryError(RuntimeError):
    """Raised when no free KV cache blocks remain."""


class BlockAllocator:
    """
    Manages a pool of num_blocks physical KV cache blocks.

    Each block can hold block_size tokens of KV cache across all layers.
    Reference counting allows future prefix sharing (not yet used by the
    scheduler in v1, but the infrastructure is in place).

    Thread-safety: not thread-safe; the engine runs a single async loop.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size

        # Free blocks are kept as a stack (LIFO) for cache locality:
        # recently freed blocks are more likely still warm in GPU L2.
        self._free: List[int] = list(range(num_blocks))
        self._ref_counts: List[int] = [0] * num_blocks

    # ── Allocation ────────────────────────────────────────────────────────────

    def allocate(self) -> int:
        """Allocate one physical block. Returns its index."""
        if not self._free:
            raise OutOfMemoryError(
                f"KV cache OOM: 0 free blocks out of {self.num_blocks}"
            )
        idx = self._free.pop()
        self._ref_counts[idx] = 1
        return idx

    def free(self, block_idx: int) -> None:
        """Decrement ref count; return block to pool when it reaches 0."""
        assert self._ref_counts[block_idx] > 0, f"double-free of block {block_idx}"
        self._ref_counts[block_idx] -= 1
        if self._ref_counts[block_idx] == 0:
            self._free.append(block_idx)

    def fork(self, block_idx: int) -> None:
        """Increment ref count (for copy-on-write / prefix sharing)."""
        self._ref_counts[block_idx] += 1

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks

    def can_allocate(self, n: int = 1) -> bool:
        return len(self._free) >= n

    def __repr__(self) -> str:
        return (f"BlockAllocator(total={self.num_blocks}, "
                f"free={self.num_free_blocks}, used={self.num_used_blocks})")


class BlockSpaceManager:
    """
    Manages per-sequence block tables on top of a BlockAllocator.

    A block table is the logical→physical mapping for one sequence:
      block_table[logical_idx] = physical_block_idx

    The last block of a sequence is typically partially filled.
    num_filled_in_last_block tracks how many token slots are used
    in the last physical block.

    Terminology matching the vLLM paper:
      logical block  — index within a sequence's own address space
      physical block — index into the shared GPU KV cache tensor
    """

    def __init__(self, allocator: BlockAllocator) -> None:
        self.allocator = allocator
        # seq_id → list of physical block indices
        self._block_tables: Dict[int, List[int]] = {}
        # seq_id → number of tokens stored in the last (partial) block
        self._last_block_fill: Dict[int, int] = {}

    # ── Sequence lifecycle ────────────────────────────────────────────────────

    def can_allocate(self, num_tokens: int) -> bool:
        """True if we have enough free blocks for a new sequence with num_tokens."""
        needed = self._blocks_needed(num_tokens)
        return self.allocator.can_allocate(needed)

    def allocate(self, seq_id: int, num_tokens: int) -> None:
        """
        Allocate blocks for a brand-new sequence.
        Call this once when a sequence transitions WAITING → RUNNING.
        """
        assert seq_id not in self._block_tables, f"seq {seq_id} already allocated"
        needed = self._blocks_needed(num_tokens)
        blocks = [self.allocator.allocate() for _ in range(needed)]
        self._block_tables[seq_id] = blocks
        fill_in_last = num_tokens % self.allocator.block_size
        self._last_block_fill[seq_id] = fill_in_last if fill_in_last != 0 else self.allocator.block_size

    def can_append_token(self, seq_id: int) -> bool:
        """
        True if we can store one more token for this sequence.
        Returns False only when the last block is full AND the allocator is OOM.
        """
        if self._last_block_fill[seq_id] < self.allocator.block_size:
            return True                        # still room in current block
        return self.allocator.can_allocate(1)  # need a new block

    def append_token(self, seq_id: int) -> None:
        """
        Record that one more token has been generated for seq_id.
        Allocates a new physical block if the current one is full.
        """
        fill = self._last_block_fill[seq_id]
        if fill == self.allocator.block_size:
            # Current block is full — allocate a new one
            new_block = self.allocator.allocate()
            self._block_tables[seq_id].append(new_block)
            self._last_block_fill[seq_id] = 1
        else:
            self._last_block_fill[seq_id] += 1

    def free(self, seq_id: int) -> None:
        """Return all blocks for seq_id to the allocator."""
        for block_idx in self._block_tables.pop(seq_id, []):
            self.allocator.free(block_idx)
        self._last_block_fill.pop(seq_id, None)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_block_table(self, seq_id: int) -> List[int]:
        """Physical block indices for seq_id, in logical order."""
        return list(self._block_tables[seq_id])

    def num_blocks_for(self, seq_id: int) -> int:
        return len(self._block_tables.get(seq_id, []))

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _blocks_needed(self, num_tokens: int) -> int:
        bs = self.allocator.block_size
        return (num_tokens + bs - 1) // bs

    def __repr__(self) -> str:
        return (f"BlockSpaceManager(seqs={len(self._block_tables)}, "
                f"allocator={self.allocator})")
