"""
Continuous Batching Scheduler.

The key idea (Yu et al., "Orca", OSDI 2022):
  Traditional serving: wait until a request finishes before starting the next.
  Continuous batching: iteration-level scheduling — each forward pass step
  decides independently which sequences to include in the next batch.

This lets new requests join a batch as soon as a slot opens up,
without waiting for long-running requests to finish first.

Scheduler logic (one call to `schedule()` per engine step):

  Step 1 — Decode phase:
    All RUNNING sequences that completed prefill join the decode batch.
    For each, try to append one more block if needed (new decode token slot).
    Sequences that can't get a block are *preempted* (dropped to waiting, v1).

  Step 2 — Prefill phase:
    Promote sequences from the waiting queue into RUNNING, allocating blocks
    for their full prompt, as long as:
      (a) we haven't exceeded max_num_seqs
      (b) we haven't exceeded max_num_batched_tokens for this step
      (c) the allocator has enough free blocks

The output of schedule() is a SchedulerOutput that the engine turns into
a forward pass.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from nano_serving.block_manager import BlockSpaceManager, OutOfMemoryError
from nano_serving.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    """
    What the engine should do on this iteration.

    prefill_seqs:  sequences whose prompts have not been processed yet.
                   The engine runs a forward pass over their full prompt.
    decode_seqs:   sequences that already have KV cache filled and need
                   one more token generated.
    """
    prefill_seqs: List[Sequence] = field(default_factory=list)
    decode_seqs:  List[Sequence] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.prefill_seqs and not self.decode_seqs

    @property
    def num_prefill_tokens(self) -> int:
        return sum(s.num_prompt_tokens for s in self.prefill_seqs)

    @property
    def num_decode_tokens(self) -> int:
        return len(self.decode_seqs)

    @property
    def total_tokens(self) -> int:
        return self.num_prefill_tokens + self.num_decode_tokens


class Scheduler:
    """
    FCFS continuous batching scheduler.

    Internally maintains two collections:
      _waiting  — requests not yet started (FIFO deque)
      _running  — requests with allocated KV cache, generating tokens

    `schedule()` is called once per engine step and returns a SchedulerOutput.
    The engine calls `on_step_complete()` after executing a step to update
    sequence states and release blocks for finished sequences.
    """

    def __init__(
        self,
        block_manager: BlockSpaceManager,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 4096,
    ) -> None:
        self.block_manager = block_manager
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens

        self._waiting: Deque[Sequence] = deque()
        self._running: List[Sequence] = []

    # ── Public interface ──────────────────────────────────────────────────────

    def add(self, seq: Sequence) -> None:
        """Enqueue a new request."""
        self._waiting.append(seq)

    def schedule(self) -> SchedulerOutput:
        """
        Produce the next batch.

        Order matters: decode goes first so that in-flight sequences aren't
        starved by a flood of new prefills.
        """
        out = SchedulerOutput()
        budgeted_tokens = 0

        # ── Phase 1: schedule decode for all running sequences ─────────────
        still_running: List[Sequence] = []
        for seq in self._running:
            if not self.block_manager.can_append_token(seq.seq_id):
                # No block space — preempt: put back in waiting, free blocks.
                # v1: simple drop. v2 would swap KV to CPU.
                self.block_manager.free(seq.seq_id)
                seq.status = SequenceStatus.WAITING
                self._waiting.appendleft(seq)   # re-queue with priority
            else:
                self.block_manager.append_token(seq.seq_id)
                out.decode_seqs.append(seq)
                budgeted_tokens += 1
                still_running.append(seq)

        self._running = still_running

        # ── Phase 2: promote waiting sequences into prefill batch ──────────
        while self._waiting:
            if len(self._running) + len(out.prefill_seqs) >= self.max_num_seqs:
                break   # batch is full

            seq = self._waiting[0]
            prompt_tokens = seq.num_prompt_tokens

            if budgeted_tokens + prompt_tokens > self.max_num_batched_tokens:
                break   # token budget exhausted for this step

            if not self.block_manager.can_allocate(prompt_tokens):
                break   # not enough KV cache blocks

            self._waiting.popleft()
            self.block_manager.allocate(seq.seq_id, prompt_tokens)
            seq.status = SequenceStatus.RUNNING
            out.prefill_seqs.append(seq)
            budgeted_tokens += prompt_tokens

        return out

    def on_step_complete(
        self,
        decode_seqs: List[Sequence],
        new_token_ids: List[int],
    ) -> None:
        """
        Called by the engine after each decode step.
        Appends generated tokens and frees blocks for finished sequences.
        """
        finished: List[Sequence] = []
        for seq, token_id in zip(decode_seqs, new_token_ids):
            seq.append_token(token_id)
            if seq.is_finished():
                seq.finish()
                self.block_manager.free(seq.seq_id)
                finished.append(seq)

        for seq in finished:
            if seq in self._running:
                self._running.remove(seq)

    def on_prefill_complete(
        self,
        prefill_seqs: List[Sequence],
        first_token_ids: List[int],
    ) -> None:
        """
        Called by the engine after a prefill step.
        The first generated token is recorded and sequences move to running.
        """
        for seq, token_id in zip(prefill_seqs, first_token_ids):
            seq.append_token(token_id)
            if seq.is_finished():
                seq.finish()
                self.block_manager.free(seq.seq_id)
            else:
                self._running.append(seq)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def num_waiting(self) -> int:
        return len(self._waiting)

    @property
    def num_running(self) -> int:
        return len(self._running)

    @property
    def is_idle(self) -> bool:
        return self.num_waiting == 0 and self.num_running == 0

    def __repr__(self) -> str:
        return f"Scheduler(waiting={self.num_waiting}, running={self.num_running})"
