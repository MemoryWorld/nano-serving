"""
Core data structures: Sequence and SamplingParams.

A Sequence is one request's lifetime — from prompt tokens through
generated tokens to completion.  The engine and scheduler only talk
about Sequence objects; they never touch the model directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class SequenceStatus(Enum):
    WAITING   = auto()   # in scheduler's waiting queue, not yet started
    RUNNING   = auto()   # actively being processed (prefill or decode)
    FINISHED  = auto()   # EOS reached or max_tokens hit
    ABORTED   = auto()   # cancelled by caller


@dataclass
class SamplingParams:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1               # -1 = disabled
    stop_token_ids: List[int] = field(default_factory=list)


class Sequence:
    """
    One request = one Sequence.

    Lifecycle:
      WAITING → RUNNING (prefill) → RUNNING (decode loop) → FINISHED

    The block_table is managed by BlockSpaceManager and holds the
    list of physical block indices allocated to this sequence.
    After prefill, len(block_table) * block_size >= num_prompt_tokens.
    """

    _id_counter = 0

    def __init__(
        self,
        prompt_token_ids: List[int],
        sampling_params: SamplingParams,
    ):
        Sequence._id_counter += 1
        self.seq_id: int = Sequence._id_counter

        self.prompt_token_ids: List[int] = list(prompt_token_ids)
        self.output_token_ids: List[int] = []
        self.sampling_params = sampling_params

        self.status: SequenceStatus = SequenceStatus.WAITING

        # Set by BlockSpaceManager after blocks are allocated
        self.block_table: List[int] = []

        # Timestamps for TTFT / ITL measurement
        self.arrival_time: float = time.monotonic()
        self.first_token_time: Optional[float] = None
        self.finish_time: Optional[float] = None

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_generated_tokens

    @property
    def all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def last_token_id(self) -> int:
        return self.all_token_ids[-1]

    # ── State transitions ─────────────────────────────────────────────────────

    def is_prefill_done(self) -> bool:
        """True after the first forward pass (prompt processed)."""
        return self.first_token_time is not None

    def append_token(self, token_id: int) -> None:
        if self.first_token_time is None:
            self.first_token_time = time.monotonic()
        self.output_token_ids.append(token_id)

    def is_finished(self) -> bool:
        if self.status in (SequenceStatus.FINISHED, SequenceStatus.ABORTED):
            return True
        sp = self.sampling_params
        if self.num_generated_tokens >= sp.max_new_tokens:
            return True
        if sp.stop_token_ids and self.last_token_id in sp.stop_token_ids:
            return True
        return False

    def finish(self) -> None:
        self.status = SequenceStatus.FINISHED
        self.finish_time = time.monotonic()

    # ── Metrics ───────────────────────────────────────────────────────────────

    @property
    def ttft(self) -> Optional[float]:
        """Time to first token (seconds)."""
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def total_latency(self) -> Optional[float]:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    def __repr__(self) -> str:
        return (f"Seq(id={self.seq_id}, prompt_len={self.num_prompt_tokens}, "
                f"gen={self.num_generated_tokens}, status={self.status.name})")
