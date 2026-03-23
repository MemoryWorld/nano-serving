"""Unit tests for BlockAllocator and BlockSpaceManager."""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nano_serving.block_manager import (
    BlockAllocator, BlockSpaceManager, OutOfMemoryError
)


# ── BlockAllocator ────────────────────────────────────────────────────────────

def test_allocate_and_free():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    assert alloc.num_free_blocks == 4
    b0 = alloc.allocate()
    b1 = alloc.allocate()
    assert alloc.num_free_blocks == 2
    alloc.free(b0)
    assert alloc.num_free_blocks == 3
    alloc.free(b1)
    assert alloc.num_free_blocks == 4


def test_oom_raises():
    alloc = BlockAllocator(num_blocks=2, block_size=16)
    alloc.allocate()
    alloc.allocate()
    with pytest.raises(OutOfMemoryError):
        alloc.allocate()


def test_ref_counting():
    alloc = BlockAllocator(num_blocks=4, block_size=16)
    idx = alloc.allocate()
    alloc.fork(idx)          # ref_count = 2
    alloc.free(idx)          # ref_count = 1 — not returned yet
    assert alloc.num_free_blocks == 3
    alloc.free(idx)          # ref_count = 0 — returned
    assert alloc.num_free_blocks == 4


# ── BlockSpaceManager ─────────────────────────────────────────────────────────

def make_manager(num_blocks=64, block_size=4):
    alloc = BlockAllocator(num_blocks=num_blocks, block_size=block_size)
    return BlockSpaceManager(alloc), alloc


def test_allocate_sequence():
    mgr, alloc = make_manager(block_size=4)
    mgr.allocate(seq_id=1, num_tokens=7)   # needs ceil(7/4) = 2 blocks
    assert mgr.num_blocks_for(1) == 2
    assert alloc.num_free_blocks == 62


def test_append_token_within_block():
    mgr, alloc = make_manager(block_size=4)
    mgr.allocate(seq_id=1, num_tokens=3)   # 1 block, 3/4 filled
    mgr.append_token(1)                    # fills the block
    assert mgr.num_blocks_for(1) == 1


def test_append_token_crosses_block():
    mgr, alloc = make_manager(block_size=4)
    mgr.allocate(seq_id=1, num_tokens=4)   # 1 block, exactly full
    mgr.append_token(1)                    # needs a second block
    assert mgr.num_blocks_for(1) == 2
    assert alloc.num_free_blocks == 62


def test_free_releases_blocks():
    mgr, alloc = make_manager(block_size=4)
    mgr.allocate(seq_id=1, num_tokens=9)   # 3 blocks
    mgr.free(1)
    assert alloc.num_free_blocks == 64


def test_can_allocate_respects_budget():
    mgr, alloc = make_manager(num_blocks=2, block_size=4)
    assert mgr.can_allocate(num_tokens=8)   # exactly 2 blocks
    assert not mgr.can_allocate(num_tokens=9)   # would need 3


def test_block_table_logical_ordering():
    mgr, _ = make_manager(block_size=4)
    mgr.allocate(seq_id=5, num_tokens=12)   # 3 blocks
    bt = mgr.get_block_table(5)
    assert len(bt) == 3
    # All physical block indices should be distinct
    assert len(set(bt)) == 3
