# Nano Serving

Nano Serving is a compact LLM serving scaffold for studying the core pieces behind paged KV-cache serving and continuous batching.

It is intentionally small: the repository is useful for reading, testing, and experimenting with serving internals without the operational surface area of a production inference server.

## What It Includes

- Paged KV-cache block allocation and block-table management.
- Continuous batching scheduler for prefill and decode steps.
- Request/sequence state tracking with sampling parameters and stop-token handling.
- Qwen2-style model runner wiring through PyTorch and Transformers.
- Unit tests for block allocation, block-space management, scheduling, token budgets, and preemption behavior.
- A lightweight throughput benchmark entrypoint for local experiments.

## Repository Layout

```text
nano_serving/
  block_manager.py   # physical block allocation and per-sequence block tables
  cache.py           # paged KV-cache tensor storage
  config.py          # model and engine configuration
  engine.py          # top-level request, scheduling, prefill, decode loop
  model_runner.py    # model/tokenizer integration
  scheduler.py       # continuous batching scheduler
  sequence.py        # request and sequence state
benchmarks/
  bench_throughput.py
tests/
  test_block_manager.py
  test_scheduler.py
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

The default model config targets a Qwen2.5-style 7B model, but the unit tests exercise the scheduler and block manager without requiring a GPU model download.

## Status

This is a learning and prototype project. It is not a production inference server, a vLLM replacement, or a benchmark claim. Use it to discuss serving architecture, KV-cache memory planning, scheduling behavior, and failure modes in a small codebase.
