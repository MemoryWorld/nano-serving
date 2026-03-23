"""
Throughput benchmark: nano-serving continuous batching vs naive sequential.

Metric:  tokens/second  (output tokens only, not counting prompt)
Setup:   N synthetic requests, each with a fixed prompt + max_new_tokens=128
         Vary concurrency level to show how batching helps.

Naive baseline:
  Process one request at a time with HuggingFace model.generate().
  Represents what you'd get without a serving framework.

nano-serving:
  All requests submitted at once; engine runs continuous batching.
  Multiple sequences decode in parallel each step.

Run:
  cd ~/nano-serving
  python benchmarks/bench_throughput.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from nano_serving import LLMEngine, ModelConfig, EngineConfig, SamplingParams


# ── Synthetic workload ─────────────────────────────────────────────────────────

PROMPTS = [
    "Explain the key ideas behind the transformer architecture in detail.",
    "What are the main differences between supervised and unsupervised learning?",
    "Describe how gradient descent works and why momentum helps.",
    "What is the vanishing gradient problem and how does batch normalization help?",
    "Explain the attention mechanism and why it replaced RNNs for NLP tasks.",
    "What are the trade-offs between FP16 and INT8 quantization for inference?",
    "How does KV cache work and why is it essential for autoregressive decoding?",
    "Explain PagedAttention and how it reduces memory fragmentation.",
    "What is speculative decoding and when does it improve throughput?",
    "Describe the roofline model and what it tells you about kernel optimization.",
]


# ── Naive baseline ─────────────────────────────────────────────────────────────

def run_naive(model, tokenizer, prompts: list[str], max_new_tokens: int, device) -> dict:
    """
    Process each request one at a time using HuggingFace generate().
    No batching — this is the baseline a developer would write naively.
    """
    total_output_tokens = 0
    t0 = time.perf_counter()

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        total_output_tokens += out.shape[1] - inputs["input_ids"].shape[1]

    elapsed = time.perf_counter() - t0
    return {
        "total_output_tokens": total_output_tokens,
        "elapsed_s":           elapsed,
        "throughput_tps":      total_output_tokens / elapsed,
        "num_requests":        len(prompts),
    }


# ── nano-serving ──────────────────────────────────────────────────────────────

def run_nano_serving(engine: LLMEngine, prompts: list[str], max_new_tokens: int) -> dict:
    """
    Submit all requests to the engine at once; run until all complete.
    """
    seqs = []
    sp   = SamplingParams(max_new_tokens=max_new_tokens)

    t0 = time.perf_counter()
    for prompt in prompts:
        seq = engine.add_request(prompt, SamplingParams(max_new_tokens=max_new_tokens))
        seqs.append(seq)

    engine.run_until_done()
    elapsed = time.perf_counter() - t0

    total_output = sum(s.num_generated_tokens for s in seqs)
    ttfts = [s.ttft for s in seqs if s.ttft is not None]

    return {
        "total_output_tokens": total_output,
        "elapsed_s":           elapsed,
        "throughput_tps":      total_output / elapsed,
        "num_requests":        len(prompts),
        "mean_ttft_ms":        (sum(ttfts) / len(ttfts) * 1000) if ttfts else None,
        "p99_ttft_ms":         (sorted(ttfts)[int(0.99 * len(ttfts))] * 1000) if ttfts else None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-requests",  type=int, default=10)
    parser.add_argument("--max-new-tokens",type=int, default=128)
    parser.add_argument("--skip-naive",    action="store_true",
                        help="Skip the naive baseline (saves time)")
    args = parser.parse_args()

    device = torch.device("cuda")

    # Repeat / trim prompt list to match requested count
    prompts = (PROMPTS * ((args.num_requests // len(PROMPTS)) + 1))[:args.num_requests]

    print(f"\n{'='*60}")
    print(f"Throughput benchmark")
    print(f"  requests={args.num_requests}  max_new_tokens={args.max_new_tokens}")
    print(f"{'='*60}\n")

    results = {}

    # ── nano-serving ──────────────────────────────────────────────────────
    print("[1/2] nano-serving (continuous batching + PagedKVCache)")
    model_cfg  = ModelConfig()
    engine_cfg = EngineConfig(
        block_size             = 16,
        gpu_memory_utilization = 0.80,   # leave headroom for model activations
        max_num_seqs           = 64,
        max_num_batched_tokens = 2048,
    )
    engine = LLMEngine.from_config(model_cfg, engine_cfg, device)

    r_nano = run_nano_serving(engine, prompts, args.max_new_tokens)
    results["nano_serving"] = r_nano
    print(f"  Throughput:  {r_nano['throughput_tps']:.1f} tokens/s")
    print(f"  Total time:  {r_nano['elapsed_s']:.1f}s")
    if r_nano.get("mean_ttft_ms"):
        print(f"  Mean TTFT:   {r_nano['mean_ttft_ms']:.0f}ms")

    # ── Naive ─────────────────────────────────────────────────────────────
    if not args.skip_naive:
        print("\n[2/2] Naive sequential (HuggingFace generate, one at a time)")
        # Reuse the already-loaded model from the engine
        r_naive = run_naive(
            engine.model_runner.model,
            engine.model_runner.tokenizer,
            prompts,
            args.max_new_tokens,
            device,
        )
        results["naive"] = r_naive
        print(f"  Throughput:  {r_naive['throughput_tps']:.1f} tokens/s")
        print(f"  Total time:  {r_naive['elapsed_s']:.1f}s")

        speedup = r_nano["throughput_tps"] / r_naive["throughput_tps"]
        results["speedup"] = speedup
        print(f"\n  nano-serving speedup: {speedup:.2f}x")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = Path(__file__).parent.parent / "results" / "bench_throughput.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
