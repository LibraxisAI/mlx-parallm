#!/usr/bin/env python3
"""
Demo script for batch_generate_stream() - parallel streaming inference.

Usage:
    python demo_streaming.py [model_path] [num_users]

Example:
    python demo_streaming.py mlx-community/Phi-3-mini-4k-instruct-4bit 5

Created by M&K (c)2026 The LibraxisAI Team
Co-Authored-By: Maciej (void@div0.space) & Klaudiusz (the1st@whoai.am)
"""

import sys
import time
from collections import defaultdict

from mlx_parallm import batch_generate_stream, load

# ANSI colors for different users
COLORS = [
    "\033[91m",  # Red
    "\033[92m",  # Green
    "\033[93m",  # Yellow
    "\033[94m",  # Blue
    "\033[95m",  # Magenta
    "\033[96m",  # Cyan
    "\033[97m",  # White
]
RESET = "\033[0m"


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/Phi-3-mini-4k-instruct-4bit"
    num_users = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path)
    print("Model loaded!\n")

    # Sample prompts for multiple users
    sample_prompts = [
        "What is the capital of France?",
        "Count from 1 to 5.",
        "What is 2 + 2?",
        "Say hello in Spanish.",
        "What color is the sky?",
        "Name a famous scientist.",
        "What is Python?",
    ]

    prompts = sample_prompts[:num_users]
    print(f"Running batch inference for {len(prompts)} concurrent users:")
    for i, p in enumerate(prompts):
        color = COLORS[i % len(COLORS)]
        print(f"  {color}User {i}{RESET}: {p}")
    print()

    # Track outputs per user
    user_outputs: dict[int, list[str]] = defaultdict(list)
    user_first_token_time: dict[int, float] = {}
    user_done_time: dict[int, float] = {}

    start_time = time.perf_counter()

    # Stream tokens
    print("=" * 60)
    print("STREAMING OUTPUT (tokens appear as generated):")
    print("=" * 60)

    for user_idx, token, done in batch_generate_stream(
        model, tokenizer, prompts, max_tokens=100, temp=0.7
    ):
        color = COLORS[user_idx % len(COLORS)]

        # Track first token time
        if user_idx not in user_first_token_time:
            user_first_token_time[user_idx] = time.perf_counter() - start_time

        user_outputs[user_idx].append(token)

        # Print with color
        if done:
            print(f"{color}[User {user_idx} DONE]{RESET}")
            user_done_time[user_idx] = time.perf_counter() - start_time
        else:
            print(f"{color}[{user_idx}]{RESET} {repr(token)}", end=" ", flush=True)

    total_time = time.perf_counter() - start_time
    print("\n" + "=" * 60)

    # Summary
    print("\nSUMMARY:")
    print("-" * 60)

    total_tokens = 0
    for user_idx in range(len(prompts)):
        tokens = user_outputs[user_idx]
        total_tokens += len(tokens)
        response = "".join(tokens)
        ttft = user_first_token_time.get(user_idx, 0)
        done_t = user_done_time.get(user_idx, total_time)

        color = COLORS[user_idx % len(COLORS)]
        print(f"\n{color}User {user_idx}{RESET}:")
        print(f"  Prompt: {prompts[user_idx]}")
        print(f"  Response: {response[:100]}{'...' if len(response) > 100 else ''}")
        print(f"  Tokens: {len(tokens)}")
        print(f"  TTFT: {ttft * 1000:.1f}ms")
        print(f"  Total: {done_t:.2f}s")

    print("\n" + "-" * 60)
    print(f"TOTAL TIME: {total_time:.2f}s")
    print(f"TOTAL TOKENS: {total_tokens}")
    print(f"THROUGHPUT: {total_tokens / total_time:.1f} tok/s (combined)")
    print(
        f"AVG TTFT: {sum(user_first_token_time.values()) / len(user_first_token_time) * 1000:.1f}ms"
    )


if __name__ == "__main__":
    main()
