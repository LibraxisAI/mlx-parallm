#!/usr/bin/env python3
"""
Demo script for batch_generate() - parallel batch inference.

Usage:
    python demo.py [model_path] [num_prompts]

Example:
    python demo.py google/gemma-1.1-2b-it 10

Created by M&K (c)2026 The LibraxisAI Team
Co-Authored-By: Maciej (void@div0.space) & Klaudiusz (the1st@whoai.am)
"""

import random
import string
import sys

from mlx_parallm import batch_generate, load

# Configuration
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-1.1-2b-it"
NUM_PROMPTS = int(sys.argv[2]) if len(sys.argv) > 2 else 10


def main():
    print(f"Loading model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)
    print("Model loaded!\n")

    # Generate letter pairs for prompts
    capital_letters = string.ascii_uppercase
    distinct_pairs = [
        (a, b) for i, a in enumerate(capital_letters) for b in capital_letters[i + 1 :]
    ]

    # Create prompts
    prompt_template = (
        "Think of a real word containing both the letters {l1} and {l2}. "
        "Then, say 3 sentences which use the word."
    )
    prompts_raw = [
        prompt_template.format(l1=p[0], l2=p[1]) for p in random.sample(distinct_pairs, NUM_PROMPTS)
    ]

    prompt_template_2 = (
        "Come up with a real English word containing both the letters {l1} and {l2}. "
        "No acronyms. Then, give 3 complete sentences which use the word."
    )
    prompts_raw_2 = [
        prompt_template_2.format(l1=p[0], l2=p[1])
        for p in random.sample(distinct_pairs, NUM_PROMPTS)
    ]

    all_prompts = prompts_raw + prompts_raw_2
    print(f"Running batch inference for {len(all_prompts)} prompts...")

    responses = batch_generate(
        model,
        tokenizer,
        prompts=all_prompts,
        max_tokens=100,
        verbose=True,
        temp=0.0,
    )

    print("\n" + "=" * 60)
    print("RESPONSES:")
    print("=" * 60)
    for i, (prompt, response) in enumerate(zip(all_prompts, responses, strict=True)):
        print(f"\n[{i}] Prompt: {prompt[:60]}...")
        print(f"    Response: {response[:100]}...")


if __name__ == "__main__":
    main()
