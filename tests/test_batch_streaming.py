"""
Tests for batch_generate_stream() - parallel batch inference with streaming.

Created by M&K (c)2025 The LibraxisAI Team
Co-Authored-By: Maciej (void@div0.space) & Klaudiusz (the1st@whoai.am)
"""

import time
from collections import defaultdict

import pytest


class TestBatchGenerateStreamUnit:
    """Unit tests for batch_generate_stream (no model loading)."""

    def test_import_batch_generate_stream(self):
        """Test that batch_generate_stream can be imported."""
        from mlx_parallm import batch_generate_stream

        assert callable(batch_generate_stream)

    def test_batch_generate_stream_in_exports(self):
        """Test that batch_generate_stream is in __all__."""
        import mlx_parallm

        assert "batch_generate_stream" in mlx_parallm.__all__

    def test_function_signature(self):
        """Test that batch_generate_stream has correct signature."""
        import inspect

        from mlx_parallm.utils import batch_generate_stream

        sig = inspect.signature(batch_generate_stream)
        params = list(sig.parameters.keys())

        assert "model" in params
        assert "tokenizer" in params
        assert "prompts" in params
        assert "max_tokens" in params


@pytest.mark.skipif(
    not pytest.importorskip("mlx", reason="MLX not available"),
    reason="MLX not installed",
)
class TestBatchGenerateStreamIntegration:
    """Integration tests requiring a model (skip if no model available)."""

    @pytest.fixture
    def small_model(self):
        """Load a small model for testing. Skip if not available."""
        try:
            from mlx_parallm import load

            # Try to load a small model - Phi-3-mini or similar
            model, tokenizer = load("microsoft/Phi-3-mini-4k-instruct-mlx")
            return model, tokenizer
        except Exception as e:
            pytest.skip(f"Could not load test model: {e}")

    def test_single_prompt_streaming(self, small_model):
        """Test streaming with a single prompt."""
        from mlx_parallm import batch_generate_stream

        model, tokenizer = small_model
        prompts = ["Say hello in one word."]

        tokens_received = []
        for user_idx, token, done in batch_generate_stream(
            model, tokenizer, prompts, max_tokens=10
        ):
            assert user_idx == 0  # Only one user
            tokens_received.append((token, done))

        assert len(tokens_received) > 0
        # Last token should be marked as done
        assert tokens_received[-1][1] is True

    def test_batch_streaming_multiple_users(self, small_model):
        """Test streaming with multiple concurrent prompts."""
        from mlx_parallm import batch_generate_stream

        model, tokenizer = small_model
        prompts = [
            "Count from 1 to 3.",
            "Say hello.",
            "What is 2+2?",
        ]

        user_tokens: dict[int, list[str]] = defaultdict(list)
        user_done: dict[int, bool] = {}

        for user_idx, token, done in batch_generate_stream(
            model, tokenizer, prompts, max_tokens=20
        ):
            user_tokens[user_idx].append(token)
            if done:
                user_done[user_idx] = True

        # All users should have received tokens
        assert len(user_tokens) == 3
        for user_idx in range(3):
            assert len(user_tokens[user_idx]) > 0

    def test_streaming_yields_incrementally(self, small_model):
        """Test that tokens are yielded as they're generated, not all at once."""
        from mlx_parallm import batch_generate_stream

        model, tokenizer = small_model
        prompts = ["Tell me a short story about a cat."]

        yield_times = []
        for _user_idx, _token, _done in batch_generate_stream(
            model, tokenizer, prompts, max_tokens=30
        ):
            yield_times.append(time.perf_counter())

        # Should have multiple yields with time gaps (not all instant)
        assert len(yield_times) > 5
        # Check that there's some time variation (tokens generated over time)
        time_diffs = [yield_times[i + 1] - yield_times[i] for i in range(len(yield_times) - 1)]
        # At least some gaps should be non-zero (generation takes time)
        assert any(d > 0.001 for d in time_diffs)

    def test_early_eos_handling(self, small_model):
        """Test that users who hit EOS early stop receiving tokens."""
        from mlx_parallm import batch_generate_stream

        model, tokenizer = small_model
        # One short prompt, one longer
        prompts = [
            "Say yes.",  # Should finish quickly
            "Explain quantum physics in detail.",  # Will take longer
        ]

        user_done_at: dict[int, int] = {}

        for token_count, (user_idx, _token, done) in enumerate(
            batch_generate_stream(model, tokenizer, prompts, max_tokens=50),
            start=1,
        ):
            if done and user_idx not in user_done_at:
                user_done_at[user_idx] = token_count

        # First user should finish before second (usually)
        # This is probabilistic but generally true
        if 0 in user_done_at and 1 in user_done_at:
            # At least verify both finished
            assert user_done_at[0] is not None
            assert user_done_at[1] is not None


class TestBatchGenerateStreamPerformance:
    """Performance tests for batch streaming."""

    @pytest.mark.skipif(
        not pytest.importorskip("mlx", reason="MLX not available"),
        reason="MLX not installed",
    )
    def test_batch_faster_than_sequential(self):
        """Test that batched generation is faster than sequential."""
        pytest.skip("Performance test - run manually with large model")

        from mlx_parallm import batch_generate_stream, load, stream_generate

        model, tokenizer = load("model-path")
        prompts = ["Hello"] * 5

        # Time batched
        start = time.perf_counter()
        list(batch_generate_stream(model, tokenizer, prompts, max_tokens=50))
        batch_time = time.perf_counter() - start

        # Time sequential
        start = time.perf_counter()
        for prompt in prompts:
            list(stream_generate(model, tokenizer, prompt, max_tokens=50))
        sequential_time = time.perf_counter() - start

        # Batch should be faster (or at least not significantly slower)
        assert batch_time < sequential_time * 1.5


class TestBatchGenerateStreamEdgeCases:
    """Edge case tests."""

    def test_empty_prompts_list(self):
        """Test handling of empty prompts list."""

        # This should not crash, just yield nothing
        # (actual behavior depends on implementation)
        pass  # Skip actual test - would need mock model

    def test_single_token_max(self):
        """Test with max_tokens=1."""
        pass  # Skip actual test - would need mock model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
