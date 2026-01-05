
# MLX ParaLLM

Batched KV caching for fast parallel inference on Apple Silicon devices, via [MLX](https://github.com/ml-explore/mlx).

This repo heavily borrows from [`mlx_lm`](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm). Will explore how to add batched generation there as a non-breaking PR.


## Installation

```bash
pip install mlx mlx_lm
git clone https://github.com/willccbb/mlx_parallm
cd mlx_parallm
pip install -e .
```

## Usage

### Batch Generation (returns complete responses)

```python
from mlx_parallm import load, batch_generate

model, tokenizer = load("mlx-community/Phi-3-mini-4k-instruct-4bit")
prompts = ["Hello!", "Count to 3.", "What is 2+2?"]

responses = batch_generate(
    model, tokenizer,
    prompts=prompts,
    max_tokens=100,
    verbose=True,
    temp=0.7
)
# Returns: ["Hello! How can I help?", "1, 2, 3.", "4"]
```

### Streaming Batch Generation (NEW!)

Stream tokens as they're generated - essential for real-time applications:

```python
from mlx_parallm import load, batch_generate_stream

model, tokenizer = load("mlx-community/Phi-3-mini-4k-instruct-4bit")
prompts = ["Hello!", "Count to 3."]

for user_idx, token, is_done in batch_generate_stream(
    model, tokenizer,
    prompts=prompts,
    max_tokens=50,
    temp=0.7
):
    if is_done:
        print(f"\n[User {user_idx} finished]")
    else:
        print(f"[{user_idx}] {token}", end="", flush=True)
```

Output (tokens interleaved as generated):
```
[0] Hello[1] 1[0] ![1] ,[0]  How[1]  2[0]  can[1] ,[0]  I[1]  3[0]  help[1] .
[User 1 finished]
[0]  you[0] ?
[User 0 finished]
```

### Streaming Yields

`batch_generate_stream()` yields tuples of `(user_idx, token_text, is_done)`:

| Field | Type | Description |
|-------|------|-------------|
| `user_idx` | `int` | Index of the user (0 to batch_size-1) |
| `token_text` | `str` | Decoded token string |
| `is_done` | `bool` | `True` if user hit EOS or max_tokens |


## Integration with Your Own Server

`mlx_parallm` is designed as a **library**, not a standalone server. Use it to add parallel inference to your existing MLX-based application.

### Basic Integration Pattern

```python
from mlx_parallm import batch_generate_stream
from mlx_parallm.models.base import BatchedKVCache

# Your server collects requests into batches
async def handle_batch(requests: list[Request]):
    prompts = [r.prompt for r in requests]

    # Stream tokens to all users simultaneously
    for user_idx, token, done in batch_generate_stream(
        model, tokenizer, prompts, max_tokens=100
    ):
        # Route token to correct user's response stream
        await requests[user_idx].stream.send(token)
        if done:
            await requests[user_idx].stream.close()
```

### SSE/Server-Sent Events Integration

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    async def generate_sse():
        for user_idx, token, done in batch_generate_stream(
            model, tokenizer, [request.prompt], max_tokens=100
        ):
            if done:
                yield f"data: [DONE]\n\n"
            else:
                chunk = {"choices": [{"delta": {"content": token}}]}
                yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(generate_sse(), media_type="text/event-stream")
```

### Using BatchedKVCache Directly

For advanced use cases, use `BatchedKVCache` directly in your model:

```python
from mlx_parallm.models.base import BatchedKVCache

# Create cache for batch of users
batch_size = 10
cache = [
    BatchedKVCache(head_dim=128, n_kv_heads=8, batch_size=batch_size)
    for _ in range(num_layers)
]

# Use in your model's forward pass
# See generate_step() in utils.py for reference implementation
```

### Adding New Model Architectures

To add support for a new model architecture:

1. Copy the model file from `mlx_lm/models/your_model.py`
2. Replace `KVCache` imports with `BatchedKVCache`:
   ```python
   # Before
   from mlx_lm.models.cache import KVCache

   # After
   from mlx_parallm.models.base import BatchedKVCache as KVCache
   ```
3. Place in `mlx_parallm/models/your_model.py`
4. The model will be auto-discovered by `model_type` in config.json


## Models

Tested models:
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `microsoft/Phi-3-mini-4k-instruct`
- `google/gemma-1.1-2b-it`
- `mlx-community/Meta-Llama-3-8B-Instruct-4bit`
- `mlx-community/Phi-3-mini-4k-instruct-4bit`
- `mlx-community/gemma-1.1-2b-it-4bit`

Both quantized and `float16` models are supported. `float16` models generally perform faster if sufficient RAM is available (up to 1300+ tok/s throughput for `gemma-2b` on M3 Max 128GB).


## Features

Supported:
- `batch_generate` - returns complete responses
- `batch_generate_stream` - **NEW!** yields tokens as generated
- Auto-padding for variable-length prompts
- Auto-formatting with chat templates (`format_prompts=True`)
- Temperature sampling (`temp=0` for greedy, `temp>0` for sampling)
- Top-p (nucleus) sampling
- Single-stream `generate` and `stream_generate` methods

Not yet supported:
- Repetition penalties
- Dynamic batching for async requests (implement in your server layer)


## Performance

Batched inference provides significant throughput improvements:

| Scenario | Throughput |
|----------|------------|
| 1 user (baseline) | ~60 tok/s |
| 10 users batched | ~400 tok/s |
| 50 users batched | ~800 tok/s |

*Benchmarks on M3 Max 128GB with Phi-3-mini-4k-instruct-4bit*


## API Reference

### `batch_generate_stream(model, tokenizer, prompts, max_tokens=100, format_prompts=True, **kwargs)`

Generator that yields tokens for batch inference with streaming.

**Parameters:**
- `model`: MLX model instance
- `tokenizer`: Tokenizer (PreTrainedTokenizer or TokenizerWrapper)
- `prompts`: List of prompt strings
- `max_tokens`: Maximum tokens per response (default: 100)
- `format_prompts`: Apply chat template (default: True)
- `**kwargs`: Additional args passed to `generate_step` (temp, top_p, etc.)

**Yields:** `Tuple[int, str, bool]` - (user_index, token_text, is_finished)


### `batch_generate(model, tokenizer, prompts, max_tokens=100, verbose=False, format_prompts=True, **kwargs)`

Generate complete responses for a batch of prompts.

**Returns:** `List[str]` - Complete responses for each prompt


## License

MIT


## Contributing

PRs welcome! Especially for:
- Additional model architectures
- Performance optimizations
- Documentation improvements

---

*Streaming support contributed by [LibraxisAI Team](https://github.com/LibraxisAI)*
