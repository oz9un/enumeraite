# Enumeraite configuration

`enumeraite doctor` shows the providers discovered from the environment and the provider that `enumeraite run` will select.

## Environment variables

Hosted backends:

| Provider | API key | Optional model | Optional base URL |
|---|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | `ANTHROPIC_MODEL` | — |
| OpenAI | `OPENAI_API_KEY` | `OPENAI_MODEL` | — |
| OpenRouter | `OPENROUTER_API_KEY` | `OPENROUTER_MODEL` | `OPENROUTER_BASE_URL` |
| MiniMax | `MINIMAX_API_KEY` | `MINIMAX_MODEL` | `MINIMAX_BASE_URL` |
| Groq | `GROQ_API_KEY` | `GROQ_MODEL` | `GROQ_BASE_URL` |
| DeepSeek | `DEEPSEEK_API_KEY` | `DEEPSEEK_MODEL` | `DEEPSEEK_BASE_URL` |
| Together | `TOGETHER_API_KEY` | `TOGETHER_MODEL` | `TOGETHER_BASE_URL` |

Local backends:

| Provider | Model | Base URL |
|---|---|---|
| Ollama | `OLLAMA_MODEL` | `OLLAMA_BASE_URL` |
| vLLM | `VLLM_MODEL` | `VLLM_BASE_URL` |
| llama.cpp | `LLAMACPP_MODEL` | `LLAMACPP_BASE_URL` |
| Generic local alias | `LOCAL_MODEL` | `LOCAL_BASE_URL` |

Hugging Face remote model code is disabled by default. Set `HUGGINGFACE_TRUST_REMOTE_CODE=true` only when a selected model repository genuinely requires it and you have reviewed that repository's code. The two bundled Enumeraite Qwen3 model names are pinned to their current immutable repository commits; custom models require a full commit SHA in `HUGGINGFACE_REVISION` (or `revision` in JSON) to avoid silently downloading changed weights or code. The advisory-clean optional-extra floors are Transformers 5.10.1 and PyTorch 2.13.0; it also includes Accelerate for automatic device placement.

Global generation controls:

```bash
export ENUMERAITE_CONFIDENCE=model
export ENUMERAITE_CONTEXT=true
export ENUMERAITE_HYBRID=true
export ENUMERAITE_TEMPERATURE=0.7
```

Provider-call and hybrid-budget controls:

```bash
export ENUMERAITE_BATCH_SIZE=20
export ENUMERAITE_MAX_BATCH_CALLS=3
export ENUMERAITE_MAX_TOKENS=8192
export ENUMERAITE_PROVIDER_TIMEOUT=180
export ENUMERAITE_HYBRID_MODEL_FRACTION=0.5
export ENUMERAITE_SEED_LIMIT=500
export ENUMERAITE_SEED_CHAR_LIMIT=50000
```

`ENUMERAITE_MAX_BATCH_CALLS` is a hard per-generation call cap. The seed limits deterministically sample large input files across their full ordering and bound model prompt size; all original seeds still remain known to the engine's duplicate filter. The model fraction must be between 0 and 1 and controls the minimum model proposal budget when hybrid generation is active and the structural pool has unfilled output slots. If high-evidence structural proposals already fill a small candidate budget, the model call is skipped instead of paying for candidates that would normally be discarded. MiniMax defaults to `MiniMax-M3`, 8,192 completion tokens, batches of 20, and at most three batch calls because reasoning can consume a substantial part of the completion budget. These are safeguards, not universal optimums; benchmark your own target distribution.

HTTP validation verifies TLS and uses depth-1, per-prefix soft-404
calibration by default. The group cap bounds calibration traffic:

```bash
export ENUMERAITE_HTTP_VERIFY_SSL=true
export ENUMERAITE_HTTP_SOFT_404_THRESHOLD=0.92
export ENUMERAITE_HTTP_BASELINE_SAMPLES=2
export ENUMERAITE_HTTP_BASELINE_DEPTH=1
export ENUMERAITE_HTTP_MAX_BASELINE_GROUPS=32
```

Set the depth to `0` for one target-wide baseline. Disable TLS verification only
for an explicitly authorized target whose certificate you expect not to validate.

Confidence modes:

- `model`: use the model's structured confidence value.
- `logprob`: derive a score from output-token log probabilities when supported.
- `heuristic`: retain the old seed-similarity score as a benchmark control.
- `uniform`: preserve generation order with flat scores.

`model` is the default. Confidence is a ranking signal, not a probability that a target exists.

`request_timeout` bounds provider calls and DNS queries. HTTP response streams also
use it as a monotonic body deadline in addition to Requests' socket-inactivity
timeout, so a peer cannot keep validation open indefinitely by drip-feeding a
byte-capped response.

## JSON file

Enumeraite checks `enumeraite.json`, `~/.enumeraite.json`, and `~/.config/enumeraite/config.json`. Explicit file values win; missing provider fields may be filled from environment variables.

```json
{
  "default_provider": "ollama",
  "default_count": 100,
  "request_timeout": 10,
  "http_verify_ssl": true,
  "http_baseline_depth": 1,
  "providers": {
    "ollama": {
      "preset": "ollama",
      "base_url": "http://localhost:11434/v1",
      "api_key": "sk-local",
      "model": "qwen3:8b",
      "confidence": "model",
      "context": true,
      "temperature": 0.7,
      "max_tokens": 4000
    },
    "openrouter": {
      "preset": "openrouter",
      "api_key": "use-an-environment-variable-instead",
      "model": "openai/gpt-4o-mini"
    },
    "minimax": {
      "preset": "minimax",
      "api_key": "use-MINIMAX_API_KEY-instead",
      "model": "MiniMax-M3",
      "hybrid": true,
      "batch_size": 20,
      "max_batch_calls": 3
    }
  }
}
```

Prefer environment variables for secrets. Configuration files containing keys are ignored by the repository's default `.gitignore`, but file permissions and backups remain your responsibility.

## Provider selection

Without `--provider`, `run` first honors a usable `default_provider` from the JSON configuration. If that provider is absent or unavailable, it chooses the first usable provider in this order:

1. Anthropic
2. OpenAI
3. OpenRouter
4. DeepSeek
5. MiniMax
6. Groq
7. Together
8. reachable Ollama, vLLM, or llama.cpp
9. legacy Claude
10. Hugging Face, only when its heavy dependencies are installed

Override the selection per run:

```bash
enumeraite run paths.txt --provider ollama --model qwen3:8b
```

Benchmark-only overrides are also available as `--temperature`, `--batch-size`, `--max-batch-calls`, `--max-tokens`, `--provider-timeout`, and `--hybrid-model-fraction`. They take precedence over the selected provider configuration for that invocation. Use `--temperature 0` when the backend supports deterministic greedy decoding and you need the closest possible replay; provider-side nondeterminism may still remain.
