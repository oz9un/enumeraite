<p align="center">
  <img src="assets/enumeraite_logo.png" alt="Enumeraite Logo" width="400"/>
</p>

<h1 align="center">enumer<span style="color: #ef4444;">ai</span>te</h1>

<p align="center">
  <strong>Target-specific, AI-assisted web attack-surface enumeration.</strong>
</p>

<p align="center">
  <a href="https://github.com/oz9un/enumeraite/actions"><img src="https://img.shields.io/github/actions/workflow/status/oz9un/enumeraite/ci.yml?style=flat-square&label=tests" alt="Tests"></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License"></a>
</p>

<p align="center">
  Presented at <strong>DEF CON 33 Recon Village</strong> by
  <a href="https://github.com/oz9un">Özgün Kültekin</a> ·
  <a href="https://www.youtube.com/watch?v=IzsBS_E2RVY">watch the talk</a>
</p>

Enumeraite learns naming, versioning, resource, environment, and infrastructure conventions from a target's existing paths or subdomains. It asks a general-purpose hosted or local model for structured, confidence-ranked candidates, then optionally validates and feeds the results into another bounded generation round.

The model is not the product. The pipeline is: target context, structured output, validation, feedback, and measurement against held-out ground truth.

## Quick start

```bash
pip install enumeraite

# Hosted provider
export ANTHROPIC_API_KEY='...'
# or: export OPENAI_API_KEY='...'
# or: export MINIMAX_API_KEY='...'

# Local provider: start Ollama and install a model
ollama pull qwen3:8b

# Auto-detect paths vs. subdomains and the best configured provider
enumeraite doctor
enumeraite run seeds.txt

# stdout stays clean when piped
enumeraite run subdomains.txt | httpx
```

The seed file is one value per line. Empty lines and lines beginning with `#` are ignored.

## What changed in 0.2

- One structured prompt/parser shared by every chat backend.
- Model-native confidence instead of similarity scores that favored obvious duplicates.
- Automatic target-context inference from the supplied seed.
- A hybrid candidate pool that combines high-evidence structural gaps with model proposals.
- OpenAI-compatible support for OpenAI, MiniMax, OpenRouter, Ollama, vLLM, llama.cpp, Groq, DeepSeek, and Together.
- Native Anthropic structured output.
- A replayable ground-truth benchmark harness with wordlist and mutation controls.
- An opt-in generate → validate → refeed loop.
- DNS wildcard suppression and bounded per-prefix HTTP soft-404 calibration;
  non-404 statuses count only when they differ from the calibrated miss response.
- Pipe-clean `run`, configuration diagnostics with `doctor`, and structured `--json` output.
- Safety-gated Recon Royale preparation and submission.

The old `generate` and `analyze` command groups remain available.

## Primary workflow

```bash
# Raw candidates
enumeraite run paths.txt --target api.example.com -c 100

# JSON with confidence and metadata
enumeraite run paths.txt --target api.example.com --json

# Only validated subdomains; wildcard DNS is filtered
enumeraite run subdomains.txt --validate

# Generate, validate, refeed evidence, and stop on diminishing returns
enumeraite run paths.txt --target https://api.example.com --loop --rounds 3 -c 150

# Select a backend explicitly
enumeraite run paths.txt --provider ollama --model qwen3:8b
enumeraite run paths.txt --provider openrouter
enumeraite run paths.txt --provider minimax
```

`--loop` implies validation. Path validation requires `--target`. Use active validation only on systems you are authorized to test.

HTTP validation verifies TLS certificates by default and calibrates miss responses
per virtual-router prefix. See [Configuration](docs/CONFIGURATION.md) before testing
an authorized target with a deliberately untrusted certificate.

Hybrid generation is enabled by default for configured chat providers. It proposes missing ancestors, version/resource grid cells, environment variants, and adjacent numbered hosts only when those relationships are evidenced by the seed; `--no-hybrid` isolates the model for an experiment.

## Providers

| Provider | Name | Configuration |
|---|---|---|
| Anthropic | `anthropic` or legacy `claude` | `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL` |
| OpenAI | `openai` | `OPENAI_API_KEY`, optional `OPENAI_MODEL` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY`, optional `OPENROUTER_MODEL` |
| MiniMax | `minimax` | `MINIMAX_API_KEY`, optional `MINIMAX_MODEL` |
| Groq | `groq` | `GROQ_API_KEY`, optional `GROQ_MODEL` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY`, optional `DEEPSEEK_MODEL` |
| Together | `together` | `TOGETHER_API_KEY`, optional `TOGETHER_MODEL` |
| Ollama | `ollama` or `local` | server on `localhost:11434`, optional `OLLAMA_MODEL` |
| vLLM | `vllm` | server on `localhost:8000`, optional `VLLM_MODEL` |
| llama.cpp | `llamacpp` | server on `localhost:8080`, optional `LLAMACPP_MODEL` |
| Hugging Face | `huggingface` | `pip install 'enumeraite[huggingface]'` |

Run `enumeraite doctor` to see what is configured and what `run` will select. Hugging Face is not reported as ready unless `torch`, `transformers`, and `accelerate` are installed at the optional extra's declared minimum versions.

See [Configuration](docs/CONFIGURATION.md) for JSON configuration, base-URL overrides, and confidence experiments.

## Benchmarking

Enumeraite scores LLMs and simple controls through the exact same harness. Hidden items are never included in the seed, and splits are deterministic across machines.

```bash
# Included smoke fixture; no model key required
enumeraite bench openapi \
  --specs benchmarks/fixtures/openapi \
  --count 100 -o bench-results/smoke

# A real model versus mutation, with a full external wordlist union
enumeraite bench openapi \
  --specs /data/openapi-directory/APIs \
  --providers anthropic,minimax,ollama \
  --wordlist /data/seclists-assetnote-union.txt \
  --repeats 5 --seed-limit 50 \
  --count 200 --budgets 25,50,100,200 \
  -o bench-results/experiment-01
```

The reusable harness and tiny smoke fixture belong in Git. Large third-party datasets and generated reports do not; `bench-data/` and `bench-results/` are ignored.

Key metrics include recall at several budgets, precision, average precision, reciprocal rank, fill rate, confidence Brier score, and cost per valid finding. Structural-only, lexical-mutation-only, and combined controls make their separate contributions visible; the structural-first union is the default minimum bar.

Rescore a saved run after metric changes without paying for inference again:

```bash
enumeraite bench replay bench-results/experiment-01/results.json \
  -o bench-results/experiment-01-replayed
```

See [Benchmarking](docs/BENCHMARKING.md) for dataset construction and interpretation.

## Recon Royale

Recon Royale support is dry-run by default:

```bash
export RECON_ROYALE_API_KEY='...'

enumeraite royale \
  --seeds current-target-subs.txt \
  --baseline passive-results.txt \
  --provider ollama

# Explicitly submit only after reviewing the dry-run report
enumeraite royale ... --submit
```

The command fetches the current target, removes out-of-scope names, resolves locally with wildcard filtering, reports Enumeraite's marginal contribution over the passive baseline, and refuses submission below the configured valid-rate threshold. No nightly job is enabled by this repository; enable automation only after a representative benchmark clears your quality threshold.

See [Recon Royale](docs/RECON_ROYALE.md).

## Legacy workflows

```bash
enumeraite generate path -i known-paths.txt -c 100
enumeraite generate subdomain -i known-subs.txt --validate
enumeraite analyze path /api/v1/users -f 'admin operations'
enumeraite analyze subdomain api-prod-us1.example.com
```

## Development

```bash
git clone https://github.com/oz9un/enumeraite.git
cd enumeraite
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

The benchmark pricing table is deliberately small and editable. It is experiment metadata, not a billing source of truth.

## Ethical use

Enumeraite is intended for authorized security testing and research. Generated candidates are hypotheses, not proof that an endpoint or subdomain exists. Obtain permission, respect scope and rate limits, and combine results with conventional reconnaissance and validation.

## License

MIT. See [LICENSE](LICENSE).
