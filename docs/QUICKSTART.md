# Enumeraite quick start

## Install

```bash
pip install enumeraite
```

Configure a hosted provider or start a local OpenAI-compatible server:

```bash
export ANTHROPIC_API_KEY='...'
# or OPENAI_API_KEY / OPENROUTER_API_KEY

# local alternative
ollama pull qwen3:8b
ollama serve
```

Check the setup:

```bash
enumeraite doctor
```

## Run

Create `paths.txt`:

```text
/api/v1/users
/api/v1/orders
/api/v1/auth/login
```

Generate candidates:

```bash
enumeraite run paths.txt --target api.example.com
```

For subdomains, the apex is inferred:

```text
api-dev.example.com
api-prod.example.com
cdn-prod.example.com
```

```bash
enumeraite run subdomains.txt
enumeraite run subdomains.txt --validate | httpx
```

Status output goes to stderr. Piped stdout contains only candidate values.

## Feedback loop

```bash
enumeraite run paths.txt \
  --target https://api.example.com \
  --loop --rounds 3 --count 150
```

Each round validates candidates, adds real findings to the evidence, summarizes hit/miss patterns, and generates again. The loop stops when it hits the round/candidate budget or two rounds produce no new findings.

Path validation compares responses with bounded random nonexistent paths to identify
soft 404s and global handlers. HTTP 401 and 403 responses are retained only when
they differ from the calibrated miss response. Subdomain validation probes random
labels and suppresses wildcard-DNS responses.

Only validate targets you are authorized to test.

## Machine-readable output

```bash
enumeraite run paths.txt --target api.example.com --json
enumeraite run paths.txt --target api.example.com --json -o candidates.json
```

## Benchmark before making claims

```bash
enumeraite bench openapi \
  --specs benchmarks/fixtures/openapi \
  --count 100 \
  -o bench-results/smoke
```

This runs structural-only, lexical-mutation-only, and combined deterministic controls. Add `--providers ollama` or another configured provider for an actual model comparison. The included fixture proves the pipeline works; use many independent real specifications before drawing conclusions about model quality.
