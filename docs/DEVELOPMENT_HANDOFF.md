# Enumeraite 0.2.0 development handoff

Last updated: 2026-09-04

## Snapshot

This branch contains the unreleased Enumeraite 0.2.0 pipeline rebuild, based on
`origin/main` at `9d8f700`. It is a development branch, not a release. No tag,
pull request, PyPI upload, scheduled automation, or Recon Royale submission has
been created.

Large research datasets and generated benchmark results intentionally remain
local under the ignored `bench-data/`, `bench-results/`, and
`benchmark-results/` directories. No API credentials belong in this repository;
`.env.example` contains names and empty placeholders only.

## What is implemented

- A shared structured prompt, parser, and provider pipeline for Anthropic and
  OpenAI-compatible services.
- First-class presets for MiniMax, OpenAI, OpenRouter, Ollama, vLLM, llama.cpp,
  Groq, DeepSeek, and Together, plus the existing Hugging Face path.
- Bounded provider batches, retries, seed context, token/cache/reasoning
  telemetry, strict candidate validation, and tolerant structured-output
  recovery.
- A structural-first hybrid candidate pool for route ancestors,
  version/resource grids, subdomain environment grids, and adjacent numbered
  hosts. Paid inference is skipped when structural candidates already fill a
  small candidate budget.
- An opt-in generate -> validate -> refeed loop. Newly validated findings become
  evidence in the next round; previous guesses are excluded; the loop stops on
  candidate, round, inconclusive-validation, or diminishing-return limits.
- DNS validation with per-query deadlines and wildcard suppression.
- HTTP validation with scoped no-redirect requests, TLS verification,
  per-prefix soft-404 calibration, bounded response bodies, and wall-clock body
  deadlines.
- Replayable OpenAPI JSON/YAML/Swagger 2 and subdomain benchmarks, deterministic
  splits, structural and lexical-mutation controls, candidate-source attribution,
  cost telemetry, and zero-inference replay.
- The `route-shape-v10` scorer, including template-aware matching, distinct route
  coverage, indexed matching, clustered confidence intervals, and cost/source
  metrics.
- A simpler CLI centered on `run`, `doctor`, `bench`, and safety-gated `royale`.
  Output remains clean when piped and can be emitted as structured JSON. The old
  `generate` and `analyze` commands remain available.
- Security and release hardening: bounded regular-file reads, bounded parser
  input, error/secret sanitization, scope filtering, immutable Hugging Face
  revisions, typed-package metadata, modern dependency floors, and expanded CI.

## Current user experience: precise limitations

The closest current subdomain workflow is:

```bash
enumeraite run known-subs.txt \
  --target example.com \
  --loop --rounds 10 --count 1000
```

Important semantics:

- A seed file is mandatory. A bare apex cannot yet bootstrap a complete scan.
- `--count N` is a maximum candidate-attempt budget, not a promise to find N
  verified subdomains.
- `--validate` is a one-shot filter. `--loop` is what reuses verified findings in
  later rounds.
- Feedback and discoveries live only for that invocation; scans cannot yet be
  resumed.
- There is no runtime USD ceiling, whole-scan time limit, verified-result target,
  persistent discovery frontier, or `enumeraite scan example.com` facade.
- The feedback loop is adaptive, validation-guided discovery, not reinforcement
  learning. It does not update model weights or persist a learned policy.
- It does not yet pivot through certificate transparency, CNAMEs, IPs, ASNs,
  certificates, or HTTP content when a host is discovered.

## Verification completed

The following checks were recorded against this implementation before the
handoff commit:

- 323 tests passed independently on Python 3.10, 3.11, 3.12, 3.13, and 3.14.
- The full suite also passed on Python 3.10 with the declared minimum core
  dependencies.
- Fresh pre-push coverage was 79.89%, above the enforced 75% floor.
- Black, isort, Ruff, mypy, Bandit, Semgrep, and Vulture passed.
- Core dependencies and a real isolated Hugging Face dependency environment
  passed `pip check` and vulnerability audits.
- The source distribution and wheel built successfully and passed Twine,
  wheel-content, metadata, clean-install, CLI doctor, and typed-marker checks.
- `git diff --check` passed, and 32,500 one-off Unicode, URL, parser, candidate,
  matcher, and mutation property cases passed.
- Secret scanning of tracked/nonignored source and generated benchmark results
  found no committed credentials. A fresh staged-tree and branch-history scan
  is required immediately before every push.

## Benchmark conclusions

These are regression and design signals, not publishable superiority claims:

- Petstore model-only at 20 candidates: MiniMax M3 and M2.7 each recovered 6/10
  hidden shapes; mutation recovered 2/10. Recorded costs were about $0.0075 and
  $0.0017 respectively. M2.7 was roughly 4.3 times cheaper per recovered shape
  in this trace, while M3 ranked its hits better.
- crAPI hybrid at 20 candidates: each MiniMax model recovered 4/28 hidden shapes;
  mutation recovered 3/28.
- A final local Ollama `qwen3:8b` integration run recovered 9/26 hidden shapes at
  20 hybrid candidates, versus 4/20 for the deterministic union on that single
  demo split.
- In the three-split synthetic subdomain fixture, MiniMax hybrid recovered 32/60
  versus mutation's 31/52 and had worse precision/average precision. Paid model
  inference was not justified on that fixture.
- Larger deterministic ablations show that structural-first generation is the
  high-precision core; broad lexical mutation is primarily a low-precision tail.

Public/famous specifications are useful smoke tests but are vulnerable to model
memorization. The next credible evaluation needs 30-100 private or temporally
held-out OpenAPI specifications, five deterministic splits at budgets
10/25/50/100, and complete ground truth for at least 20 independent DNS apexes.

## Next milestone: effortless bounded scanning

Keep `run` as the expert seed-driven interface and add a user-facing facade:

```bash
enumeraite scan example.com
enumeraite scan example.com --find 100
enumeraite scan example.com --budget-usd 5
enumeraite scan example.com --time 30m --until-stale
enumeraite scan example.com --resume
```

Recommended implementation order:

1. Add `scan TARGET`, with optional `--seeds FILE` and stdin support. Infer DNS
   apex versus URL and provide a conservative built-in seed bootstrap when no
   seeds are supplied. Keep scope and authorization messaging explicit.
2. Introduce one composable stop controller covering verified-find target,
   maximum candidates, rounds, stale rounds, monotonic deadline, and USD spend.
   A request for N findings must count unique validated in-scope results, not
   guesses, and must require at least one hard safety bound.
3. Make USD limits pre-call constraints. Use a dated model-price registry with
   explicit overrides, reserve worst-case prompt/completion cost before each
   batch, reconcile against returned usage, shrink the final affordable batch,
   and fail closed for unknown prices. Strict budget mode should not retry an
   ambiguous paid request automatically.
4. Replace the ephemeral feedback loop with a resumable `DiscoverySession` and
   an atomic, versioned per-target state file outside the repository. Track each
   frontier entry's value, source/parent, confidence, validation state, round,
   timestamps, and expansion status, plus session spend and stop reason.
5. Add a persistent priority frontier: bootstrap seeds -> structural/model
   candidates -> validation -> enqueue verified findings as new evidence and
   pivots. Apply hard breadth, depth, rate, scope, candidate, time, and cost caps.
6. If adaptive allocation is added, use an auditable bandit over candidate-source
   families based on verified discoveries per cost. Describe it as adaptive
   validation-guided discovery; do not claim model-weight reinforcement learning.
7. Add live TTY status for verified count, attempts, spend/reserved budget,
   elapsed/remaining time, last discovery, and final stop reason. Preserve
   findings-only stdout when piped and expose the full ledger in JSON.
8. Test with fake clocks, providers, pricing, and DNS. Prove that cost reservation
   cannot cross the configured ceiling; test `--find` semantics, deadline
   propagation, scope/wildcard safety, resume equivalence, corrupt-state
   rejection, and duplicate-spend prevention. Then benchmark fixed candidate,
   USD, and time policies.

## Research and maintenance queue

- Design zero-seed bootstrap carefully: conservative bundled labels first;
  certificate-transparency and other passive sources should be explicit adapters.
- Calibrate candidate confidence on representative private data; current scores
  are ranking signals, not reliable probabilities.
- Compare MiniMax M3, M2.7, `qwen3:8b`, and one independent model in pure-model
  and hybrid modes before changing the default model fraction.
- Refactor the highest-complexity orchestration functions only with
  behavior-preserving tests.

## Continue on another machine

```bash
git fetch origin
git switch codex/enumeraite-v0.2-handoff
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

Start with the `scan` facade and stop-controller tests. Do not begin another
provider rewrite until those user journeys and hard-budget semantics exist.
