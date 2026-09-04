# Changelog

## 0.2.0 — Unreleased

- Added structured, confidence-ranked generation shared by all chat providers.
- Added Anthropic and OpenAI-compatible backends for hosted and local models.
- Added first-class MiniMax OpenAI-compatible support with reasoning-aware completion budgets.
- Added automatic target-context inference and cross-apex output filtering.
- Added a deterministic structural+model hybrid candidate pool.
- Added deterministic JSON/YAML OpenAPI and subdomain benchmarks with repeated splits, zero-inference replay, mutation and wordlist controls, route-shape-safe indexed scoring, paired confidence intervals, source-attributed/model-only hits, confidence/ranking metrics, and cost reporting.
- Added the bounded generate → validate → refeed loop.
- Added per-query DNS deadlines and wildcard suppression, plus scoped HTTP redirects,
  normalized base URLs, byte- and wall-clock-bounded streamed validation, and
  dynamic soft-404 detection.
- Added `run`, `doctor`, `bench`, and safety-gated `royale` commands.
- Strengthened Recon Royale submission gating around generated-set precision and marginal findings.
- Preserved the existing `generate` and `analyze` workflows.
- Replaced the stale failing test suite with passing current-behavior coverage.
- Updated package metadata, examples, documentation, and CI for supported Python 3.10–3.14, including mypy and package-build gates; Python 3.9 was retired after upstream end-of-life and dependency security fixes moved to Python 3.10+.
- Corrected minimum OpenAI (`1.55.3`) and Anthropic (`0.40.0`) SDK versions to the
  first releases supporting Enumeraite's request fields while constructing safely
  with modern `httpx`; Python 3.10 CI now tests every declared runtime lower bound
  together and checks the resolved environment for dependency conflicts.
- Hardened benchmark fairness by enforcing the requested candidate budget, rejecting
  incomplete or misaligned replay traces, preserving failed-run costs on replay, and
  refusing non-finite numeric output.
- Removed stale hard-coded CLI price guesses; debug output now reports sanitized token
  telemetry while dated, model-specific cost estimates remain in benchmark artifacts.
- Added bounded per-prefix HTTP miss calibration, configurable calibration depth,
  strict TLS verification by default, and calibrated recognition of non-GET route
  evidence such as 405 and 422 responses.
- Corrected the optional Hugging Face runtime to Qwen3-capable, advisory-clean
  Transformers and PyTorch floors, declared the required Accelerate dependency,
  and retained the compatible `torch_dtype` loader argument.
- Added Bandit and dependency-audit CI gates, and corrected stale installation hints
  that referenced provider extras which do not exist.
- Raised the Pydantic and dnspython floors to their first releases fixing the
  advisories found by auditing the complete minimum-dependency environment.
- Made provider auto-detection honor configured local base URLs, require the complete
  Hugging Face runtime, and share strict subdomain validation with generation.
- Versioned the corrected raw-emission candidate cap and content-hash uncertainty
  clustering and concrete-candidate enforcement as benchmark metric `route-shape-v10`;
  experiment metadata can no longer
  overwrite canonical manifest fields, and replay rejects coerced boolean or fractional
  counts.
- Made direct provider inputs reject fractional/boolean generation budgets and
  ambiguous boolean configuration strings instead of silently coercing them.
- Added a 75% package coverage floor to CI; the expanded regression suite now
  exercises provider, validation, benchmark, replay, and compatibility boundaries.
- Split benchmark controls into structural-only, lexical-mutation-only, and their
  structural-first union so deterministic gains can be attributed instead of hidden
  inside one baseline score.
- Added the PEP 561 `py.typed` marker so downstream type checkers can consume the
  package's public annotations.
- Made paired candidate/score and benchmark iteration fail on length mismatches,
  and added Ruff gates that prevent future silent `zip` truncation and useless
  expression statements.
- Bounded tolerant-JSON salvage to the documented candidate shape and made
  excessive JSON nesting a cheap rejection instead of a recursion failure or
  overlapping-span workload.

## 0.1.4

- Last public proof-of-concept release before the unified discovery pipeline.
