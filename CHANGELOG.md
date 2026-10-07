# Changelog

### Local live dashboard (unreleased)

- Follow observed in-scope CNAME targets through independent DNS checks and
  propose literal DNS parents of verified names. Dashboard and HTML evidence
  distinguish observed alias targets from confirmed findings.
- Preserve usable crt.sh observations alongside malformed rows and combine both
  certificate name fields. Sample passive intake across naming families within
  remaining capacity; a full wildcard cache no longer prevents fresh calibration.
- Resume newly eligible pattern expansion from older checkpoints and retry
  admission of saved aliases after candidate-cap increases, without resetting
  completed DNS attempts or recorded model spend.
- Added `tools/recon_bench.py` for offline production-session replay against
  frozen, incomplete DNS inventories, with query-order scoring, source-hash
  gates for final tests, and explicit unknown observations.
- Preserve detailed DNS and proposal provenance in checkpoints, reports, and the
  dashboard. Add optional authenticated MerkleMap intake with isolated credentials
  and bounded collection, and remove direct and subfinder HackerTarget collection.
- Preserve open-ended DNS scans through matching CLI/dashboard controls,
  streamed checkpoints, full exclusion history, and optional prediction
  deadlines. Finite defaults and explicit spending/rate/scope safeguards remain.
- Added off-by-default witnessed-layout vocabulary expansion in the dashboard
  and `--vocabulary-breadth`. It replaces eligible third AI passes, shares the
  existing budget, and keeps Python in charge of structure, scope and bounded
  expansion. Guesses require DNS verification; malformed responses do not
  trigger another paid request. Old checkpoints and running dashboard APIs
  remain compatible.
- Added per-method queued/checked/result counts to reports and the Patterns
  tab; unresolved DNS and wildcard matches are not conflated with misses.
- Improved numeric planning with shared indexes and lazy admitted-plan
  materialization. Tightened wildcard calibration and numeric-resume handling,
  and added the scoped MiniMax M3 fast subdomain profile.
- Broader discovery uses up to three distinct, checkpointed hosted-model passes
  per round, each requesting at most 20 names from rotating verified cohorts.
  Numeric families no longer crowd other naming styles out of the prompt.
  Python pattern pages skip historical guesses before truncation. New scans
  keep exploring through misses; `--until-stale` explicitly restores the two
  no-hit-round stop. Existing budgets, scope, DNS rate, and hard limits remain.
- Added optional TypeSafe Jev plan ranking (`--predictor jev`), a separate
  session-only UI credential, explicitly additional cumulative prediction budget,
  recent plan scores, and deterministic fallback. Fixed HTTPS endpoint, pinned
  model, bounded typed requests, durable reservations, no paid retries, and
  simulated-response tests; no live Jev quality or cost benchmark claimed.
- Numeric families now preserve fixed tokens, enumerate observed changing slots
  in bounded waves with history-based range progress, and sample generic child
  hypotheses across representative hosts instead of repeating them everywhere.
  Supported family ranges are generated in Python, not listed by the text model.
- Accept root domains or known subdomains as scan input, with an offline,
  private-suffix-aware scope preview, optional starting hostnames, `--seed`, and
  explicit `--scope`. Seed-first scans verify supplied clues before expansion;
  scans without seeds collect public names automatically.
- Added optional, bounded, keyless subfinder collection when installed, alongside
  crt.sh, with isolated configuration and no shell invocation.
- Added `enumeraite ui`: start, pause, resume, and inspect saved scans from a
  local browser dashboard, with worker activity, discovery trees, findings,
  inferred patterns, resource counters, and downloadable reports.
- Added session-only MiniMax, OpenAI, and Anthropic credential entry and optional budgeted AI from the
  dashboard. Credentials never belong in persisted scan state or exports.
- Removed the dashboard access-token requirement for the local workflow; plain
  local URLs work without login or browser storage. Provider API keys remain
  session-only. Loopback binding and cross-origin/host checks remain enabled.
- Added a loopback-only HTTP boundary with origin/host checks,
  bounded requests, no request logging, and self-contained packaged assets.

## 0.2.0 — Unreleased

- Added `scan TARGET` with keyless passive discovery from certificate transparency,
  common-label bootstrap, wildcard-aware DNS verification, and
  recursive expansion from verified findings.
- Added observed-slot pattern transfers, bounded cross-combinations, numeric
  slots anywhere in a label with preserved zero padding, tentative region/product/
  environment hypotheses, and bounded deeper DNS labels.
- Added shared concurrent DNS workers and global query-rate, candidate, round,
  depth, time, and verified-finding limits. `--find` includes known verified names;
  candidate limits include passive names, and source failures/caps are visible.
- Made bare scans free (`--provider none`). An explicit dollar budget can
  auto-select configured AI; conservative pre-call token reservations and a
  resumable ledger bound estimated spending, with explicit price overrides for
  unknown models/endpoints. Estimates are not provider invoices, and already
  in-flight SDK requests remain subject to SDK I/O timeout behavior.
- Added private atomic scan checkpoints and `--resume`, preserving cumulative
  attempts and estimated spending while starting a fresh time allowance.
- Added findings-only stdout, structured scan reports, a terminal discovery tree
  on stderr, and self-contained HTML trees with escaped remote metadata.
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
