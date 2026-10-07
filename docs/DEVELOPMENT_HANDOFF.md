# Enumeraite 0.2.0 development handoff

Last updated: 2026-10-03

## Status

This is an unreleased development branch:
`codex/enumeraite-discovery-console-20261003`. It continues the 0.2 pipeline
rebuild with a resumable scan workflow and a local discovery dashboard.
It is not a release, tag, PyPI upload, or claim of exhaustive enumeration.

Only source, tests, public documentation and synthetic fixtures belong in Git.
Credentials, real-target inventories, scan checkpoints, model response traces,
private research reports and local session notes must stay outside the published
tree. The ignored benchmark directories are for local work, not public fixtures.

## Getting started

```bash
git clone --branch codex/enumeraite-discovery-console-20261003 https://github.com/oz9un/enumeraite.git
cd enumeraite
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
enumeraite ui
```

The dashboard opens a loopback-only URL. No dashboard login is required.
Choose free discovery or a supported hosted provider. Provider keys are entered
for the server session only; restarting clears them. Do not expose or tunnel
the dashboard: other local applications/users can access it.

CLI examples, using only a domain you are authorized to assess:

```bash
enumeraite scan example.com
enumeraite scan example.com --find 100 --tree
enumeraite scan ny-01-iphone.example.com --tree
enumeraite scan example.com --provider minimax --budget-usd 5 --vocabulary-breadth
enumeraite scan example.com --resume --time 10m --max-candidates 5000 --rounds 20
```

See [Quick start](QUICKSTART.md) for scope, provider configuration, limits,
pause/resume behavior and exports.

## Implemented discovery workflow

- Root-domain scans collect historical names from crt.sh, optional installed
  keyless subfinder, and MerkleMap when configured. Failed, capped and timed-out
  sources remain visible; public names are not findings until DNS verifies them.
- Hostname input starts with that seed. An offline Public Suffix List resolves
  registrable and private-hosting boundaries. Explicit `--scope`, `--seed`
  and `--seeds` controls reject out-of-scope or over-depth input before work.
  Seed-first scans skip passive collection and generic bootstrap by default.
- Verified names supply evidence for bounded recursive discovery. Guesses,
  wildcard results and inconclusive lookups are never treated as positive
  model evidence. This is a feedback loop, not model-weight reinforcement learning.
- Numeric families retain literal tokens and padding. Python generates changed
  numeric slots within the configured ceiling (99 by default). Observed
  IPv4-encoded families require supporting address evidence before applying
  their specialized bounds. Numeric alternatives do not need model generation.
- Shared immutable pattern indexes and lazy numeric-plan materialization avoid
  repeatedly rebuilding large candidate pools. Parent and plan paging preserve
  bounded exploration. Generic child labels sample representative hosts instead
  of being repeated beneath every numeric sibling.
- Hosted discovery uses up to three complementary, checkpointed passes per wave,
  at most 20 names each, with capacity reserved for Python-generated candidates.
  Verified cohorts rotate so one large family does not monopolize context.
- MiniMax M3 subdomain generation uses a fast non-reasoning profile by default.
  `providers.minimax.subdomain_thinking: default` restores server-default
  reasoning. Other models, ordinary chat and path generation are unchanged.
- Experimental `--vocabulary-breadth` / **Explore more naming patterns** is off
  by default. An eligible third pass asks for words fitting immutable one-slot
  layouts witnessed in verified names. Python owns structure, target scope,
  exclusions and expansion. It replaces a pass rather than adding a fourth.
  Empty or malformed paid answers do not trigger another paid fallback.
- Jev plan ranking is a separate, optional, off-by-default adapter. Its prediction
  ledger is additional to the generation ledger. It scores supplied plans; it
  neither proves DNS existence nor replaces open-ended generation.
- The dashboard shows DNS workers, provenance trees, inferred patterns,
  searchable findings, activity, resource limits, method-result counts and
  TXT/JSON/HTML exports. Method counts use the first-queued origin and are not
  exclusive causal attribution. Tree edges show discovery inspiration, not
  necessarily DNS delegation.
- Older checkpoints remain supported. Vocabulary options, model-pass rotation,
  cumulative spending, pending work and pattern paging survive resume.
  Capability-gated frontend fields remain compatible with older running servers.

## Resource and safety boundaries

- A bare scan is free. Hosted inference requires explicit selection or an
  explicit budget with a configured provider. Unknown model prices/custom
  endpoints require explicit price overrides.
- Shared limits cover total candidates, finding goals, rounds, time, DNS worker
  concurrency, DNS rate, depth and numeric alternatives. New scans continue
  through misses; `--until-stale` explicitly opts into an early no-hit stop.
- Rate counters represent logical resolver calls, including wildcard probes;
  resolver retries and transport fallback may produce additional packets.
- DNS calibration treats partial failures, conflicting answers and inconsistent
  wildcard probes conservatively. Address resolution does not establish HTTP
  reachability or a specific application's existence.
- Each model request durably reserves estimated cost before transmission.
  Ambiguous outcomes retain conservative charges, with no automatic paid retry.
  Reported usage beyond a reservation is recorded and stops later model calls.
  Dollar amounts are token estimates, not provider invoices.
- Successful model batches are saved before subsequent paid requests.
  Exhausting the AI budget can leave free discovery running within other limits.
- Tightening a numeric ceiling on resume leaves affected deterministic hypotheses
  dormant; it does not ban user-supplied, passive or model names containing larger
  numeric literals. Raising the limit can make dormant work eligible again.
- Checkpoints are private regular files outside the repository by default.
  Scope, counters, provider identity, metadata and budgets are validated before
  resume; the dashboard also detects stale concurrent state.
- Provider keys stay in memory, not browser storage, URLs, logs, checkpoints or
  reports. Optional subfinder has isolated configuration and does not inherit
  saved provider/model credentials.
- Dashboard requests enforce loopback binding, strict Host/Origin checks,
  bounded JSON input and a restrictive content policy. The service is not
  designed for public or multi-user deployment.
- Pause prevents new work; it cannot revoke an already-sent provider request.
  Strict process-level cancellation of synchronous model calls remains future work.

## Implementation map

| Component | Location |
|---|---|
| CLI scan and dashboard launch | `src/enumeraite/cli/scan_command.py`, `ui_command.py` |
| Sessions, queues and limits | `src/enumeraite/core/discovery_session.py` |
| Public evidence and input scope | `core/passive.py`, `core/scan_input.py` |
| DNS verification | `core/scan_dns.py` |
| Patterns, numeric plans and context | `core/subdomain_patterns.py`, `discovery_plans.py`, `discovery_context.py` |
| Optional vocabulary | `core/discovery_vocabulary.py`, `providers/llm_provider.py` |
| Spending and saved state | `core/scan_budget.py`, `core/scan_state.py` |
| Per-method accounting | `core/discovery_metrics.py` |
| Optional plan ranking | `providers/jev.py` |
| Local API and UI | `web/manager.py`, `web/server.py`, `web/assets/` |
| Offline regression coverage | `tests/core/`, `tests/providers/`, `tests/cli/`, `tests/web/` |

The earlier generation/path pipeline, OpenAPI benchmark support and provider
presets remain available. See [Benchmarking](BENCHMARKING.md) for reproducible
evaluation mechanics; public fixtures are not independent evidence of model
generalization.

## Verification

The complete suite passed 1,410 tests on Python 3.10 and 3.14. Python 3.10
coverage exceeded 86%. Tests include isolated frontend checks, simulated hosted
responses, budget failures, checkpoint compatibility, numeric boundaries,
wildcard calibration and subprocess isolation.

Black, isort, Ruff, mypy and dependency consistency checks passed. Source and
wheel builds include the new modules and all dashboard assets. No live API call
or target scan is required to run the test suite.

Repeat staged-tree and reachable-history secret checks before publication;
a passing test suite or a clean working-directory scan alone is insufficient.

## Next work

1. Add configurable passive-source adapters with explicit credentials,
   provider-specific limits, pagination, provenance and a data-access budget
   separate from model spending. Do not silently enable paid source calls.
2. Evaluate the production vocabulary option on multiple authorized domains
   under fixed DNS, time and token budgets. Measure new verified names and
   marginal contributions, not only candidate volume.
3. Keep adaptive search inputs separate from held-out evaluation. Do not promote
   negative, unverified or evaluation-only names into verified generation context.
4. Improve large-tree navigation and family collapsing.
5. Strengthen cancellation of already-running provider calls while retaining
   durable accounting and avoiding retries of uncertain paid requests.
6. Keep experimental scheduling and local decoding approaches optional until
   matched tests demonstrate useful improvements. A local exact-token-ID fork
   prototype now uses `llama.cpp` token arrays to compare alternate-token
   continuations with independent sampling. Its fixture and reports remain in
   ignored `bench-results/`; it is not a production feature. Evaluate verified
   marginal yield across multiple authorized targets before enabling it.

No discovery strategy guarantees additional names on every iteration or complete
coverage. Research artifacts and target-specific results are intentionally not
included in this handoff.
