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

Enumeraite discovers a domain's known subdomains, checks DNS, and expands the naming patterns it finds. Start with a domain or a known subdomain; optionally add a hosted model with a token-spending budget. The seed-driven `run` workflow also supports target-specific path and subdomain generation.

The pipeline combines target context, structured proposals, validation, feedback, and measurement against held-out ground truth.

## Quick start

```bash
# Install from the checkout containing the unreleased 0.2 changes
python -m pip install -e .

# Open the local live dashboard
enumeraite ui

# No seed file or API key needed
enumeraite scan example.com
enumeraite scan example.com --find 100 --tree

# Start with a known hostname and generate related names within example.com
enumeraite scan ny-01-iphone.example.com --tree

# Save an interactive discovery tree and verified names
enumeraite scan example.com --tree-file discovery.html -o subdomains.txt
```

The dashboard accepts a root domain or a starting subdomain, previews the scan
scope before starting, and lets you pause or resume. Watch
actual DNS workers, wildcard probes, a growing discovery tree, findings, inferred
patterns, and the estimated model-token spend. Free discovery is the default.
To add hosted AI, enter a MiniMax, OpenAI, or Anthropic key in the dashboard's
session-only credential field, select that provider, and set a budget before
starting. No local model runtime is needed. Verified names and validation
feedback are sent to the selected API for context-aware candidate generation;
candidate explanations remain hypotheses until DNS verifies them. Keys are not written to
checkpoints or browser storage; restarting the server requires entering one again.
Open the plain local URL without a dashboard token or login. Do not expose this
local-only server publicly; other applications on your computer can access it.

Select a hostname to inspect **Origin & proposal evidence** separately from its
DNS results: exact public collectors and saved source records, or the generating
pattern/model, rationale, and verified seed cohort. Certificate dates and
MerkleMap first-seen timestamps are not current DNS evidence. Older scans expose their
saved collector memberships but explicitly label details they did not record.

Verified DNS answers can also expose in-scope CNAME targets. These appear as
**Observed CNAME targets** in the evidence panel and become candidates for
independent DNS validation; merely appearing in a CNAME chain does not count as
a finding. Literal parent hostnames of verified names are also checked, rather
than assumed to resolve. Both paths obey the same scope, depth, wildcard, and
candidate limits as other discovery.

Without supplied subdomains, `scan` first collects historical names from
certificate transparency (`crt.sh`), optional keyless `subfinder` if installed,
and MerkleMap if configured, then validates them and common service
names. With a starting
subdomain, it verifies that seed and expands its naming context; public-source
collection is skipped by default but can be enabled with `--passive`.
Use `--scope dev.example.com` to keep a scan inside a narrower zone, or
`--seed HOST` / `--seeds FILE` to supply additional starting names. Every verified
discovery can guide further expansion. Passive sources may fail, truncate, or
contain stale names; those limits are reported, and only current DNS findings
appear on stdout. Scanning uses passive lookups and DNS, not HTTP path probing.
HackerTarget is not queried directly and is explicitly excluded from subfinder.
When intake reaches its remaining candidate capacity, passive names are sampled
across naming families instead of admitting only the earliest collector output.
The crt.sh collector combines certificate `name_value` and `common_name` fields;
malformed individual records do not discard other usable observations, but the
response is reported as partial.

To add your paid MerkleMap search subscription, add a top-level
`"MERKLEMAP_API_KEY": "..."` string to a supported JSON config, such as
`~/.config/enumeraite/config.json`. Alternatively, export the key:

```bash
export MERKLEMAP_API_KEY='...'
enumeraite scan example.com --passive
# Or: enumeraite ui
```

The selected JSON file takes precedence over the environment; an explicit blank
file value disables MerkleMap. Without either key source, it is skipped. `.env`
files are not loaded automatically. Protect a credential-bearing config with
`chmod 600`. MerkleMap uses your API quota separately from AI budgets; its key is
not saved in scan checkpoints or reports.

The default uses deterministic generation (`--provider none`), bounded by
1,000 unique DNS candidates, five minutes, and ten expansion rounds. A finding
goal is best effort: `--find 100` counts unique verified subdomains, including
previously known names, and other limits can stop the scan sooner. No scan can
promise to find every subdomain.

Resuming after increasing the candidate cap can admit saved CNAME targets that
previously did not fit. Checkpoints from the earlier pattern-paging version also
make newly supported expansions eligible once, without resetting completed DNS
attempts or recorded model spend. Existing round, time, depth, and budget limits
still apply.

For AI-assisted expansion, configure a provider and request a budget:

```bash
export MINIMAX_API_KEY='...'
enumeraite scan example.com --budget-usd 5 --time 15m --max-candidates 5000

# Choose a model explicitly, or keep all discovery deterministic
enumeraite scan example.com --provider minimax --budget-usd 5
enumeraite scan example.com --provider none --time 15m
```

An explicit dollar budget auto-selects a configured provider unless you also
specify `--provider none`. Token charges are conservatively estimated and
reserved before each call; they are not provider invoices. Unknown models or
custom endpoints need both `--input-price` and `--output-price`, in USD per
million tokens. If the model budget runs out, free pattern discovery continues
within the other limits. See [Quick start](docs/QUICKSTART.md) for resume, limits,
and the in-flight model-request timeout caveat.

### Optional Jev plan ranking

Jev (TypeSafe) can rank the next discovery plans alongside your generative
provider, or with deterministic generation alone. It does not generate text or
numbers: Python fills numeric gaps and extends supported ranges, while DNS
verifies every candidate. In the dashboard, enable **Jev plan ranking**, save a
separate TypeSafe session key, and set its **additional** prediction budget.

For the CLI, set `TYPESAFE_API_KEY` privately, then use:

```bash
enumeraite scan example.com --predictor jev --prediction-budget-usd 0.05
enumeraite scan example.com --provider minimax --budget-usd 1 \
  --predictor jev --prediction-budget-usd 0.05
```

The second example allows $1 for generation **plus** $0.05 for Jev prediction.
Both are cumulative estimates, preserved on resume. Enabling prediction sends
bounded target names, naming plans, and DNS feedback to TypeSafe's fixed HTTPS
endpoint. Failed/unaffordable prediction falls back to deterministic priorities.
Scores indicate plan usefulness, not proof that a hostname exists. The pinned
model and dated rates come from [TypeSafe's documentation](https://docs.typesafe.ai/models).

For the existing seed-driven workflow:

```bash
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

- Seed-free `scan TARGET`: passive collection, DNS verification, recursive
  naming-pattern expansion, optional budgeted AI, private checkpoints, and
  terminal/interactive HTML discovery trees.
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

### Discover subdomains from a domain

```bash
enumeraite scan example.com --find 100 --tree
enumeraite scan example.com --time 30m --rounds 100 --max-candidates 5000
enumeraite scan example.com --workers 20 --rate 25 --known-limit 500
enumeraite scan example.com --resume --time 10m --max-candidates 5000 --rounds 20
enumeraite scan example.com --max-candidates 0 --rounds 0 --time 0
enumeraite scan example.com --seeds known-subs.txt --no-passive
enumeraite scan example.com --json -o scan-report.json
```

A name such as `ny-01-iphone.example.com` suggests a zero-padded numeric slot
in the middle of a label. The scan tries observed alternatives first, then
bounded numeric, region, product, and environment hypotheses, and deeper DNS
labels. New hits join the evidence for the next round. A single `ny` or `iphone`
token does not prove a state or product convention; these meanings remain
explicitly tentative. This feedback loop does not train model weights.

New scans favor breadth: they continue through misses while fresh hypotheses and
scan capacity remain. Hosted AI rotates small, diverse verified-name cohorts
through up to three complementary questions per round; Python expands numeric
ranges and observed combinations, skipping names already tried. Every retained
guess is eligible for DNS regardless of its model confidence. Use `--until-stale`
if you prefer to stop after two rounds without findings. All calls still share
the configured AI budget; all DNS workers share the rate and scan limits.

DNS workers share one global rate limit, including wildcard checks, and one
discovery queue rather than multiplying the limits for every host. The candidate
cap includes passive names. `--max-depth` bounds discovery depth and nested DNS
labels; `--max-number` bounds numeric alternatives while preserving padding.

For open-ended discovery, choose **Run until paused** in the dashboard's start
or resume settings, or use `--max-candidates 0 --rounds 0 --time 0` in the CLI.
These remove total-work stops, not DNS rate/concurrency, scope, depth, numeric,
or explicit spending controls. Leave the finding goal empty to remove that
stop too. You can rerun a target without a saved-session quota. Discovery can
still exhaust its available hypotheses, and larger scans use more memory and
disk. Existing finite defaults remain unchanged.

Checkpoints are saved automatically under
`$XDG_STATE_HOME/enumeraite/<target>/`, or `~/.local/state/enumeraite/<target>/`
when that variable is unset. Use `--state FILE` to choose a path. Resume retains
attempts, findings, rounds, and estimated spending; `--time` provides a fresh
allowance for each invocation. `--tree` and progress use stderr, so stdout stays
one verified name per line. `--tree-file FILE` creates a self-contained HTML
report whose edges show which discovery prompted each hypothesis.

### Generate from existing seeds

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
