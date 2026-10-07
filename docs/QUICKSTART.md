# Enumeraite quick start

## Install

```bash
# From the checkout containing the unreleased 0.2 changes
python -m pip install -e .
```

Python 3.10 or newer is required. The new `scan` workflow does not require an
API key, a local model, or a seed file.

## Use the live dashboard

```bash
enumeraite ui
# If the default local port is occupied:
enumeraite ui --port 0
# Print the local URL without opening a browser:
enumeraite ui --no-open
```

Enter an authorized root domain or a known subdomain. Review the displayed
discovery scope, then press **Start**. The default scan is free.
The dashboard shows current DNS queries (including wildcard probes), the
discovery tree, verified names, patterns being inferred, an activity timeline,
and shared resource limits. All branches use the same bounded DNS worker pool.
One target session runs at a time; earlier sessions remain available in history.

Entering `ny-01-iphone.example.com` uses it as an initial clue within
`example.com`, verifies it, then proposes related names. You can also enter a
root domain and add starting hostnames in **Add starting subdomains**. With no
supplied hostnames, automatic mode collects known hosts from public sources
first. With seeds, automatic mode skips that collection and focuses on them.
Under scan settings, choose whether to collect public names anyway, and set an
explicit **Discovery scope** if you are authorized for only a narrower zone.

For AI-assisted scanning, choose MiniMax, OpenAI, or Anthropic and enter that
provider's API key in the password field. Save it for the current dashboard
session, select a supported model and USD token budget, then start the scan.
The UI runs on your computer; model inference uses the hosted API, not a local
model. Verified names and recent scan feedback go to the selected provider to
interpret the target's context and generate candidates. Candidate details retain
the model's short explanation; these explanations are hypotheses, not findings.
Saving a key alone makes no provider call. Each key stays in the server's
process memory and is sent only to its provider's fixed HTTPS API endpoint when
AI inference is needed. It is not saved in checkpoints,
exports, logs, URLs, browser local storage, or browser session storage. No
separate dashboard access token is required. Clearing or
replacing a key affects future scan invocations, not a request already in flight.
Restarting the dashboard clears credentials; re-enter the key to resume AI work.

The **Patterns** tab also contains **Which methods are finding names?**. It
separates public-source names, common-name guesses, numeric and other patterns,
and AI proposals. Counts follow the method that first queued each unique name;
they do not claim that another method could not have found it. Unresolved DNS
checks and wildcard matches remain separate from verified findings and misses.

Under scan settings, **Explore more naming patterns** enables experimental AI
vocabulary expansion. Python supplies naming layouts witnessed in at least two
verified examples; AI suggests related words, and Python builds the candidates.
This replaces some existing AI passes, uses the same budget, and is off by
default. It can find complementary names, but does not improve every target.
An older running dashboard server will hide this option until restarted; restart
only after pausing its scan, and re-enter the session-only provider key afterward.

**Jev plan ranking** is a separate, off-by-default prediction layer. Add a
TypeSafe API key and an additional prediction budget (default $0.05 when
enabled). It can work alongside MiniMax/OpenAI/Anthropic or with deterministic
generation alone. Plan priorities, prediction cost, and fallback status are
shown separately. Jev evaluates small groups of naming hypotheses; Python
generates numbers and DNS verifies existence. Enabling it sends bounded target
names, verified evidence, plans, and recent DNS outcomes to TypeSafe. Its key
has the same memory-only lifetime as other dashboard keys.

The current hosted model menu includes MiniMax M2.7/M3, GPT-4.1 Mini/GPT-4.1,
and Claude Sonnet 4.6. These are explicitly supported options, not a claim to
list every provider model. Dated prices are taken from the providers' model
documentation; the scan budget remains a conservative token estimate, not an
invoice. See [GPT-4.1 Mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini),
[GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1), and
[Claude Sonnet 4.6](https://platform.claude.com/docs/en/models/sonnet-4-6/overview).

**Pause** saves the frontier. Already-running DNS/passive requests may take a
few seconds to finish; an active model request can take longer. The display says
**Pausing** until the worker has stopped. **Resume** keeps earlier findings,
attempts, rounds, and cumulative spending; increase exhausted limits when
needed. Closing a browser tab does not pause a scan. Ctrl-C in the terminal
requests a pause and stops the local server. JSON, plain-text findings, and
self-contained HTML tree exports are available from the dashboard.

Dashboard checkpoints default to `$XDG_STATE_HOME/enumeraite/ui/`, or
`~/.local/state/enumeraite/ui/`; `--state-dir DIRECTORY` selects another private
directory. Dashboard history lists its own saved sessions, not arbitrary CLI
checkpoint files. Reports contain target findings; share them deliberately.

Open the plain local URL directly; no dashboard token, login, or browser storage
is needed. The service still binds only `127.0.0.1`, rejects foreign
origins/hosts, and loads no external assets. Other local applications or users
on this computer can access it while it runs. It is a local desktop helper, not
a multi-user deployment: do not tunnel, reverse-proxy, or publish it.

## Scan a domain

```bash
enumeraite scan example.com
enumeraite scan example.com --find 100 --tree
enumeraite scan example.com --tree-file discovery.html -o subdomains.txt

# Start directly from a known naming example; no seed file needed
enumeraite scan ny-01-iphone.example.com --tree

# Explicit root scope and one or more starting hostnames
enumeraite scan example.com --seed ny-01-iphone.example.com --seed ca-02-iphone.example.com

# Keep sibling generation inside an authorized delegated zone
enumeraite scan ny-01-iphone.dev.example.com --scope dev.example.com

# Start with a hostname AND collect other known names
enumeraite scan ny-01-iphone.example.com --passive
```

Use a domain you are authorized to assess. Supply a DNS name rather than a URL.
With no seeds, the scan first collects historical subdomains from `crt.sh`
certificate transparency, optional installed `subfinder`, and MerkleMap when
configured. It verifies those names and common service
labels, filters wildcard DNS, and expands patterns from real findings. Public
sources are incomplete: failures and result caps are reported, and historical
names that no longer resolve are not counted as findings. This command performs
passive collection and DNS validation; it does not probe HTTP paths.

Hostname input is separated into its registrable domain and starting hostname
using a bundled, offline Public Suffix List, including private hosting suffixes.
Thus `ny-01.example.co.uk` stays under `example.co.uk`, and
`api.team.github.io` stays under `team.github.io`, not every GitHub Pages site.
No online lookup is made just to preview scope. Unrecognized/internal suffixes
are kept at the exact entered scope; use an explicit `--scope` to identify their
parent zone. Never infer authorization from an automatically selected boundary.
An explicit scope must contain the input and all seeds. Public suffixes alone
and out-of-scope seeds are rejected before scanning.
If a starting hostname is deeper than the configured discovery depth, increase
**Max discovery depth** / `--max-depth` or narrow the scope; it is rejected with
an explanation instead of being silently skipped.

A seed-first scan verifies supplied names before pattern or hosted-AI expansion,
without spending its initial candidate allowance on generic bootstrap names.
Dead, inconclusive, or wildcard seeds are not treated as verified context. If
none validate, it reports that outcome; enable public discovery or supply other
seeds to continue. `--seed` is repeatable, and `--seeds FILE` / `--seeds -`
still accept a list. On resume, saved scope, evidence and passive settings stay
in force unless an allowed setting is explicitly overridden.

### Optional subfinder

Install [ProjectDiscovery subfinder](https://docs.projectdiscovery.io/tools/subfinder/install)
separately to add it to passive collection. Enumeraite checks `PATH`, `GOBIN`,
`GOPATH` bin directories, `~/go/bin`, and standard Homebrew locations; it does
not install or update subfinder. When absent, `crt.sh` still works. The
integration runs keyless with isolated temporary configuration, not your saved
subfinder credentials or model API keys. HackerTarget is explicitly excluded.
It uses passive output
only, enforces the shared collection deadline and output/result caps, filters
every hostname to the scan scope, and reports failures or incomplete results.
Subfinder does not replace Enumeraite's DNS verification. CLI flags follow its
[official usage documentation](https://github.com/projectdiscovery/subfinder#usage).

For example, `ny-01-iphone.example.com` can suggest
`ny-02-iphone.example.com`, a product token seen on another related host, and
`api.ny-01-iphone.example.com`. Numeric slots work in the middle of labels and
keep zero padding. Observed same-slot tokens take priority over tentative
region, product, and environment vocabulary. One hostname alone cannot prove
that `ny` means a state or that `iphone` identifies a product. Every candidate
still needs DNS validation before it becomes evidence for another round.

### Set useful limits

```bash
# Stop at 100 verified names, or sooner if another limit is reached
enumeraite scan example.com --find 100

# Broader free discovery, still bounded
enumeraite scan example.com --time 15m --max-candidates 5000 --rounds 100

# All DNS workers, including wildcard probes, share the rate limit
enumeraite scan example.com --workers 20 --rate 25 --known-limit 500
```

| Option | Meaning | Default |
|---|---|---|
| `--find N` | Goal for unique verified names, including known/passive names | No finding goal |
| `--max-candidates N` | Total unique DNS candidates, including passive names | 1,000 |
| `--time 5m` | Time allowance per invocation, including passive discovery | 5 minutes |
| `--rounds N` | Total recursive expansion rounds | 10 |
| `--until-stale` | Stop after two rounds without findings; up to 100 rounds unless `--rounds` is set | Off |
| `--workers N` | Concurrent DNS workers | 20 |
| `--rate N` | Global DNS queries per second, including wildcard probes | 50 |
| `--known-limit N` | Maximum passive names retained | 200 |
| `--max-depth N` | Maximum discovery depth and labels below the target | 4 |
| `--max-number N` | Deterministic numeric-slot alternative ceiling, inclusive | 99 |

A goal is best effort, not a guarantee. New scans continue through DNS misses
while fresh candidates remain, until a configured limit or an exhausted frontier.
Use `--until-stale` to opt into stopping after two rounds without findings.
Resumed scans retain their saved stale-round policy. The tool cannot guarantee
complete enumeration. Branches share these limits; discovering another host
does not start an unbounded independent scan.

### Optional AI and dollar budgets

The default is `--provider none`, so a bare scan makes no paid model calls.
Configure a provider only when you want AI to supplement pattern expansion:

```bash
export MINIMAX_API_KEY='...'
enumeraite scan example.com --budget-usd 5
enumeraite scan example.com --provider minimax --budget-usd 5 --time 15m

# Optional: trade some ordinary AI passes for witnessed-layout vocabulary
enumeraite scan example.com --provider minimax --budget-usd 5 --vocabulary-breadth

# Choose another hosted provider explicitly
export OPENAI_API_KEY='...'
enumeraite scan example.com --provider openai --model gpt-4.1-mini --budget-usd 5
```

Supplying `--budget-usd` without a provider auto-selects a configured provider.
Explicit `--provider none` keeps the scan deterministic. An explicitly enabled
AI provider otherwise receives a default $1 token budget. If no suitable
provider is available, automatic selection reports the issue and continues
free discovery.

When a hosted provider is enabled, each expansion asks up to three complementary
questions, requesting at most 20 names per call (60 total), while reserving room
for Python-generated patterns. Small, rotating groups of verified names prevent
one large numeric family from dominating the prompt. The passes explore observed
vocabulary combinations, related services/products, and naming variations; Python
handles numeric ranges. Plausible lower-confidence guesses are eligible for DNS,
not silently discarded on a model-confidence threshold.

With `--vocabulary-breadth`, every third AI pass uses the vocabulary method when
its verified cohort supports a finite, nonnumeric naming layout. Otherwise that
pass uses ordinary name generation. The model cannot rewrite the target or
layout, and Python enforces scope, exclusions and candidate limits. Empty or
invalid paid responses do not trigger an extra paid fallback request. This is a
complementary experiment, not exhaustive token branching or reinforcement
learning. Resume preserves the option; `--no-vocabulary-breadth` disables it.

MiniMax M3 subdomain generation requests its faster, non-reasoning mode by default.
This does not change path generation, ordinary chat, MiniMax M2.7, or other
providers. To use M3's server-default reasoning behavior instead, set
`providers.minimax.subdomain_thinking: default` in your configuration; the default
value is `disabled`. Both modes still produce hypotheses that need DNS checks.

Each successful batch is checkpointed before the next paid request. All previous
names are excluded locally, and pattern pages skip previously tried names before
applying their output limit. Both use verified evidence; AI proposals become
evidence only after DNS validation. Every request still obeys the shared dollar
budget and scan limits. More calls may use the budget faster; these are not free
alternative outputs from one model request. Activity shows the current AI focus.

The budget caps conservative token-cost estimates, not a provider invoice.
Each request reserves estimated prompt cost and its completion allowance
before transmission, then reconciles reported usage. Uncertain requests retain
their reserved charge and are not automatically retried. If returned usage
exceeds the reservation, the ledger records it and blocks further model calls.
Exhausting the model budget still allows free discovery within the other limits.

To add Jev predictions, set `TYPESAFE_API_KEY` through your environment or secret
manager, then run:

```bash
enumeraite scan example.com --predictor jev --prediction-budget-usd 0.05
enumeraite scan example.com --provider minimax --budget-usd 1 \
  --predictor jev --prediction-budget-usd 0.05
enumeraite scan example.com --resume --prediction-budget-usd 0.10
```

The Jev budget is **additional** to `--budget-usd`, not part of it. The combined
example permits $1.05 total estimated token cost. Resume retains predictor
identity and cumulative spending; use a new scan to enable/disable the layer.
Jev is pinned to `jev-1.13.0`; [official rates](https://docs.typesafe.ai/models),
checked 2026-10-03, are $0.042 per million input tokens with free output. Each
request conservatively reserves the entire supported 65,536-token input bound
(about $0.002753) before transmission, then reconciles usage. A smaller remaining
budget cannot admit a call even if its eventual charge might have been lower.
Unavailable, invalid, or unaffordable predictions fall back to deterministic
ordering, with no automatic paid retry. Actual benefits require workload-specific
evaluation; a high score is not a verified finding.

Numeric families with at least two verified examples enumerate one changing
slot while preserving the other literals. The existing name history tracks
checked values, including misses, across resume. Gaps and continuations are
generated in Python in bounded waves; `--max-number` (default 99) remains the
inclusive ceiling for changed numeric slots regardless of padding. It is not a
ban on all numbers in hostnames: unchanged literals (such as `209` in an
IP-shaped name), observed child-label vocabulary, explicit input, passive names,
and AI proposals are unaffected. Lowering this ceiling on resume leaves queued
deterministic numeric alternatives above it dormant; raising it later makes
that saved work eligible again, without forgetting prior DNS results.
Generic child guesses under such a family initially sample two representative
hosts per child label; confirmed
children permit further exploration. This does not promise exhaustive coverage.

The built-in price registry is deliberately small. Unknown models and custom
endpoints require both `--input-price` and `--output-price`, using the provider's
current USD-per-million-token rates. Local loopback presets have zero token
charges; machine costs, subscriptions, and other provider fees are outside the
estimate.

The scan passes its remaining time to the model SDK and admits no later request
after the deadline. A synchronous request already in flight uses SDK I/O
timeouts and can take longer to unwind; `--time` is not a hard process-kill
deadline for that request.

### Resume and supply your own names

```bash
enumeraite scan example.com --resume
enumeraite scan example.com --resume --time 10m --max-candidates 5000 --rounds 20
enumeraite scan example.com --state saved-scan.json
enumeraite scan example.com --state saved-scan.json --resume
enumeraite scan example.com --seeds known-subs.txt --no-passive
cat known-subs.txt | enumeraite scan example.com --seeds - --no-passive
```

Private checkpoints are written automatically under
`$XDG_STATE_HOME/enumeraite/<target>/`, falling back to
`~/.local/state/enumeraite/<target>/`. Resume selects the latest scan for that
target unless `--state` selects a file. It preserves previous attempts,
findings, expansion rounds, queued work, and estimated spend. `--time` starts
a fresh allowance for each invocation. A resumed `--budget-usd 5` means a $5
total session budget, not another $5. The saved AI provider/model is reused
unless you explicitly request `--provider none`.

`--seeds FILE` adds names to passive evidence. `--no-passive` skips public
sources but retains common service bootstrap and recursive discovery. Seed
files/stdin use one name per line; blank lines and `#` comments are ignored.

### Output and trees

```bash
enumeraite scan example.com --tree > verified.txt
enumeraite scan example.com --json -o scan-report.json
enumeraite scan example.com --tree-file discovery.html
```

Progress and the terminal tree go to stderr. Default stdout contains only
verified names, one per line; `--json` emits one structured report instead.
`--quiet` suppresses progress, and `-o` writes the selected output to a file.
On resume the output includes the earlier findings as well as new discoveries.
The HTML report is self-contained and includes attempts, validation results,
inferred patterns, and discovery parents. Its edges show why a name was tried,
rather than claiming that every edge is a DNS delegation. Reports and saved
states contain findings, so keep them outside source control unless intended.

## Seed-driven generation

The existing `run` workflow supports both paths and subdomains. Configure a
hosted provider or start a local OpenAI-compatible server:

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
