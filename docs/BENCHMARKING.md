# Benchmarking Enumeraite

The benchmark answers a narrow question: given the same visible seed and candidate budget, how many held-out real paths or subdomains does each engine recover?

LLMs, structural inference, wordlists, and mutation scripts use the same dataset split and scoring code.

## Datasets

### OpenAPI paths

Point `--specs` at a directory containing JSON, YAML, or YML OpenAPI documents. Files with fewer than four paths are skipped. The specification's declared paths are complete ground truth for that document.

```bash
enumeraite bench openapi \
  --specs /data/openapi-directory/APIs \
  --seed-frac 0.2 --salt experiment-01 \
  --seed-limit 50 --repeats 5 --limit 500
```

OpenAPI template segments such as `{petId}`, `:id`, and `<id>` match concrete generated values with the same segment count. ID-named parameters require an ID-looking value, so `/pets/create` does not satisfy `/pets/{petId}`. Equivalent parameter-name aliases such as `/users/{id}` and `/users/{userId}` are collapsed before splitting, so one route shape cannot leak from the seed into held-out truth. A candidate covers at most one route shape: exact literal routes take precedence, followed by the most specific matching template. Emitting many concrete IDs for one template therefore counts once. Paths remain case-sensitive, as OpenAPI field and path matching requires; DNS names are lowercased.

### Subdomains

Create one `<apex>.txt` file per target, with one confirmed FQDN per line. Build the truth set independently using multiple passive sources, certificate transparency, DNS resolution, and deduplication.

```bash
enumeraite bench subdomain \
  --truth /data/subdomain-truth \
  --seed-frac 0.5 --salt experiment-01
```

The harness cannot know whether a subdomain truth file is complete. Results inherit the limitations of the collection process.

`--repeats` creates deterministic alternative held-out splits using the declared salt. `--seed-limit` prevents very large specifications from exposing hundreds of routes to the model while small specifications expose only a handful. Every result records the split salt, source SHA-256, full known/hidden manifest, runtime, and Git revision so saved generations can be rescored after a metric correction without another provider call.

## Controls

Every default benchmark now reports three deterministic ablations:

- `structural` performs only seed-derived gap completion (ancestors, observed grids, and adjacent numbered instances);
- `mutation-only` performs lexical plural/singular changes, version bumps, CRUD/action suffixes, subresources, environment swaps, numbered-instance changes, and common subdomain prefixes;
- `mutation` is the structural-first union of both and remains the reference used for paired deltas.

This separation matters: a combined control can beat a model while hiding whether the gain came from high-precision structural evidence or broad low-precision mutation. The structural control may return far fewer than the requested budget, so interpret its precision alongside fill rate.

For a wordlist control, pass the union you actually want to beat:

```bash
--wordlist /data/seclists-assetnote-union.txt
```

A small wordlist makes ordinary mutations appear novel. Do not publish a `novel_valid` result without documenting and distributing the exact wordlist hash.

## Metrics

- Recall@budget: distinct hidden ground-truth shapes recovered among the top N candidates, ranked by the engine's confidence.
- Precision@budget and overall precision: distinct held-out shapes recovered divided by distinct non-seed candidates. Template repetitions cannot inflate the numerator.
- Average precision and reciprocal rank: whether true discoveries appear early, not merely somewhere in the candidate set.
- Fill rate: distinct non-seed candidates returned divided by the requested budget.
- Brier score: squared error of the model confidence against whether each ranked candidate adds a new held-out shape. Lower is better; treat confidence as a ranking signal until it is calibrated on representative private data.
- Novel valid and wordlist unique: valid candidates absent from the supplied wordlist, with exact supplied wordlist hashes recorded.
- Cost per valid: measured tokens, USD estimate, and wall time divided by valid candidate hits.

Scoring hard-caps every engine at the raw emission boundary, before confidence
ranking. A custom engine cannot gain extra truth matches—or additional candidates
that compete for a top rank—by returning an oversized list, and fill rate therefore
cannot exceed 1.0. Results using this contract carry metric version
`route-shape-v10`. Version 10 also refuses to credit literal OpenAPI placeholders
as concrete discoveries. It clusters repeated or copied specifications by
their content SHA-256, so the same document under multiple paths cannot narrow
confidence intervals by masquerading as independent evidence.

OpenAI-compatible and Anthropic responses record cached input tokens when the backend exposes them. The small built-in USD table applies a cache-read rate only for models with an explicitly recorded rate; unknown models conservatively use their ordinary input rate. Treat this table as experiment metadata and update it when provider pricing changes.

Every result records the pricing source and date checked. The built-in MiniMax M3 price applies to contexts at or below 512K tokens; long-context tiers require an updated experiment-specific table.

Recall is macro-averaged across targets and reported with a small-sample Student-t 95% interval. Repeated splits of one source are averaged into one confidence-interval cluster, so repeats do not masquerade as independent targets and make the interval artificially narrow. The report also gives a source-clustered, paired 95% interval for each engine's recall delta versus mutation; this is more informative than comparing two overlapping standalone intervals. Precision is micro-averaged across candidates. Always report the recall curve at multiple budgets rather than selecting one favorable point. Report fill rate and provider errors too: a high precision from returning only a few candidates is not an equivalent result.

A failed provider case contributes zero to macro recall, average precision, reciprocal rank, and fill rate; it is not silently dropped. Brier calibration excludes failed cases because no confidence predictions exist to score.
Failed calls retain measured wall time. Token/USD cost may still be incomplete when a provider raises before returning usage metadata, so failures must be examined alongside the aggregate cost table.

The report includes each model's delta in distinct hits and paired recall versus the mutation control. That marginal result is generally more actionable than an isolated model score because a good deterministic candidate pool is cheap, fast, and reproducible.

Hybrid engine telemetry also records how many retained outputs came from the deterministic structural pool versus the model. This makes it possible to audit whether paid inference actually survives ranking rather than attributing all hybrid coverage to the model.

The aggregate report separates valid shapes covered by each candidate source and "model-only" shapes not covered by any structural proposal. This is the cleanest measure of inference's marginal contribution inside a hybrid run.

## Replay after metric changes

Saved runs contain the complete dataset manifest and ranked candidate traces. Rescore them with the current metric implementation without another model call:

```bash
enumeraite bench replay bench-results/experiment-01/results.json \
  -o bench-results/experiment-01-replayed
```

If the source run used `--wordlist`, replay requires the same file and verifies its SHA-256 before scoring:

```bash
enumeraite bench replay old/results.json \
  --wordlist /data/seclists-assetnote-union.txt -o new-results
```

Replay refuses results whose candidate trace was truncated, malformed, or no longer
aligned with the saved dataset manifest, because silently rescoring a different trace
would change precision and fill rate. Failed provider cases retain their original
error and recorded costs. The new result records the source file hash and source
metric version.

## Confidence experiment

Run the same cases with identical model and salt:

```bash
ENUMERAITE_CONFIDENCE=model enumeraite bench openapi ... -o bench-results/model
ENUMERAITE_CONFIDENCE=heuristic enumeraite bench openapi ... -o bench-results/heuristic
ENUMERAITE_CONFIDENCE=logprob enumeraite bench openapi ... -o bench-results/logprob
ENUMERAITE_CONFIDENCE=uniform enumeraite bench openapi ... -o bench-results/uniform
```

Not every backend exposes log probabilities. When it does not, logprob mode returns a neutral score and therefore behaves like generation order.

## Reproducibility checklist

- Record Enumeraite commit, model identifier, provider, temperature, seed fraction, salt, candidate budgets, and dataset revision.
- Keep hidden items out of every prompt and context source.
- Use the same target set for every engine.
- Report provider failures and timeouts; do not silently drop failed cases.
- Pin or archive the external wordlist and record a cryptographic hash.
- Treat the included demo fixture only as a smoke test.
- Label synthetic fixtures as regression tests, not evidence of real-world performance.
- Treat results on famous public specifications (for example Petstore, GitHub, Stripe, Kubernetes, or crAPI) as engineering smoke tests: hosted models may have seen them during training. Use private or temporally held-out specifications for publishable model comparisons.

Large datasets and outputs belong outside Git or under the ignored `bench-data/` and `bench-results/` directories.

## Production-session replay with incomplete DNS inventories

`tools/recon_bench.py` measures the actual `DiscoverySession` scheduler against
archived DNS observations. Unlike `enumeraite bench subdomain`, it exercises
iterative production discovery and scores candidates in their **query order**,
not a confidence-reranked proposal list. It runs offline, without a model,
passive collectors, or scan-history intake. Generation sees visible seeds and
the outcomes of candidates it asks the archived oracle to validate.

Save scan snapshots such as `GET /api/scans/<id>` responses. Each needs a
`target` and a `nodes` list or map. Create a JSON manifest containing their
paths, resolved relative to the working directory:

```json
[
  {"path": "bench-data/development-scan.json"},
  {"path": "bench-data/held-out-scan.json"}
]
```

Each manifest entry may also supply `sha256`; a mismatch fails rather than
silently changing the source inventory. A positive observation must have an
attempt, valid IP addresses, no error, and acceptable wildcard evidence when
that evidence was recorded. Legacy snapshots lacking wildcard detail are
accepted as historical observations, not equivalent to fresh validation.
Targets with fewer than four verified names cannot form cases; requested test
targets below that threshold fail the freeze.

```bash
python tools/recon_bench.py freeze \
  --manifest bench-data/scan-manifest.json \
  --test-target heldout.example \
  --seed-fraction 0.35 --seed-limit 100 --repeats 5 \
  --salt recon-experiment-01 \
  --output bench-results/recon/dataset.json

python tools/recon_bench.py replay \
  --dataset bench-results/recon/dataset.json --partition development \
  --label candidate \
  --budgets 100 250 500 1000 2500 \
  --max-depth 5 --max-number 99 --case-seconds 120 \
  --output bench-results/recon/development.json
```

Freeze before tuning. Repeat `--test-target` for every DNS zone belonging to
the held-out organizations: the tool does not infer organization ownership.
Repeated splits of a target, and related zones within an organization, are not
independent replications. A held-out target that a hosted model may have seen in
training is not proof of model-training isolation.

After selecting on development data, record each source version's fingerprint
**before** opening the test partition:

```bash
python tools/recon_bench.py fingerprint > bench-results/recon/selected-source.sha256

python tools/recon_bench.py replay \
  --dataset bench-results/recon/dataset.json --partition test \
  --label selected \
  --frozen-source-sha256 "$(cat bench-results/recon/selected-source.sha256)" \
  --budgets 100 250 500 1000 2500 \
  --max-depth 5 --max-number 99 --case-seconds 120 \
  --output bench-results/recon/final-test.json
```

Test replay requires that saved fingerprint to match the imported package.
Run baseline and selected code separately, using identical dataset bytes,
budgets, depth, number range, and time limits. The fingerprint covers the
package's Python files; results separately record the runner and dataset
SHA-256 values. Archive the dependency versions and interpreter as well.
Reusing test results to choose another variant requires a new holdout;
recomputing a fingerprint does not restore blindness.

Budgets count unique **non-seed** candidate attempts, between 1 and 5,000.
The session's total candidate cap adds the visible seeds to the largest
requested budget. Every visible seed is depth-validated before any case runs;
an insufficient `--max-depth` fails instead of silently changing the seed
cohort. Frozen datasets cannot be overwritten.

Unknown candidates are **inconclusive**, not invented DNS negatives. Historical
positives take precedence over later transient misses when freezing the
inventory, so recall is historical recovery, not current DNS availability.
The reported `known_hit_rate_lower_bound_at_budget` is only a lower bound for
an incomplete inventory. Inspect candidate traces, stop reasons, and the full
recall curve; do not treat early termination as an equal-budget completed run.

This archived oracle does not reproduce live CNAME chains, wildcard probes,
resolver costs, or DNS changes. Follow a selected change with a bounded live
comparison using the same seeds and candidate allowance. Report verified names
beyond seeds, names genuinely absent from the pre-campaign inventory, resolver
calls including wildcard probes, and model spend separately. Neither a
certificate name nor a model-generated candidate is a verified discovery.
