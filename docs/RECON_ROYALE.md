# Recon Royale integration

Enumeraite can prepare a Recon Royale submission and, only with `--submit`, send it through the documented API.

## Safety model

The command:

1. Fetches the current round's target with the user's API key.
2. Removes names outside that exact apex.
3. Generates candidates through the bounded feedback loop.
4. Resolves the passive baseline and generated union locally.
5. Probes random names to suppress wildcard DNS.
6. Separately calculates union valid rate, Enumeraite valid rate, and Enumeraite's marginal valid findings beyond the passive baseline.
7. Refuses submission when Enumeraite itself is below `--min-valid-rate` (20% by default) or adds no marginal validated finding; a strong passive baseline cannot mask poor generation.
8. Performs no POST unless `--submit` is explicitly present.

## Usage

```bash
export RECON_ROYALE_API_KEY='...'

# Review only
enumeraite royale \
  --seeds current-target-seeds.txt \
  --baseline passive-results.txt \
  --provider ollama \
  --count 200 --rounds 3

# Send the locally validated list after review
enumeraite royale \
  --seeds current-target-seeds.txt \
  --baseline passive-results.txt \
  --provider ollama \
  --submit
```

The API key is read only from `RECON_ROYALE_API_KEY` and is never written to the output.

## Automation gate

This repository intentionally does not ship an active scheduled workflow. First run representative offline benchmarks and several reviewed dry runs. Enable a nightly job only when precision remains above the chosen threshold across targets and the marginal contribution is consistently positive.

Leaderboard position is not the useful research metric. Record valid subdomains added by Enumeraite beyond the same passive baseline in each round.
