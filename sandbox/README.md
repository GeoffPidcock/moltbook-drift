# Experiment sandbox

This directory contains the supported path for reproducing or extending the
forced-choice feed-exposure experiment. Start with `notebook.ipynb`: it preserves
the project's notebook workflow while using tested functions from `runner.py`.
The numbered notebooks under `experiments/` remain evidence records.

## Notebook workflow

```bash
uv sync --frozen
uv run jupyter lab sandbox/notebook.ipynb
```

The notebook follows the structure of the completed experiments: configuration,
data loading, coverage, feed sampling, forced-choice construction, three Inspect
arms, scoring, and plotting. It reproduces the committed positive-control figure
without model calls. Its final paid-run cell is disabled by default.

## Configuration

`runner.py` defines the frozen settings in `ExperimentConfig` and the runnable
arms in `CONDITIONS`. The defaults pin:

- `thelionlies/moltbook-drift-data` at
  `b1bd75cd5f3d25c21cf3cf10ed43efd41cc9e6cd`;
- `openrouter/qwen/qwen3-32b`, served by DeepInfra fp8 without fallback;
- seed 42, one epoch, and 20 forced-choice decisions;
- 10 feed rounds of five unique posts;
- deterministic feed and option shuffling;
- no automatic retries and no decision chain-of-thought request.

The parser accepts exactly one final in-range letter, optionally prefixed with
`ANSWER:`. Parse failures are retained and excluded from both outcome numerators
and the valid-choice denominator.

## Automated offline validation

Place the pinned data repository beside this repository, or set
`MOLTBOOK_DATA_REPO`, then run:

```bash
uv sync --frozen
uv run python sandbox/runner.py --mode dry-run
uv run --with pytest python -m pytest tests/test_runner.py -q
```

The dry run refreshes `coverage.csv` and writes resolved configuration and fixture
output to ignored `local/runs/dry-run/`. It does not make model or network calls.

## Live run

Set `OPENROUTER_API_KEY`, select conditions explicitly, inspect their dry-run cost
estimates, and then confirm paid calls with `--yes`:

```bash
uv run python sandbox/runner.py --mode live \
  --conditions a_neutral poet_a --yes
```

The runner writes raw Inspect logs, `resolved-config.json`, and `results.csv` to
`local/runs/<timestamp>/`. This is a working area, not evidence by default. To
publish a run, move the complete folder into a numbered `experiments/` directory,
write a short provenance README, and add its result rows to
`experiments/results.csv`.

`runner.py` is the testable implementation behind the notebook, not a replacement for
the notebook workflow.
