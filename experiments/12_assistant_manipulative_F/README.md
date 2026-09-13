# Manipulative-F

This experiment completes the previously missing Promotion-category check for the
malicious-manipulative persona. It compares a no-feed baseline, a category- and
option-matched assistant-feed control, and a manipulative-feed exposure.

## Results

| Arm | Non-assistant choices | Valid decisions |
|---|---:|---:|
| Baseline | 1 mild / 20 | 20 |
| Neutral assistant feed | 0 / 20 | 20 |
| Manipulative feed | 0 / 20 | 20 |

The content contrast (exposed minus neutral) is `0.00` in this run. The single
baseline deviation was `malicious-manipulative-mild`; no arm selected the high
option. This small run does not establish a null effect.

**Category F** is the Promotion topic stratum. Holding category fixed matters:
promotion prompts and available posts may differ systematically from Identity,
Technology, and Economics. The result therefore provides a category-F check; it
does not by itself justify assuming one common effect across A, B, D, and F.

## Reproduction record

[`run/`](run/) contains the resolved configuration, per-arm summary, and raw
Inspect logs. [`results.csv`](results.csv) is the three-row public summary. The
configuration matches experiment 11: pinned data, Qwen3-32B through DeepInfra fp8
without fallback, seed 42, 20 decisions per arm, 10 × 5 unique-post feeds, a
1,024-token feed-reaction cap, and a 2,048-token decision cap.

Use [`../../sandbox/notebook.ipynb`](../../sandbox/notebook.ipynb) with:

```python
selected = [
    "f_manipulative_baseline",
    "f_manipulative_neutral",
    "manipulative_f",
]
```

Or run the same tested implementation directly:

```bash
uv run python sandbox/runner.py --mode dry-run \
  --conditions f_manipulative_baseline f_manipulative_neutral manipulative_f --seed 42
uv run python sandbox/runner.py --mode live \
  --conditions f_manipulative_baseline f_manipulative_neutral manipulative_f --seed 42 --yes
```

Raw-log SHA-256 checksums:

```text
f730dca783b9b85d01b3fd3c7c838ce8f8b789eae27b0af1b3808d6e81a0286f  run/f_manipulative_baseline/2026-09-13T09-14-08-00-00_task_ZQETYiz3qMZRSbXhPEssvv.eval
584950917d13d4c502ad8a0920aebaaa5d479517d96587fff4236e2ee5870423  run/f_manipulative_neutral/2026-09-13T09-15-45-00-00_task_mSzTuWAuicWnDLRnrCLXcX.eval
cb316d84acf5c043908f4cb6955fc92a4863e3ac67bc04b07e2b53c5dd0524ac  run/manipulative_f/2026-09-13T09-30-16-00-00_task_o76oSfWFFYKS3RpoNJWLnr.eval
```
