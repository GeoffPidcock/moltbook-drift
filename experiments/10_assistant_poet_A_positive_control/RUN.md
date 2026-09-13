# Poet-A positive-control run — 2026-08-06

This run is preserved with its raw Inspect logs. Its assistant-neutral feed contained 50 distinct
source rows and only 49 unique content strings. Under the repository's unique-content rule, the
baseline and poet-exposed arms match the current method; the neutral arm is a documented imperfect
control.

Configuration: repository `7b0592beb6573d7aca562e4377884fed672da9c3`; pinned data
`b1bd75cd5f3d25c21cf3cf10ed43efd41cc9e6cd`; Qwen3-32B through OpenRouter; DeepInfra fp8,
no fallback; category A; assistant/poet/pirate options; 20 prompts; seed 42; one epoch; 10 rounds ×
5 posts; reasoning/history defaults retained; decision CoT off. The source notebook is
[`notebook.ipynb`](notebook.ipynb)
and its SHA-256 at inventory time was
`a608112cb5c8bf98da646763639ebd1dc1823e36f90f4bc5525e1ea75675956b`.

| Arm | Assistant | Poet | Pirate | Non-assistant | Tokens in/out | Cost |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 20 | 0 | 0 | 0/20 | 12,047 / 8,530 | $0.00335216 |
| assistant-neutral | 18 | 2 | 0 | 2/20 | 1,604,469 / 102,632 | $0.15709448 |
| poet | 2 | 17 | 1 | 18/20 | 1,743,476 / 135,660 | $0.17746288 |

Aggregate cost `$0.33790952` exactly matched the OpenRouter credit API usage snapshot.

## Log checksums

- baseline: `0c6cdd9f90b8af394d432b04b4ab08a4306d04e608101f95faee0cc20e6a6838`
- neutral: `4a810bb25b74ebaea87bacbb0eff1cbba0bd3d1f9d5b2271a245e000a3ef04a3`
- poet: `f1110f16f6d77b8c722319ba650ce9363b24d506c6e6632ea19a30f1614e5754`
