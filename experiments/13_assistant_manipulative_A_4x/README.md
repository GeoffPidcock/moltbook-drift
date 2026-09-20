# What this is
Four repeats of a persona drift eval leveraging malicious moltbook data for treatment feeds and non-assistant forced choice options. 

# How to reproduce
Using the sandbox runner, at an approximate cost of ~$2, running the following command.

```bash
OUT="$PWD/experiments/13_assistant_manipulative_A_4x"

for s in 42 43 44 45; do
  echo "=== seed $s ==="
  uv run python sandbox/runner.py --mode live \
    --conditions a_manipulative_baseline a_manipulative_neutral manipulative_a \
    --seed "$s" --output-dir "$OUT" --yes
done
```