# Practical benchmark for presentation

This benchmark creates a judge-ready comparison between baseline and hybrid with one explicit win rule:

- quality target reached (`val_ppl <= quality_target`)
- `tokens/sec` during generation
- peak memory (`peak_mem_mb`)
- estimated cost per 1M generated tokens (`estimated_cost_per_1m_tokens_usd`)

## Run

```bash
python3 -m presentation.practical_benchmark \
  --config configs/science_fair_s33.yaml \
  --baseline-ckpt results/checkpoints/s33/baseline_ctx1024_s33.pt \
  --hybrid-ckpt results/checkpoints/s33/hybrid_ctx1024_s33.pt \
  --quality-target 8.0 \
  --max-new-tokens 256 \
  --usd-per-hour 0.00 \
  --power-watts 60 \
  --usd-per-kwh 0.34
```

## Outputs

- `presentation/practical_benchmark_results.csv`
- `presentation/practical_benchmark_summary.txt`

## Notes

- For local runs, keep `--usd-per-hour 0` and set realistic `--power-watts`.
- For cloud runs, set both `--usd-per-hour` and power/electricity if you want combined compute+energy estimate.
- Winner rule: among models that meet quality target, lowest cost per 1M tokens wins.
