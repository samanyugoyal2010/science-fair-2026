# Split-Stream Hybrid LM MVP Report

## 1. Objective
Evaluate pilot-scale tradeoffs between a split-stream hybrid LM and a parameter-matched transformer baseline on:
- validation perplexity (quality),
- peak memory (efficiency),
- throughput (step time / tokens per second).

## 2. Experimental Controls
- Same tokenizer, data split, optimizer, LR, train steps, seeds.
- Parameter-matched comparison (target <0.5% param gap).

## 3. Main Results
- See `results/raw/runs.csv` and `results/processed/summary.csv`.
- Paired seed deltas are in `results/processed/paired_deltas.csv`.
- Paired aggregate outcomes are in `results/processed/paired_summary.csv`.

## 4. Ablations
- local_only, ssm_only, sum_fusion.
- See `results/raw/ablations.csv` and `results/processed/ablation_summary.csv`.

## 5. Statistical Analysis
- Mean, std, and 95% CI are reported.
- Inference guard: if replication count is <3, results are explicitly marked as underpowered/pilot and should not be presented as conclusive.

## 6. Claim Decision Rule
- Multi-objective reporting (not a single binary pass/fail):
  - quality_direction from val PPL paired delta (lower is better),
  - memory_direction from peak memory paired delta (lower is better),
  - speed_direction from throughput paired delta (higher is better).
- Evidence labels:
  - replicated (`n>=3`),
  - underpowered (`n=2`),
  - pilot (`n<=1`).

## 7. Figures
- `results/figures/memory_vs_context.png`
- `results/figures/ppl_vs_context.png`
- `results/figures/efficiency_frontier.png`
- `results/figures/ablation_bars.png`

## 8. Limitations (Mandatory Disclosure)
- This is a pilot-scale study on a small model and small data budget.
- Effective epochs may be far below 1.0 in short runs, which limits generalization.
- Byte-level tokenization is used for simplicity and does not represent large-scale production LM tokenization.
- Architecture-level conclusions are preliminary under local compute constraints.
