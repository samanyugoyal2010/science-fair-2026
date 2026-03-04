# Split-Stream Hybrid LM MVP Report

## 1. Objective
Evaluate whether a split-stream hybrid LM reduces peak memory vs. a parameter-matched transformer baseline while keeping validation perplexity within +4%.

## 2. Experimental Controls
- Same tokenizer, data split, optimizer, LR, train steps, seeds.
- Parameter-matched comparison (target <0.5% param gap).

## 3. Main Results
- See `results/raw/runs.csv` and `results/processed/summary.csv`.

## 4. Ablations
- local_only, ssm_only, sum_fusion.
- See `results/raw/ablations.csv` and `results/processed/ablation_summary.csv`.

## 5. Statistical Analysis
- Mean, std, and 95% CI across 3 seeds.

## 6. Claim Decision Rule
- Supported: hybrid memory < baseline memory and val PPL delta <= +4%.
- Partially supported: otherwise.

## 7. Figures
- `results/figures/memory_vs_context.png`
- `results/figures/ppl_vs_context.png`
- `results/figures/efficiency_frontier.png`
- `results/figures/ablation_bars.png`
