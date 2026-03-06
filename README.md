# Split-Stream Hybrid LM MVP

## Setup
```bash
pip install -r requirements.txt
```

## Run core experiments (canonical single-source output)
```bash
python -m src.experiments.run_experiment \
  --config configs/experiment.yaml \
  --out results/raw/runs_submission.csv \
  --run-group-id isef_submission_a \
  --device auto
```

## Run ablations
```bash
python -m src.experiments.run_ablation --config configs/experiment.yaml --out results/raw/ablations.csv
```

## Aggregate and plot
```bash
python -m src.analysis.aggregate_results \
  --runs results/raw/runs_submission.csv \
  --ablations results/raw/ablations.csv \
  --out results/processed/summary.csv
python -m src.analysis.plot_results \
  --runs results/raw/runs_submission.csv \
  --ablations results/raw/ablations.csv \
  --outdir results/figures
```

## Scientific reporting policy
- Primary metrics: validation perplexity, peak memory, throughput.
- Wordle, tic-tac-toe, and story generation are exploratory demos only.
- If replication count is <3, treat results as pilot/underpowered and non-conclusive.
# science-fair-2026
