# Split-Stream Hybrid LM MVP

## Setup
```bash
pip install -r requirements.txt
```

## Run core experiments
```bash
python -m src.experiments.run_experiment --config configs/experiment.yaml --out results/raw/runs.csv
```

## Run ablations
```bash
python -m src.experiments.run_ablation --config configs/experiment.yaml --out results/raw/ablations.csv
```

## Aggregate and plot
```bash
python -m src.analysis.aggregate_results --runs results/raw/runs.csv --ablations results/raw/ablations.csv --out results/processed/summary.csv
python -m src.analysis.plot_results --runs results/raw/runs.csv --ablations results/raw/ablations.csv --outdir results/figures
```
# science-fair-2026
