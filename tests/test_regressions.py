from pathlib import Path

import pandas as pd
import pytest
import torch

from src.analysis.aggregate_results import main as aggregate_main
from src.data.openwebtext_subset import SequenceDataset
from src.eval.evaluate import evaluate_model
from src.experiments.run_ablation import ensure_csv as ensure_ablation_csv
from src.experiments.run_experiment import RUNS_HEADER, ensure_csv as ensure_runs_csv
from src.models.hybrid_splitstream import HybridConfig, SplitStreamHybridLM
from src.models.transformer_baseline import GPTBaseline, TransformerConfig
from src.train.train import TrainConfig, train_model


def test_train_model_raises_when_loader_has_zero_batches():
    train_ds = SequenceDataset(tokens=[1, 2, 3], block_size=1)  # len == 1
    val_ds = SequenceDataset(tokens=[1, 2, 3, 4, 5], block_size=1)
    model = GPTBaseline(TransformerConfig(d_model=32, n_heads=4, n_layers=1, max_seq_len=8))
    cfg = TrainConfig(batch_size=16, max_steps=1)

    with pytest.raises(ValueError, match="yielded zero batches"):
        train_model(model, train_ds, val_ds, cfg, torch.device("cpu"))


def test_hybrid_max_seq_len_guard_has_clear_error():
    cfg = HybridConfig(d_model=64, n_heads=8, n_local_layers=1, n_ssm_layers=1, max_seq_len=16)
    model = SplitStreamHybridLM(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    with pytest.raises(ValueError, match="max_seq_len"):
        model(x, x)


def test_aggregate_creates_outdir_and_uses_out_root(tmp_path: Path, monkeypatch):
    runs = tmp_path / "raw" / "runs.csv"
    abls = tmp_path / "raw" / "ablations.csv"
    out = tmp_path / "custom" / "summary.csv"
    runs.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"run_id": "r1", "model_type": "baseline", "seed": 1, "context_len": 256, "params_total": 10, "train_steps": 1, "val_loss": 1.0, "val_ppl": 2.0, "peak_mem_mb": 100.0, "step_time_ms": 10.0, "toks_per_sec": 1000.0},
            {"run_id": "r2", "model_type": "hybrid", "seed": 1, "context_len": 256, "params_total": 10, "train_steps": 1, "val_loss": 0.9, "val_ppl": 1.8, "peak_mem_mb": 110.0, "step_time_ms": 12.0, "toks_per_sec": 900.0},
        ]
    ).to_csv(runs, index=False)

    pd.DataFrame(
        [
            {
                "run_id": "a1",
                "ablation_name": "local_only",
                "seed": 1,
                "context_len": 256,
                "val_ppl": 2.2,
                "peak_mem_mb": 90.0,
            }
        ]
    ).to_csv(abls, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "aggregate_results.py",
            "--runs",
            str(runs),
            "--ablations",
            str(abls),
            "--out",
            str(out),
        ],
    )
    aggregate_main()

    assert out.exists()
    assert (out.parent / "ablation_summary.csv").exists()
    assert (out.parent / "paired_deltas.csv").exists()
    assert (out.parent / "paired_summary.csv").exists()


def test_ensure_csv_allows_filename_only_out(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ensure_runs_csv("runs.csv", RUNS_HEADER)
    ensure_ablation_csv("ablations.csv")
    assert (tmp_path / "runs.csv").exists()
    assert (tmp_path / "ablations.csv").exists()


def test_sequence_dataset_too_short_has_zero_length_and_safe_eval():
    ds = SequenceDataset(tokens=[1], block_size=1)
    assert len(ds) == 0
    model = GPTBaseline(TransformerConfig(d_model=32, n_heads=4, n_layers=1, max_seq_len=8))
    val_loss, val_ppl = evaluate_model(model, ds, batch_size=4, device=torch.device("cpu"), max_batches=2)
    assert val_loss == 0.0
    assert val_ppl == 1.0


def test_aggregate_raises_for_empty_runs(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs.csv"
    pd.DataFrame(
        columns=[
            "run_id",
            "model_type",
            "seed",
            "context_len",
            "val_ppl",
            "peak_mem_mb",
            "toks_per_sec",
        ]
    ).to_csv(runs, index=False)
    out = tmp_path / "summary.csv"
    monkeypatch.setattr("sys.argv", ["aggregate_results.py", "--runs", str(runs), "--out", str(out), "--ablations", ""])
    with pytest.raises(ValueError, match="no rows"):
        aggregate_main()


def test_aggregate_raises_for_duplicate_run_id(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs.csv"
    pd.DataFrame(
        [
            {"run_id": "dup", "model_type": "baseline", "seed": 1, "context_len": 256, "val_ppl": 2.0, "peak_mem_mb": 100.0, "toks_per_sec": 1000.0},
            {"run_id": "dup", "model_type": "hybrid", "seed": 1, "context_len": 256, "val_ppl": 1.9, "peak_mem_mb": 110.0, "toks_per_sec": 900.0},
        ]
    ).to_csv(runs, index=False)
    out = tmp_path / "summary.csv"
    monkeypatch.setattr("sys.argv", ["aggregate_results.py", "--runs", str(runs), "--out", str(out), "--ablations", ""])
    with pytest.raises(ValueError, match="Duplicate run_id"):
        aggregate_main()


def test_aggregate_raises_for_unpaired_seed(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs.csv"
    pd.DataFrame(
        [
            {"run_id": "b1", "model_type": "baseline", "seed": 1, "context_len": 256, "val_ppl": 2.0, "peak_mem_mb": 100.0, "toks_per_sec": 1000.0},
            {"run_id": "b2", "model_type": "baseline", "seed": 2, "context_len": 256, "val_ppl": 2.1, "peak_mem_mb": 101.0, "toks_per_sec": 990.0},
            {"run_id": "h1", "model_type": "hybrid", "seed": 1, "context_len": 256, "val_ppl": 1.9, "peak_mem_mb": 110.0, "toks_per_sec": 900.0},
        ]
    ).to_csv(runs, index=False)
    out = tmp_path / "summary.csv"
    monkeypatch.setattr("sys.argv", ["aggregate_results.py", "--runs", str(runs), "--out", str(out), "--ablations", ""])
    with pytest.raises(ValueError, match="Inconsistent paired runs"):
        aggregate_main()
