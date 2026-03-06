import argparse
import csv
import os
from datetime import datetime
from dataclasses import asdict

import torch
import yaml

from src.data.openwebtext_subset import DataConfig, build_datasets
from src.models.hybrid_splitstream import HybridConfig, SplitStreamHybridLM
from src.models.transformer_baseline import GPTBaseline, TransformerConfig
from src.train.train import TrainConfig, count_params, set_seed, train_model


RUNS_HEADER = [
    "run_group_id",
    "run_id",
    "model_type",
    "seed",
    "device",
    "context_len",
    "params_total",
    "train_steps",
    "val_loss",
    "val_ppl",
    "peak_mem_mb",
    "step_time_ms",
    "toks_per_sec",
    "batch_size",
    "steps_per_epoch",
    "effective_epochs",
]


def ensure_csv(path: str, header: list[str]):
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)


def append_row(path: str, row: dict):
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUNS_HEADER, extrasaction="ignore")
        writer.writerow(row)


def match_hybrid_params(base_params: int, cfg: HybridConfig) -> HybridConfig:
    best = cfg
    best_gap = float("inf")
    for d_model in range(max(64, cfg.d_model - 96), cfg.d_model + 97, 8):
        trial = HybridConfig(**{**asdict(cfg), "d_model": d_model})
        if d_model % trial.n_heads != 0:
            continue
        p = count_params(SplitStreamHybridLM(trial))
        gap = abs(p - base_params) / base_params
        if gap < best_gap:
            best, best_gap = trial, gap
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--out", default="results/raw/runs.csv")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--run-group-id", default=None, help="Optional immutable batch ID for this run set.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        resolved_device = args.device
    device = torch.device(resolved_device)
    run_group_id = args.run_group_id or datetime.now().strftime("rg_%Y%m%d_%H%M%S")
    train_cfg = TrainConfig(**cfg["train"])
    ensure_csv(args.out, RUNS_HEADER)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for context_len in cfg["contexts"]:
        data_cfg = DataConfig(**{**cfg["data"], "block_size": context_len})
        train_ds, val_ds, _meta = build_datasets(data_cfg)

        base_cfg = TransformerConfig(**{**cfg["transformer"], "max_seq_len": context_len})
        base_model = GPTBaseline(base_cfg)
        base_params = count_params(base_model)

        hy_cfg_seed = HybridConfig(**{**cfg["hybrid"], "max_seq_len": context_len})
        hy_cfg = match_hybrid_params(base_params, hy_cfg_seed)

        for seed in cfg["seeds"]:
            print(f"[run] model=baseline seed={seed} context={context_len} starting")
            set_seed(seed)
            model = GPTBaseline(base_cfg)
            bl_loss_log = os.path.join(args.save_dir, f"loss_baseline_ctx{context_len}_s{seed}.csv") if args.save_dir else None
            res = train_model(model, train_ds, val_ds, train_cfg, device, loss_log_path=bl_loss_log)
            print(
                f"[run] model=baseline seed={seed} context={context_len} "
                f"done val_ppl={res['val_ppl']:.4f} peak_mem_mb={res['peak_mem_mb']:.2f} "
                f"batch_size={res['batch_size']} effective_epochs={res['effective_epochs']:.4f}"
            )
            append_row(
                args.out,
                {
                    "run_group_id": run_group_id,
                    "run_id": f"baseline_ctx{context_len}_s{seed}",
                    "model_type": "baseline",
                    "seed": seed,
                    "device": device.type,
                    "context_len": context_len,
                    "params_total": count_params(model),
                    **res,
                },
            )
            if args.save_dir:
                torch.save(
                    {
                        "model_type": "baseline",
                        "seed": seed,
                        "context_len": context_len,
                        "config": vars(base_cfg),
                        "model_state": model.state_dict(),
                    },
                    os.path.join(args.save_dir, f"baseline_ctx{context_len}_s{seed}.pt"),
                )

            print(f"[run] model=hybrid seed={seed} context={context_len} starting")
            set_seed(seed)
            hmodel = SplitStreamHybridLM(hy_cfg)
            hy_loss_log = os.path.join(args.save_dir, f"loss_hybrid_ctx{context_len}_s{seed}.csv") if args.save_dir else None
            hres = train_model(hmodel, train_ds, val_ds, train_cfg, device, loss_log_path=hy_loss_log)
            print(
                f"[run] model=hybrid seed={seed} context={context_len} "
                f"done val_ppl={hres['val_ppl']:.4f} peak_mem_mb={hres['peak_mem_mb']:.2f} "
                f"batch_size={hres['batch_size']} effective_epochs={hres['effective_epochs']:.4f}"
            )
            append_row(
                args.out,
                {
                    "run_group_id": run_group_id,
                    "run_id": f"hybrid_ctx{context_len}_s{seed}",
                    "model_type": "hybrid",
                    "seed": seed,
                    "device": device.type,
                    "context_len": context_len,
                    "params_total": count_params(hmodel),
                    **hres,
                },
            )
            if args.save_dir:
                torch.save(
                    {
                        "model_type": "hybrid",
                        "seed": seed,
                        "context_len": context_len,
                        "config": asdict(hy_cfg),
                        "model_state": hmodel.state_dict(),
                    },
                    os.path.join(args.save_dir, f"hybrid_ctx{context_len}_s{seed}.pt"),
                )


if __name__ == "__main__":
    main()
