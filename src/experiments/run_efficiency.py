"""
Efficiency frontier experiment: find the smallest hybrid that matches baseline PPL.
Trains hybrid models at several d_model sizes and reports params vs PPL.
"""
import argparse
import csv
import os

import torch
import yaml

from src.data.openwebtext_subset import DataConfig, build_datasets
from src.models.hybrid_splitstream import HybridConfig, SplitStreamHybridLM
from src.train.train import TrainConfig, count_params, set_seed, train_model


SIZES = [64, 96, 128, 160, 192]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--out", default="results/raw/efficiency_sweep.csv")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.device == "auto":
        resolved = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        resolved = args.device
    device = torch.device(resolved)

    train_cfg = TrainConfig(**cfg["train"])
    data_cfg = DataConfig(**{**cfg["data"], "block_size": args.context})
    train_ds, val_ds, _ = build_datasets(data_cfg)

    header = ["d_model", "n_heads", "params_total", "val_loss", "val_ppl", "peak_mem_mb", "step_time_ms"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        csv.writer(f).writerow(header)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for d in SIZES:
        n_heads = max(1, d // 16)
        if d % n_heads != 0:
            n_heads = max(1, d // 32)
        hcfg = HybridConfig(
            **{
                **cfg["hybrid"],
                "d_model": d,
                "n_heads": n_heads,
                "max_seq_len": args.context,
            }
        )
        set_seed(args.seed)
        model = SplitStreamHybridLM(hcfg)
        params = count_params(model)
        print(f"[efficiency] d_model={d} n_heads={n_heads} params={params:,}")

        loss_log = os.path.join(args.save_dir, f"loss_hybrid_d{d}.csv") if args.save_dir else None
        res = train_model(model, train_ds, val_ds, train_cfg, device, loss_log_path=loss_log)
        print(f"[efficiency] d_model={d} val_ppl={res['val_ppl']:.4f} peak_mem_mb={res['peak_mem_mb']:.2f}")

        with open(args.out, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=header).writerow({
                "d_model": d,
                "n_heads": n_heads,
                "params_total": params,
                "val_loss": res["val_loss"],
                "val_ppl": res["val_ppl"],
                "peak_mem_mb": res["peak_mem_mb"],
                "step_time_ms": res["step_time_ms"],
            })

    print(f"\n[efficiency] Results written to {args.out}")
    print("[efficiency] Compare hybrid PPL at each size against baseline PPL ~13.1 (5M params)")


if __name__ == "__main__":
    main()
