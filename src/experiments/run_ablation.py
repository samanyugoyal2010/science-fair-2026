import argparse
import csv
import os

import torch
import yaml

from src.data.openwebtext_subset import DataConfig, build_datasets
from src.models.hybrid_splitstream import HybridConfig, SplitStreamHybridLM
from src.train.train import TrainConfig, set_seed, train_model


ABL_HEADER = ["run_id", "ablation_name", "seed", "context_len", "val_ppl", "peak_mem_mb"]


def ensure_csv(path: str):
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(ABL_HEADER)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--out", default="results/raw/ablations.csv")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tcfg = TrainConfig(**cfg["train"])
    ensure_csv(args.out)

    ablations = {
        "local_only": {"disable_ssm_stream": True, "disable_local_stream": False, "fusion_mode": "sum"},
        "ssm_only": {"disable_ssm_stream": False, "disable_local_stream": True, "fusion_mode": "sum"},
        "sum_fusion": {"disable_ssm_stream": False, "disable_local_stream": False, "fusion_mode": "sum"},
    }

    for context_len in cfg["contexts"]:
        dcfg = DataConfig(**{**cfg["data"], "block_size": context_len})
        train_ds, val_ds, _ = build_datasets(dcfg)
        for seed in cfg["seeds"]:
            for name, overrides in ablations.items():
                set_seed(seed)
                hcfg = HybridConfig(**{**cfg["hybrid"], **overrides, "max_seq_len": context_len})
                model = SplitStreamHybridLM(hcfg)
                out = train_model(model, train_ds, val_ds, tcfg, device)
                row = {
                    "run_id": f"{name}_ctx{context_len}_s{seed}",
                    "ablation_name": name,
                    "seed": seed,
                    "context_len": context_len,
                    "val_ppl": out["val_ppl"],
                    "peak_mem_mb": out["peak_mem_mb"],
                }
                with open(args.out, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=ABL_HEADER).writerow(row)


if __name__ == "__main__":
    main()
