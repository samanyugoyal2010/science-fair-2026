import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.eval.evaluate import evaluate_model
from src.profiler.memory_profiler import get_current_memory_mb, get_peak_memory_mb, timed_step


@dataclass
class TrainConfig:
    batch_size: int = 16
    max_steps: int = 1000
    eval_interval: int = 200
    eval_batches: int = 30
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_workers: int = 0


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_ds, val_ds, cfg: TrainConfig, device: torch.device) -> dict:
    model = model.to(device)
    model.train()
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
    if len(loader) == 0:
        raise ValueError(
            "Training DataLoader yielded zero batches. "
            "Increase dataset size or reduce batch_size when drop_last=True."
        )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    step = 0
    running_loss = 0.0
    step_times = []
    if device.type == "cpu":
        import tracemalloc

        tracemalloc.start()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sampled_peak_mb = get_current_memory_mb(device)

    while step < cfg.max_steps:
        for x, y in loader:
            if step >= cfg.max_steps:
                break
            x = x.to(device)
            y = y.to(device)

            def _step():
                opt.zero_grad(set_to_none=True)
                _, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                return loss

            loss, elapsed_ms = timed_step(_step)
            step_times.append(elapsed_ms)
            running_loss += loss.item()
            step += 1
            sampled_peak_mb = max(sampled_peak_mb, get_current_memory_mb(device))
            if step == 1 or step % max(1, cfg.eval_interval) == 0 or step == cfg.max_steps:
                steps_per_epoch = max(1, len(loader))
                epoch_progress = step / steps_per_epoch
                print(
                    f"[train] step={step}/{cfg.max_steps} "
                    f"batch_size={cfg.batch_size} "
                    f"epoch~={epoch_progress:.4f} "
                    f"loss={loss.item():.4f}"
                )

    val_loss, val_ppl = evaluate_model(model, val_ds, cfg.batch_size, device, cfg.eval_batches)
    peak_mem_mb = get_peak_memory_mb(device, sampled_peak_mb=sampled_peak_mb)
    avg_step_ms = float(sum(step_times) / max(1, len(step_times)))
    toks_per_sec = float((cfg.batch_size * train_ds.block_size) / (avg_step_ms / 1000.0)) if avg_step_ms > 0 else 0.0
    steps_per_epoch = max(1, len(loader))
    effective_epochs = cfg.max_steps / steps_per_epoch

    return {
        "train_loss": float(running_loss / max(1, cfg.max_steps)),
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "peak_mem_mb": peak_mem_mb,
        "step_time_ms": avg_step_ms,
        "toks_per_sec": toks_per_sec,
        "train_steps": cfg.max_steps,
        "batch_size": cfg.batch_size,
        "steps_per_epoch": steps_per_epoch,
        "effective_epochs": effective_epochs,
    }
