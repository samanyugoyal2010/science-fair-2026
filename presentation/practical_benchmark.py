import argparse
import csv
import gc
import os
import time
import tracemalloc
from dataclasses import dataclass
from typing import Optional

import torch

from src.data.openwebtext_subset import ByteTokenizer, SequenceDataset
from src.eval.evaluate import evaluate_model
from src.eval.wordle_eval import load_models

DEFAULT_EVAL_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "In the morning the sun rose over the mountains. "
    "The river flows through the valley and into the sea. "
    "They sat by the fire and talked until midnight. "
    "She opened the letter and read it twice before replying. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "A wizard's job is to vex chumps quickly in fog. "
    "The engineer profiled the model and compared latency against quality. "
    "Each experiment used paired seeds and controlled settings for fairness. "
    "The prototype was tested on a temporary server for real deployment conditions."
)


@dataclass
class BenchResult:
    model_type: str
    val_ppl: float
    quality_target: float
    quality_target_met: bool
    generated_tokens: int
    elapsed_sec: float
    toks_per_sec: float
    peak_mem_mb: float
    estimated_cost_per_1m_tokens_usd: float


def select_device(explicit: Optional[str]) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_eval_dataset(text: str, block_size: int) -> SequenceDataset:
    tok = ByteTokenizer()
    # Ensure the default path is robust even if caller provides a very short text.
    min_chars = max(2 * block_size + 32, 512)
    if len(text) < min_chars:
        repeats = (min_chars // max(1, len(text))) + 1
        text = (text.strip() + " ") * repeats
    ids = tok.encode(text)
    return SequenceDataset(ids, block_size)


def get_current_mem_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / (1024 * 1024)
    if device.type == "mps" and hasattr(torch, "mps"):
        try:
            return torch.mps.driver_allocated_memory() / (1024 * 1024)
        except Exception:
            return 0.0
    current, _ = tracemalloc.get_traced_memory()
    return current / (1024 * 1024)


def get_peak_mem_mb(device: torch.device, sampled_peak_mb: float) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    if device.type == "mps":
        return sampled_peak_mb
    _current, peak = tracemalloc.get_traced_memory()
    return peak / (1024 * 1024)


@torch.no_grad()
def greedy_generate_and_profile(
    model,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    sample_every: int,
) -> tuple[int, float, float]:
    tok = ByteTokenizer()
    prompt_ids = tok.encode(prompt)[:-1]
    if not prompt_ids:
        prompt_ids = [ord("A")]

    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    sampled_peak_mb = get_current_mem_mb(device)

    maybe_sync(device)
    start = time.perf_counter()
    for i in range(max_new_tokens):
        logits, _ = model(x, None)
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        nxt = torch.tensor([[next_id]], dtype=torch.long, device=device)
        x = torch.cat([x, nxt], dim=1)

        if sample_every > 0 and ((i + 1) % sample_every == 0 or i + 1 == max_new_tokens):
            sampled_peak_mb = max(sampled_peak_mb, get_current_mem_mb(device))

    maybe_sync(device)
    elapsed = time.perf_counter() - start
    generated = max_new_tokens
    toks_per_sec = generated / max(elapsed, 1e-9)
    peak_mb = get_peak_mem_mb(device, sampled_peak_mb)
    return generated, elapsed, toks_per_sec, peak_mb


def estimate_cost_per_1m_tokens(
    toks_per_sec: float,
    usd_per_hour: float,
    power_watts: float,
    usd_per_kwh: float,
) -> float:
    if toks_per_sec <= 0:
        return float("inf")

    hours_for_1m = 1_000_000.0 / toks_per_sec / 3600.0
    instance_cost = hours_for_1m * max(0.0, usd_per_hour)
    energy_kwh = (max(0.0, power_watts) / 1000.0) * hours_for_1m
    energy_cost = energy_kwh * max(0.0, usd_per_kwh)
    return instance_cost + energy_cost


def run_model_benchmark(
    model,
    model_type: str,
    device: torch.device,
    eval_ds: SequenceDataset,
    quality_target: float,
    eval_batch_size: int,
    eval_max_batches: int,
    prompt: str,
    max_new_tokens: int,
    sample_every: int,
    usd_per_hour: float,
    power_watts: float,
    usd_per_kwh: float,
) -> BenchResult:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    val_loss, val_ppl = evaluate_model(
        model,
        eval_ds,
        batch_size=eval_batch_size,
        device=device,
        max_batches=eval_max_batches,
    )
    _ = val_loss

    generated, elapsed, toks_per_sec, peak_mem_mb = greedy_generate_and_profile(
        model,
        device=device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        sample_every=sample_every,
    )

    cost = estimate_cost_per_1m_tokens(
        toks_per_sec=toks_per_sec,
        usd_per_hour=usd_per_hour,
        power_watts=power_watts,
        usd_per_kwh=usd_per_kwh,
    )

    return BenchResult(
        model_type=model_type,
        val_ppl=float(val_ppl),
        quality_target=quality_target,
        quality_target_met=bool(val_ppl <= quality_target),
        generated_tokens=generated,
        elapsed_sec=float(elapsed),
        toks_per_sec=float(toks_per_sec),
        peak_mem_mb=float(peak_mem_mb),
        estimated_cost_per_1m_tokens_usd=float(cost),
    )


def choose_winner(results: list[BenchResult], quality_target: float) -> str:
    qualified = [r for r in results if r.quality_target_met]
    if not qualified:
        return f"no winner: no model met quality target (val_ppl <= {quality_target})"

    best = sorted(
        qualified,
        key=lambda r: (r.estimated_cost_per_1m_tokens_usd, r.peak_mem_mb, -r.toks_per_sec),
    )[0]
    return (
        f"winner={best.model_type} (meets quality target; lowest estimated_cost_per_1m_tokens_usd among qualified models)"
    )


def write_csv(path: str, rows: list[BenchResult]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_type",
                "val_ppl",
                "quality_target",
                "quality_target_met",
                "generated_tokens",
                "elapsed_sec",
                "toks_per_sec",
                "peak_mem_mb",
                "estimated_cost_per_1m_tokens_usd",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Practical benchmark for science fair presentation: compares baseline vs hybrid on "
            "quality target, inference throughput, peak memory, and estimated cost per 1M tokens."
        )
    )
    parser.add_argument("--config", default="configs/science_fair_s33.yaml")
    parser.add_argument("--baseline-ckpt", default="results/checkpoints/s33/baseline_ctx1024_s33.pt")
    parser.add_argument("--hybrid-ckpt", default="results/checkpoints/s33/hybrid_ctx1024_s33.pt")
    parser.add_argument("--device", default=None, help="cpu|mps|cuda (default: auto)")

    parser.add_argument("--eval-data", default=None, help="Optional text file for perplexity evaluation")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-max-batches", type=int, default=50)
    parser.add_argument("--quality-target", type=float, default=8.0, help="val_ppl target; lower is better")

    parser.add_argument("--prompt", default="In a practical deployment, the assistant should")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sample-every", type=int, default=4, help="Memory sample interval during generation")

    parser.add_argument("--usd-per-hour", type=float, default=0.0, help="Compute rental cost in USD/hour")
    parser.add_argument("--power-watts", type=float, default=60.0, help="Approx average machine power draw in W")
    parser.add_argument("--usd-per-kwh", type=float, default=0.34, help="Electricity price in USD/kWh")

    parser.add_argument("--out-csv", default="presentation/practical_benchmark_results.csv")
    parser.add_argument("--out-summary", default="presentation/practical_benchmark_summary.txt")
    args = parser.parse_args()

    device = select_device(args.device)

    text = DEFAULT_EVAL_TEXT
    if args.eval_data and os.path.exists(args.eval_data):
        with open(args.eval_data, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.eval_data:
        print(f"[presentation] Warning: eval file not found: {args.eval_data}. Using built-in text.")

    eval_ds = build_eval_dataset(text, args.block_size)
    if len(eval_ds) == 0:
        raise ValueError(
            f"Eval dataset empty for block_size={args.block_size}. Use longer eval text or smaller block-size."
        )

    tracemalloc.start()
    baseline, hybrid = load_models(args.config, args.baseline_ckpt, args.hybrid_ckpt, device)
    baseline.eval()
    hybrid.eval()

    rows = []
    rows.append(
        run_model_benchmark(
            baseline,
            model_type="baseline",
            device=device,
            eval_ds=eval_ds,
            quality_target=args.quality_target,
            eval_batch_size=args.eval_batch_size,
            eval_max_batches=args.eval_max_batches,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            sample_every=args.sample_every,
            usd_per_hour=args.usd_per_hour,
            power_watts=args.power_watts,
            usd_per_kwh=args.usd_per_kwh,
        )
    )
    rows.append(
        run_model_benchmark(
            hybrid,
            model_type="hybrid",
            device=device,
            eval_ds=eval_ds,
            quality_target=args.quality_target,
            eval_batch_size=args.eval_batch_size,
            eval_max_batches=args.eval_max_batches,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            sample_every=args.sample_every,
            usd_per_hour=args.usd_per_hour,
            power_watts=args.power_watts,
            usd_per_kwh=args.usd_per_kwh,
        )
    )

    decision = choose_winner(rows, args.quality_target)
    write_csv(args.out_csv, rows)

    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)
    with open(args.out_summary, "w", encoding="utf-8") as f:
        f.write(f"device={device.type}\n")
        f.write(f"quality_target_val_ppl<={args.quality_target}\n")
        f.write(f"decision={decision}\n")

    print(f"[presentation] device={device.type}")
    for r in rows:
        print(
            "[presentation] "
            f"model={r.model_type} "
            f"val_ppl={r.val_ppl:.4f} target_met={r.quality_target_met} "
            f"toks_per_sec={r.toks_per_sec:.2f} peak_mem_mb={r.peak_mem_mb:.2f} "
            f"cost_per_1m_tokens_usd={r.estimated_cost_per_1m_tokens_usd:.4f}"
        )
    print(f"[presentation] {decision}")
    print(f"[presentation] wrote csv: {args.out_csv}")
    print(f"[presentation] wrote summary: {args.out_summary}")


if __name__ == "__main__":
    main()
