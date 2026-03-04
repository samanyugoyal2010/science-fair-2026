"""
Compute training speed comparison: baseline vs hybrid (step_time_ms, toks_per_sec).
Run from repo root: python -m src.analysis.speed_comparison [--runs path/to/runs.csv]
"""
import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs",
        default="results/raw/runs_s33_with_epochs.csv",
        help="CSV with model_type, step_time_ms, toks_per_sec",
    )
    args = parser.parse_args()

    if not os.path.exists(args.runs):
        print(f"File not found: {args.runs}")
        return

    df = pd.read_csv(args.runs)
    if "step_time_ms" not in df.columns or "toks_per_sec" not in df.columns:
        print("CSV must have columns: model_type, step_time_ms, toks_per_sec")
        return

    for model in ["baseline", "hybrid"]:
        sub = df[df.model_type == model]
        if len(sub) == 0:
            continue
        print(f"  {model}: n={len(sub)}  step_time_ms mean={sub.step_time_ms.mean():.2f}  toks_per_sec mean={sub.toks_per_sec.mean():.2f}")

    base = df[df.model_type == "baseline"]
    hy = df[df.model_type == "hybrid"]
    if len(base) == 0 or len(hy) == 0:
        return

    base_time = base.step_time_ms.mean()
    hy_time = hy.step_time_ms.mean()
    base_tps = base.toks_per_sec.mean()
    hy_tps = hy.toks_per_sec.mean()

    # Who is faster (lower step time = faster; higher toks/sec = faster)
    print("\n--- Training speed (your CSV) ---")
    print(f"  Baseline:  {base_time:.2f} ms/step,  {base_tps:.2f} tok/s")
    print(f"  Hybrid:    {hy_time:.2f} ms/step,  {hy_tps:.2f} tok/s")

    # Percent difference (positive = baseline faster in time; negative = hybrid faster)
    pct_time = (hy_time - base_time) / base_time * 100
    pct_tps = (hy_tps - base_tps) / base_tps * 100

    if pct_time > 0:
        print(f"  → Baseline is {pct_time:.1f}% faster per step (lower ms/step).")
    else:
        print(f"  → Hybrid is {-pct_time:.1f}% faster per step.")

    if pct_tps < 0:
        print(f"  → Baseline has {-pct_tps:.1f}% higher throughput (toks/sec).")
    else:
        print(f"  → Hybrid has {pct_tps:.1f}% higher throughput.")

    # If you had a "10.7% faster" number from inference, you'd note it here
    print("\n  (If you measured 10.7% in a different test, e.g. inference/generation, note that separately.)")


if __name__ == "__main__":
    main()
