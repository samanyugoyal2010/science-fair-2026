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
        default="results/raw/runs_submission.csv",
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

    # Consumer-hardware framing and scientific caveats
    print("\n--- Context for claims ---")
    if "device" in df.columns:
        devices = sorted(df["device"].dropna().astype(str).unique().tolist())
        print(f"  Device(s) used: {', '.join(devices)}")
    else:
        print("  Device(s) used: not recorded in this CSV")

    if "effective_epochs" in df.columns:
        mean_eff = float(df["effective_epochs"].mean())
        print(f"  Mean effective epochs: {mean_eff:.6f}")
        if mean_eff < 0.01:
            print("  Note: this is pilot-scale training on consumer hardware (results are preliminary).")

    if "params_total" in df.columns and "context_len" in df.columns:
        print("  Parameter gap by context (hybrid vs baseline):")
        for ctx in sorted(df["context_len"].dropna().unique().tolist()):
            b = df[(df.model_type == "baseline") & (df.context_len == ctx)]
            h = df[(df.model_type == "hybrid") & (df.context_len == ctx)]
            if len(b) == 0 or len(h) == 0:
                continue
            b_params = float(b.iloc[0]["params_total"])
            h_params = float(h.iloc[0]["params_total"])
            gap_pct = ((h_params - b_params) / b_params) * 100.0
            print(f"    ctx={int(ctx)}: {gap_pct:+.2f}%")
        print("  Note: report this gap whenever claiming fairness/matching.")


if __name__ == "__main__":
    main()
