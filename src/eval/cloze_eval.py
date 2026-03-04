"""
Cloze-style completion eval: score both models on the same (prefix, continuation) pairs.
Reports average log-probability of the correct continuation so you can compare fairly.
"""
import argparse
from typing import Optional

import torch
import yaml

from src.data.openwebtext_subset import ByteTokenizer
from src.eval.wordle_eval import load_models

# Pairs (prefix, correct_next_phrase). Model is scored on P(correct | prefix).
DEFAULT_CLOZE_PAIRS = [
    ("The cat sat on the ", "mat."),
    ("In the morning the sun ", "rose."),
    ("She went to the store to buy ", "bread."),
    ("The river flows through the ", "valley."),
    ("He opened the door and ", "walked in."),
    ("They sat by the fire and ", "talked."),
    ("The quick brown fox ", "jumps over the lazy dog."),
    ("Pack my box with five dozen ", "liquor jugs."),
    ("Once upon a time there was a ", "king."),
    ("The old house stood at the end of the ", "road."),
]


def _select_device(explicit: Optional[str]) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def score_completion(model, device: torch.device, tokenizer: ByteTokenizer, prefix: str, continuation: str) -> float:
    """Average log-probability of the continuation tokens given the prefix."""
    full = prefix + continuation
    prefix_ids = tokenizer.encode(prefix)[:-1]  # no EOS in prefix
    full_ids = tokenizer.encode(full)
    # Continuation starts after prefix; we want log P(cont_tokens | prefix)
    start = len(prefix_ids)
    end = len(full_ids) - 1  # last is EOS
    if start >= end:
        return 0.0

    x = torch.tensor(full_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(full_ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    logits, _ = model(x, y)
    logp = torch.log_softmax(logits, dim=-1)

    total = 0.0
    for t in range(start, end):
        tok = y[0, t].item()
        total += float(logp[0, t, tok].item())
    n = max(1, end - start)
    return total / n


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline and hybrid on cloze (prefix → continuation) log-probability."
    )
    parser.add_argument("--config", default="configs/science_fair_s33.yaml")
    parser.add_argument(
        "--baseline-ckpt",
        default="results/checkpoints/s33/baseline_ctx1024_s33.pt",
        help="Path to baseline checkpoint.",
    )
    parser.add_argument(
        "--hybrid-ckpt",
        default="results/checkpoints/s33/hybrid_ctx1024_s33.pt",
        help="Path to hybrid checkpoint.",
    )
    parser.add_argument(
        "--pairs-file",
        default=None,
        help="Optional path to TSV: prefix<TAB>continuation per line. Else uses built-in pairs.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device (e.g. cpu, mps, cuda). Default: auto.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional CSV path to write model_type,avg_logp.",
    )
    args = parser.parse_args()

    device = _select_device(args.device)
    tok = ByteTokenizer()

    if args.pairs_file:
        pairs = []
        with open(args.pairs_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
        if not pairs:
            raise ValueError(f"No valid prefix<TAB>continuation lines in {args.pairs_file}")
    else:
        pairs = DEFAULT_CLOZE_PAIRS

    baseline, hybrid = load_models(args.config, args.baseline_ckpt, args.hybrid_ckpt, device)

    bl_scores = [score_completion(baseline, device, tok, p, c) for p, c in pairs]
    hy_scores = [score_completion(hybrid, device, tok, p, c) for p, c in pairs]

    bl_avg = sum(bl_scores) / len(bl_scores)
    hy_avg = sum(hy_scores) / len(hy_scores)

    print(f"[cloze] device={device} num_pairs={len(pairs)}")
    print("\n[cloze] === Average log-prob of correct continuation (higher = better) ===")
    print(f"  baseline  avg_logp = {bl_avg:.4f}")
    print(f"  hybrid    avg_logp = {hy_avg:.4f}")
    diff = hy_avg - bl_avg
    print(f"  diff (hybrid - baseline) = {diff:.4f}  ({'hybrid better' if diff > 0 else 'baseline better'})")

    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            f.write("model_type,avg_logp\n")
            f.write(f"baseline,{bl_avg:.6f}\n")
            f.write(f"hybrid,{hy_avg:.6f}\n")
        print(f"[cloze] Wrote {args.out}")


if __name__ == "__main__":
    main()
