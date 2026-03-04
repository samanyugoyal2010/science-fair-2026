"""
Held-out perplexity comparison: evaluate baseline and hybrid on the same sequences
and report loss/perplexity so you can compare them fairly.
"""
import argparse
import os
from typing import Optional

import torch
import yaml

from src.data.openwebtext_subset import ByteTokenizer, SequenceDataset
from src.eval.evaluate import evaluate_model
from src.eval.wordle_eval import load_models

# Small built-in corpus so the script runs without an external file
DEFAULT_EVAL_TEXT = """
The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.
How vexingly quick daft zebras jump. Sphinx of black quartz judge my vow.
The five boxing wizards jump quickly. Two driven jocks help fax my big quiz.
Fake swords put back into the box. The wizard quickly jinxed the gnomes before they vaporized.
She had a habit of taking long walks in the evening. The old house stood at the end of the road.
He opened the letter and read it twice. In the morning the sun rose over the mountains.
The river flows through the valley and into the sea. They sat by the fire and talked until midnight.
"""


def _select_device(explicit: Optional[str]) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_eval_dataset(text: str, block_size: int) -> SequenceDataset:
    tok = ByteTokenizer()
    # Encode strips to one EOS at end; for a single doc we get one long id list
    ids = tok.encode(text)
    return SequenceDataset(ids, block_size)


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline and hybrid on held-out perplexity (same data, same settings)."
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
        "--eval-data",
        default=None,
        help="Path to text file for eval. If unset, uses a small built-in passage.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Context length for eval (default 256 for memory; max is model max_seq_len).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=50,
        help="Max batches per model (cap eval time).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device (e.g. cpu, mps, cuda). Default: auto.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional CSV path to write model_type,val_loss,val_ppl.",
    )
    args = parser.parse_args()

    device = _select_device(args.device)

    if args.eval_data and os.path.exists(args.eval_data):
        with open(args.eval_data, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = DEFAULT_EVAL_TEXT.strip()
        if args.eval_data:
            print(f"[heldout_ppl] Warning: --eval-data not found, using built-in text.")

    eval_ds = build_eval_dataset(text, args.block_size)
    if len(eval_ds) == 0:
        raise ValueError(
            f"Eval dataset is empty (text too short for block_size={args.block_size}). "
            "Use a longer file or smaller --block-size."
        )

    baseline, hybrid = load_models(args.config, args.baseline_ckpt, args.hybrid_ckpt, device)
    n_batches = min(args.max_batches, max(1, (len(eval_ds) + args.batch_size - 1) // args.batch_size))

    print(f"[heldout_ppl] device={device} block_size={args.block_size} batches={n_batches} eval_samples={len(eval_ds)}")

    bl_loss, bl_ppl = evaluate_model(
        baseline, eval_ds, args.batch_size, device, max_batches=args.max_batches
    )
    hy_loss, hy_ppl = evaluate_model(
        hybrid, eval_ds, args.batch_size, device, max_batches=args.max_batches
    )

    print("\n[heldout_ppl] === Results (same data, same settings) ===")
    print(f"  baseline  loss={bl_loss:.4f}  ppl={bl_ppl:.4f}")
    print(f"  hybrid    loss={hy_loss:.4f}  ppl={hy_ppl:.4f}")
    if bl_ppl > 0:
        ratio = hy_ppl / bl_ppl
        print(f"  ratio (hybrid/baseline) ppl = {ratio:.4f}  ({'hybrid better' if ratio < 1 else 'baseline better'})")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            f.write("model_type,val_loss,val_ppl\n")
            f.write(f"baseline,{bl_loss:.6f},{bl_ppl:.6f}\n")
            f.write(f"hybrid,{hy_loss:.6f},{hy_ppl:.6f}\n")
        print(f"[heldout_ppl] Wrote {args.out}")


if __name__ == "__main__":
    main()
