import argparse
from typing import Optional

import torch

from src.eval.wordle_eval import load_models, load_word_list, play_one


def _select_device(explicit: Optional[str]) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Play one Wordle-style game on a specific target word with baseline and hybrid models."
    )
    parser.add_argument("--config", default="configs/science_fair_s33.yaml")
    parser.add_argument(
        "--baseline-ckpt",
        default="results/checkpoints/s33/baseline_ctx1024_s33.pt",
        help="Path to baseline checkpoint (default: s33 ctx1024).",
    )
    parser.add_argument(
        "--hybrid-ckpt",
        default="results/checkpoints/s33/hybrid_ctx1024_s33.pt",
        help="Path to hybrid checkpoint (default: s33 ctx1024).",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="The 5-letter target word to test (e.g. today's Wordle word).",
    )
    parser.add_argument(
        "--word-list",
        default=None,
        help="Optional newline-separated 5-letter word file; if omitted, uses built-in list.",
    )
    parser.add_argument(
        "--max-guesses",
        type=int,
        default=6,
        help="Maximum number of guesses per game (default 6, like Wordle).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device (e.g. cpu, mps, cuda). Default: auto-detect.",
    )
    args = parser.parse_args()

    device = _select_device(args.device)
    target = args.target.strip().lower()
    if len(target) != 5 or not target.isalpha():
        raise ValueError("Target word must be exactly 5 alphabetic letters.")

    words = load_word_list(args.word_list)
    if target not in words:
        print(f"[wordle-single] Warning: target '{target}' not in word list (len={len(words)}).")

    baseline, hybrid = load_models(args.config, args.baseline_ckpt, args.hybrid_ckpt, device)

    print(f"[wordle-single] device={device} target={target} max_guesses={args.max_guesses}")

    # Baseline game
    n_b, ok_b, hist_b = play_one(baseline, device, target, words, max_guesses=args.max_guesses)
    print("\n[wordle-single] === Baseline (GPT) ===")
    print(f"solved={ok_b} guesses={n_b}")
    for i, (g, f) in enumerate(hist_b, start=1):
        print(f"  {i}: guess={g} feedback={f}")

    # Hybrid game
    n_h, ok_h, hist_h = play_one(hybrid, device, target, words, max_guesses=args.max_guesses)
    print("\n[wordle-single] === Hybrid (Split-Stream) ===")
    print(f"solved={ok_h} guesses={n_h}")
    for i, (g, f) in enumerate(hist_h, start=1):
        print(f"  {i}: guess={g} feedback={f}")


if __name__ == "__main__":
    main()

