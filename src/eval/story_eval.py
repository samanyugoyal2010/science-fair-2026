import argparse
from typing import Optional

import torch
import yaml

from src.data.openwebtext_subset import ByteTokenizer
from src.eval.wordle_eval import load_models


def _select_device(explicit: Optional[str]) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def generate_story(
    model,
    tokenizer: ByteTokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    model.eval()
    # ByteTokenizer.encode appends EOS; drop it for the conditioning prompt.
    ids = tokenizer.encode(prompt)[:-1]
    eos_id = tokenizer.eos_token

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = model(x)
            logits = logits[0, -1] / max(1e-5, temperature)

            if top_k > 0:
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                thresh = v[-1]
                logits = torch.where(logits >= thresh, logits, torch.full_like(logits, float("-inf")))

            probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            ids.append(next_id)
            if next_id == eos_id:
                break

    # Strip everything after the first EOS for decoding.
    if eos_id in ids:
        cutoff = ids.index(eos_id)
        ids = ids[:cutoff]

    text = bytes(ids).decode("utf-8", errors="ignore")
    return text


def main():
    parser = argparse.ArgumentParser()
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
        "--device",
        default=None,
        help="Optional explicit device string (e.g. cpu, cuda, mps). Defaults to MPS, then CUDA, then CPU.",
    )
    parser.add_argument(
        "--prompt",
        default="Once upon a time, in a small village by the sea,",
        help="Text prompt to condition the story on.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    device = _select_device(args.device)
    torch.manual_seed(args.seed)

    baseline, hybrid = load_models(args.config, args.baseline_ckpt, args.hybrid_ckpt, device)
    tok = ByteTokenizer()

    print(f"[story] device={device} prompt={repr(args.prompt)}")

    print("\n[story] === Baseline (GPT) ===")
    baseline_story = generate_story(
        baseline,
        tok,
        device,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(baseline_story)

    print("\n[story] === Hybrid (Split-Stream) ===")
    hybrid_story = generate_story(
        hybrid,
        tok,
        device,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(hybrid_story)


if __name__ == "__main__":
    main()

