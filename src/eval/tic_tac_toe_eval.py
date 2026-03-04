"""
Best-of-3 tic-tac-toe: baseline (transformer) vs hybrid (split-stream).
Each model chooses moves by scoring legal next-token log-probs for digits 1-9.
Hybrid always plays X (goes first); baseline plays O in all three games.
"""
import argparse
from typing import Optional

import torch

from src.models.hybrid_splitstream import HybridConfig, SplitStreamHybridLM
from src.models.transformer_baseline import GPTBaseline, TransformerConfig

# Board: index 0-8 = positions 1-9. Values: '' (empty), 'X', 'O'
WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6),             # diagonals
]


def format_board(board: list[str]) -> str:
    """3x3 grid; each cell is one of 'X', 'O', '.'."""
    chars = [board[i] if board[i] else "." for i in range(9)]
    return f"{chars[0]} {chars[1]} {chars[2]}\n{chars[3]} {chars[4]} {chars[5]}\n{chars[6]} {chars[7]} {chars[8]}"


def build_prompt(board: list[str], side: str) -> str:
    """Side is 'X' or 'O'. Board state with labels."""
    grid = format_board(board)
    return f"Tic-tac-toe. You play {side}.\nBoard:\n{grid}\nYour move (1-9): "


def check_winner(board: list[str]) -> Optional[str]:
    """Returns 'X', 'O', or None. No draw detection here."""
    for a, b, c in WIN_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_draw(board: list[str]) -> bool:
    return all(board[i] for i in range(9))


@torch.no_grad()
def choose_move(model, device: torch.device, board: list[str], side: str) -> int:
    """Return 1-indexed cell (1-9) with highest log-prob among empty cells."""
    prompt = build_prompt(board, side)
    ctx_ids = list(prompt.encode("utf-8", errors="ignore"))
    if not ctx_ids:
        # fallback: first empty cell
        for i in range(9):
            if not board[i]:
                return i + 1
    x = torch.tensor(ctx_ids, dtype=torch.long, device=device).unsqueeze(0)
    logits, _ = model(x)
    logp = torch.log_softmax(logits[0, -1], dim=-1)
    # Digits '1'..'9' are bytes 49..57
    best_cell = None
    best_score = -1e30
    for i in range(9):
        if board[i]:
            continue
        tok_id = 49 + i
        if tok_id < logp.size(0):
            sc = float(logp[tok_id].item())
            if sc > best_score:
                best_score = sc
                best_cell = i + 1
    if best_cell is None:
        for i in range(9):
            if not board[i]:
                return i + 1
    return best_cell or 1


def play_one_game(
    baseline_model,
    hybrid_model,
    device: torch.device,
    baseline_plays: str,
    verbose: bool = True,
) -> Optional[str]:
    """
    Play one full game. baseline_plays is 'X' or 'O' (baseline's side).
    Returns winner: 'X', 'O', or None for draw.
    """
    board = [""] * 9
    hybrid_plays = "O" if baseline_plays == "X" else "X"
    turn = "X"
    move_count = 0
    while move_count < 9:
        winner = check_winner(board)
        if winner:
            if verbose:
                print(f"  Winner: {winner}")
            return winner
        if is_draw(board):
            if verbose:
                print("  Draw.")
            return None

        if turn == baseline_plays:
            cell = choose_move(baseline_model, device, board, turn)
            name = "baseline"
        else:
            cell = choose_move(hybrid_model, device, board, turn)
            name = "hybrid"

        idx = cell - 1
        if idx < 0 or idx > 8 or board[idx]:
            # Illegal: pick first empty
            for i in range(9):
                if not board[i]:
                    idx = i
                    cell = i + 1
                    break
        board[idx] = turn
        if verbose:
            print(f"  {name} ({turn}) -> {cell}  |  {format_board(board).replace(chr(10), ' | ')}")
        turn = "O" if turn == "X" else "X"
        move_count += 1

    if is_draw(board):
        if verbose:
            print("  Draw.")
        return None
    w = check_winner(board)
    if verbose and w:
        print(f"  Winner: {w}")
    return w


def load_models(
    cfg_path: str,
    baseline_ckpt: Optional[str],
    hybrid_ckpt: Optional[str],
    device: torch.device,
):
    import yaml
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ctx = cfg["contexts"][0]

    b = torch.load(baseline_ckpt, map_location=device) if baseline_ckpt else None
    h = torch.load(hybrid_ckpt, map_location=device) if hybrid_ckpt else None

    if b and "config" in b:
        bcfg = TransformerConfig(**b["config"])
    else:
        bcfg = TransformerConfig(**{**cfg["transformer"], "max_seq_len": ctx})

    if h and "config" in h:
        hcfg = HybridConfig(**h["config"])
    else:
        hcfg = HybridConfig(**{**cfg["hybrid"], "max_seq_len": ctx})

    baseline = GPTBaseline(bcfg).to(device).eval()
    hybrid = SplitStreamHybridLM(hcfg).to(device).eval()
    if b:
        baseline.load_state_dict(b["model_state"])
    if h:
        hybrid.load_state_dict(h["model_state"])
    return baseline, hybrid


def main():
    parser = argparse.ArgumentParser(description="Best-of-3 tic-tac-toe: baseline vs hybrid")
    parser.add_argument("--config", default="configs/science_fair_s33.yaml")
    parser.add_argument(
        "--baseline-ckpt",
        default="results/checkpoints/s33/baseline_ctx1024_s33.pt",
    )
    parser.add_argument(
        "--hybrid-ckpt",
        default="results/checkpoints/s33/hybrid_ctx1024_s33.pt",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet", action="store_true", help="Less per-move output")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    baseline, hybrid = load_models(args.config, args.baseline_ckpt, args.hybrid_ckpt, device)
    verbose = not args.quiet

    baseline_wins = 0
    hybrid_wins = 0
    draws = 0

    # Hybrid always goes first (plays X); baseline plays O in all three games
    baseline_side = "O"
    for game in range(1, 4):
        if verbose:
            print(f"\n--- Game {game} (baseline=O, hybrid=X; hybrid goes first) ---")
        winner = play_one_game(
            baseline, hybrid, device,
            baseline_plays=baseline_side,
            verbose=verbose,
        )
        if winner == "X":
            if baseline_side == "X":
                baseline_wins += 1
            else:
                hybrid_wins += 1
        elif winner == "O":
            if baseline_side == "O":
                baseline_wins += 1
            else:
                hybrid_wins += 1
        else:
            draws += 1

    print("\n=== Best of 3 result ===")
    print(f"  Baseline (transformer): {baseline_wins} wins")
    print(f"  Hybrid (split-stream):  {hybrid_wins} wins")
    print(f"  Draws:                 {draws}")
    if baseline_wins > hybrid_wins:
        print("  Overall: Baseline wins.")
    elif hybrid_wins > baseline_wins:
        print("  Overall: Hybrid wins.")
    else:
        print("  Overall: Tie.")


if __name__ == "__main__":
    main()
