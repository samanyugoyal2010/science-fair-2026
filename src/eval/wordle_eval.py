import argparse
import random
from typing import Optional
import os

import torch
import yaml

from src.models.hybrid_splitstream import HybridConfig, SplitStreamHybridLM
from src.models.transformer_baseline import GPTBaseline, TransformerConfig

DEFAULT_WORDS = [
    "about","other","which","their","there","apple","grape","crane","slate","trace",
    "adieu","sound","round","stare","later","cigar","rebut","sissy","humph","awake",
    "blush","focal","evade","naval","serve","heath","dwarf","model","karma","grade",
    "quiet","bench","abate","feign","major","death","fresh","crust","stool","colon",
    "abase","marry","react","batty","pride","floss","helix","croak","staff","paper",
    "unfed","whelp","trawl","outdo","adobe","crazy","sower","repay","digit","crate",
    "cluck","spike","mimic","pound","maxim","linen","unmet","flesh","booby","forth",
    "alone","along","among","angel","anger","angle","angry","apart","arena","argue",
    "arise","array","arrow","aside","atoms","audio","avoid","award","aware","badly",
    "baker","bases","basic","basis","batch","beach","beans","beast","began","begin",
    "begun","being","below","bench","billy","birth","black","blade","blame","blank",
    "blast","blend","bless","blind","block","blood","bloom","board","boast","bones",
    "bonus","boost","booth","bound","brain","brand","brass","brave","bread","break",
    "breed","brick","brief","bride","broad","broke","brown","build","built","burst",
    "buyer","cabin","cable","calif","carry","catch","cause","chain","chair","chant",
    "chaos","charm","chart","chase","cheap","check","chest","chief","child","china",
    "chose","civil","claim","class","clean","clear","climb","clock","close","cloth",
    "cloud","coach","coast","could","count","court","cover","crack","craft","crash",
    "cream","crime","cross","crowd","crown","curve","cycle","daily","dance","dated",
    "dealt","death","debut","delay","delta","dense","depth","doing","doubt","dozen",
    "draft","drama","drawn","dream","dress","drill","drink","drive","drove","dying",
    "eager","early","earth","eight","elder","elect","empty","enemy","enjoy","enter",
    "entry","equal","error","event","every","exact","exist","extra","faith","false",
    "fault","favor","field","fifth","fifty","fight","final","first","fixed","flash",
    "fleet","floor","fluid","focus","force","forth","forty","forum","found","frame",
    "frank","fresh","front","fruit","fully","funny","giant","given","glass","globe",
    "glory","grace","grade","grain","grand","grant","grass","great","green","gross",
    "group","grown","guard","guess","guest","guide","happy","heart","heavy","hello",
    "hence","horse","hotel","house","human","ideal","image","index","inner","input",
    "issue","joint","jones","judge","juice","known","label","large","laser","later",
    "laugh","layer","learn","least","leave","legal","level","light","limit","local",
    "logic","loose","lower","lucky","lunch","lying","magic","major","maker","march",
    "match","maybe","mayor","meant","media","metal","meter","might","minor","minus",
    "mixed","model","money","month","moral","motor","mount","mouse","mouth","movie",
    "music","naked","nasty","naval","needs","never","newly","night","noise","north",
    "noted","novel","nurse","occur","ocean","offer","often","order","other","ought",
    "outer","owner","panel","paper","party","patch","pause","peace","phase","phone",
    "photo","piece","pilot","pitch","place","plain","plane","plant","plate","point",
    "pound","power","press","price","pride","prime","print","prior","prize","proof",
    "proud","prove","queen","quick","quiet","quite","quote","radio","raise","range",
    "rapid","ratio","reach","ready","refer","right","rival","river","robot","roman",
    "rough","round","route","royal","rural","scale","scene","scope","score","sense",
    "serve","seven","shall","shape","share","sharp","sheet","shelf","shell","shift",
    "shine","shirt","shock","shoot","short","shown","sight","since","sixth","sixty",
    "skill","sleep","slide","small","smart","smile","smith","smoke","solid","solve",
    "sorry","sound","south","space","spare","speak","speed","spend","spoke","sport",
    "staff","stage","stake","stamp","stand","start","state","steam","steel","stick",
    "still","stock","stone","store","storm","story","strip","stuck","study","stuff",
    "style","sugar","suite","super","sweet","table","taken","taste","taxes","teach",
    "teeth","thank","theft","their","theme","there","these","thick","thing","think",
    "third","those","three","throw","tight","times","title","today","total","touch",
    "tough","tower","track","trade","train","treat","trend","trial","tribe","trick",
    "tried","truck","truly","trust","truth","twice","under","union","unity","until",
    "upper","urban","usual","valid","value","video","virus","visit","vital","voice",
    "waste","watch","water","wheel","where","which","while","white","whole","whose",
    "woman","women","world","worry","worse","worst","worth","would","write","wrong",
    "wrote","yield","young","youth",
]


def load_word_list(path: Optional[str]) -> list[str]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            words = [w.strip().lower() for w in f if w.strip()]
        words = [w for w in words if len(w) == 5 and w.isalpha()]
        if words:
            return sorted(set(words))
    return sorted(set([w for w in DEFAULT_WORDS if len(w) == 5 and w.isalpha()]))


def byte_encode(text: str) -> list[int]:
    b = list(text.encode("utf-8", errors="ignore"))
    b.append(256)
    return b


def feedback(guess: str, target: str) -> str:
    out = ["B"] * 5
    used = [False] * 5
    for i in range(5):
        if guess[i] == target[i]:
            out[i] = "G"
            used[i] = True
    for i in range(5):
        if out[i] == "G":
            continue
        for j in range(5):
            if not used[j] and guess[i] == target[j]:
                out[i] = "Y"
                used[j] = True
                break
    return "".join(out)


def consistent(word: str, history: list[tuple[str, str]]) -> bool:
    return all(feedback(g, word) == f for g, f in history)


@torch.no_grad()
def score_candidate(model, device: torch.device, history: list[tuple[str, str]], cand: str) -> float:
    context = "Wordle\n"
    for g, f in history:
        context += f"guess={g} feedback={f}\n"
    context += "next="

    ctx_ids = byte_encode(context)
    full_ids = byte_encode(context + cand)
    x = torch.tensor(full_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(full_ids[1:], dtype=torch.long, device=device).unsqueeze(0)
    logits, _ = model(x, y)
    logp = torch.log_softmax(logits, dim=-1)

    start = len(ctx_ids) - 1
    end = len(full_ids) - 1
    s = 0.0
    for t in range(start, end):
        tok = y[0, t].item()
        s += float(logp[0, t, tok].item())
    return s


def choose_guess(model, device, candidates: list[str], history: list[tuple[str, str]]) -> str:
    if not history:
        return "slate" if "slate" in candidates else candidates[0]
    if model is None:
        return random.choice(candidates)
    best = None
    best_score = -1e30
    sample = candidates if len(candidates) <= 80 else random.sample(candidates, 80)
    for c in sample:
        sc = score_candidate(model, device, history, c)
        if sc > best_score:
            best_score = sc
            best = c
    return best or candidates[0]


def play_one(model, device, target: str, words: list[str], max_guesses: int = 6):
    history = []
    pool = words[:]
    for i in range(1, max_guesses + 1):
        guess = choose_guess(model, device, pool, history)
        fb = feedback(guess, target)
        history.append((guess, fb))
        if guess == target:
            return i, True, history
        pool = [w for w in pool if consistent(w, history)]
        if not pool:
            pool = words[:]
    return max_guesses, False, history


def eval_model(name: str, model, device, words: list[str], n_games: int, seed: int):
    rnd = random.Random(seed)
    targets = rnd.sample(words, min(n_games, len(words)))
    guess_counts = []
    wins = 0
    print(f"[wordle] model={name} games={len(targets)}")
    for t in targets:
        n, ok, hist = play_one(model, device, t, words)
        guess_counts.append(n)
        wins += int(ok)
        print(f"[wordle] model={name} target={t} solved={ok} guesses={n} trail={hist}")
    avg = sum(guess_counts) / max(1, len(guess_counts))
    win_rate = wins / max(1, len(targets))
    print(f"[wordle] model={name} avg_guesses={avg:.3f} win_rate={win_rate:.3f}")
    return {"model": name, "games": len(targets), "avg_guesses": avg, "win_rate": win_rate}


def load_models(cfg_path: str, baseline_ckpt: Optional[str], hybrid_ckpt: Optional[str], device: torch.device):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/science_fair_s33.yaml")
    parser.add_argument("--baseline-ckpt", default=None)
    parser.add_argument("--hybrid-ckpt", default=None)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--word-list", default=None, help="Optional newline-separated 5-letter word file")
    parser.add_argument("--randomize-targets", action="store_true", help="Use non-deterministic targets for live demos")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    baseline, hybrid = load_models(args.config, args.baseline_ckpt, args.hybrid_ckpt, device)

    words = load_word_list(args.word_list)
    seed = random.randint(1, 10_000_000) if args.randomize_targets else args.seed
    print(f"[wordle] target_pool={len(words)} seed={seed} randomize={args.randomize_targets}")
    _ = eval_model("baseline", baseline, device, words, args.games, seed)
    _ = eval_model("hybrid", hybrid, device, words, args.games, seed)


if __name__ == "__main__":
    main()
