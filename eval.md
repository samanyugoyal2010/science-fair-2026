## Evaluation guide

This file shows how to run all evaluation scripts and how to tell which model (baseline transformer vs split‑stream hybrid) is better on each.

All commands assume you are in the project root (`fair-p2`) and that the s33 checkpoints exist:

- Baseline: `results/checkpoints/s33/baseline_ctx1024_s33.pt`
- Hybrid: `results/checkpoints/s33/hybrid_ctx1024_s33.pt`

If your checkpoints are in a different location, replace the paths accordingly.

---

### 1. Held-out perplexity (language modeling quality)

**Command:**

```bash
python3 -m src.eval.heldout_ppl_eval
```

This uses:

- Config: `configs/science_fair_s33.yaml`
- Default checkpoints (s33 ctx1024) from `results/checkpoints/s33/`
- A small built-in evaluation passage

**How to read it:**

- It prints something like:

  - `baseline  loss=X  ppl=Y`
  - `hybrid    loss=A  ppl=B`
  - `ratio (hybrid/baseline) ppl = R (hybrid better | baseline better)`

- **Better model:** the one with **lower perplexity (`ppl`)**.  
  If the ratio line says `hybrid better`, the hybrid wins; if it says `baseline better`, the transformer wins.

To use your own text file:

```bash
python3 -m src.eval.heldout_ppl_eval --eval-data data/local_corpus.txt
```

---

### 2. Cloze completion (prefix → correct continuation)

**Command:**

```bash
python3 -m src.eval.cloze_eval
```

This scores both models on the same (prefix, continuation) pairs and prints:

- `baseline  avg_logp = ...`
- `hybrid    avg_logp = ...`
- `diff (hybrid - baseline) = ... (hybrid better | baseline better)`

**Better model:** the one with **higher `avg_logp`** (average log-probability of the correct continuation).  
If the diff line is positive and says `hybrid better`, the hybrid wins; if negative and says `baseline better`, the transformer wins.

To use your own TSV file (`prefix<TAB>continuation` per line):

```bash
python3 -m src.eval.cloze_eval --pairs-file path/to/pairs.tsv
```

---

### 3. Wordle-style reasoning eval

**Command:**

```bash
python3 -m src.eval.wordle_eval \
  --config configs/science_fair_s33.yaml \
  --baseline-ckpt results/checkpoints/s33/baseline_ctx1024_s33.pt \
  --hybrid-ckpt   results/checkpoints/s33/hybrid_ctx1024_s33.pt \
  --games 50 \
  --seed 33
```

**Options:**

- `--config`, `--baseline-ckpt`, `--hybrid-ckpt` — config and checkpoint paths (defaults use s33 ctx1024).
- `--games` — number of Wordle games per model (default 20).
- `--seed` — random seed for target words (default 33).
- `--word-list` — optional path to a newline-separated file of 5-letter words; if not set, a large built-in dictionary (500+ words) is used.
- `--randomize-targets` — use a different random seed each run for live demos.

**How to read it:**

- For each model, it prints lines like:

  - `[wordle] model=baseline ... solved=... guesses=...`
  - `[wordle] model=hybrid ...`

- At the end, it prints for each model:

  - `avg_guesses=...`
  - `win_rate=...`

**Better model:**

- Higher **`win_rate`** is better.
- Lower **`avg_guesses`** is better (solves in fewer tries).

You can increase `--games` or change `--seed` to explore different target sets.

---

### 4. Story generation (qualitative coherence)

**Command:**

```bash
python3 -m src.eval.story_eval
```

**Options:** `--config`, `--baseline-ckpt`, `--hybrid-ckpt`, `--device`, `--prompt`, `--max-new-tokens`, `--temperature`, `--top-k`, `--seed`. Defaults use `configs/science_fair_s33.yaml`, s33 checkpoints in `results/checkpoints/s33/`, and prompt: `Once upon a time, in a small village by the sea,`

It prints two sections:

- `=== Baseline (GPT) ===` → story from the transformer baseline
- `=== Hybrid (Split-Stream) ===` → story from the hybrid

**Better model (qualitative judgment):**

- Read both stories and judge which one:
  - Uses fewer non-words / gibberish tokens
  - Maintains a more coherent sentence structure
  - Feels more like real text

You can change the prompt, for example:

```bash
python3 -m src.eval.story_eval \
  --prompt "In the emergency room, the doctor examined the patient and noticed that" \
  --max-new-tokens 200
```

---

### 5. Tic-tac-toe (best of 3: baseline vs hybrid)

**Command:**

```bash
python3 -m src.eval.tic_tac_toe_eval
```

**Options:**

- `--config`, `--baseline-ckpt`, `--hybrid-ckpt` — config and checkpoint paths (defaults use s33 ctx1024).
- `--device` — e.g. `cpu`, `cuda`, `mps` (default: MPS if available, else CPU).
- `--quiet` — only print final best-of-3 summary, not per-move output.

**How it works:** Baseline (transformer) and hybrid (split-stream) play best of 3. The **hybrid always plays X (goes first)**; the baseline plays O in all three games. Each model chooses moves by next-token log-probability on cells 1–9. Output: per-game moves (unless `--quiet`) and a final summary of baseline wins, hybrid wins, and draws.

---

### 6. Quick summary of “who is better where”

After running the above:

- **Held-out perplexity:** better = **lower `ppl`** (and check the ratio line).
- **Cloze:** better = **higher `avg_logp`** (and check the `diff` line).
- **Wordle:** better = **higher `win_rate`**, **lower `avg_guesses`**.
- **Story:** better = **more coherent, less gibberish** (your human judgment).
- **Tic-tac-toe:** better = **more wins** in the best-of-3 match.

You can copy these results into your report/poster as a table comparing baseline vs hybrid across all evals.

