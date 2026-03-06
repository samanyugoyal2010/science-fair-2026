# Judge Defense Sheet (ISEF)

## One-line positioning
This project is a controlled pilot comparison under tight local compute, not a production-scale architecture claim.

## What changed and why
- We removed the original memory-win claim because measured peak memory showed hybrid slightly higher in the reported runs.
- We now report tradeoffs across quality, memory, and speed instead of forcing a binary winner.
- We guard inference by replication count and label low-n results as pilot/underpowered.

## Core evidence policy
- Primary metrics: validation perplexity, peak memory, tokens/sec (and step time).
- Exploratory only: Wordle, tic-tac-toe, story generation.
- Pairing rule: comparisons are paired by the same context length and random seed.

## High-risk judge questions and answers
1. Why did your memory claim change?
- Because the measured values did not support a memory reduction. We corrected the claim to match data.

2. Why were earlier CIs weak?
- The main result had insufficient replication in earlier runs. We now gate inference and mark low-n results as non-conclusive.

3. Why not claim architecture superiority?
- The study is small-scale and undertrained; we only claim observed pilot tradeoffs under these constraints.

4. Why is hybrid slower?
- The SSM component uses sequential recurrence, which reduced throughput in our implementation.

## Required transparency statements
- Effective epochs and total training steps are disclosed.
- Model/data scale and tokenizer choice are disclosed.
- Hardware/device and exact commands are disclosed.
