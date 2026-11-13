# Brainstorm: binary-docking-gates

## Objective
Cut wasted computation in **existing, already-trained** transformer models. Each token gets a
cheap **binary (0/1) "docking" match** against blocks of weights, so only the blocks it matches
are computed. Inspired by RNA/receptor binding: weights carry an attach site, tokens dock only
where they fit, and the matched small matrices are fed forward.

What "done" looks like (Q6): **a write-up** of the idea for this first round, in the form of a
visual HTML explainer. Q6 said "no new code"; later answers (Q11, Q21–Q26, most complete test,
provenance) added a measurement suite, listed under Agreed scope. No fine-tune is run.
It specifies:
- docking on feed-forward units (Q7),
- a light fine-tune allowed to recover accuracy (Q8),
- speed judged by counted work, including the cost of the match (Q9).

## Context / current state
- **The diagnosis (Michael):** today's gating is soft. Every unit is computed, and weak matches
  are only multiplied down. GPUs handle per-element `if` badly, so a hard 0/1 match plus
  compaction (mask → index list → gather → one smaller dense matmul) is the hardware-friendly
  form.
- **Testbed already in place:** SmolLM2-135M (`smollm2:135m` in Ollama 0.34.3, F16 GGUF,
  30 layers × 9 heads, 576 wide, 1536 FFN units per layer).
  - `head-viz/extract.mjs` runs the full forward pass in Node 24 straight from the GGUF. It
    matches Ollama token for token on two prompts.
  - It can already ablate single heads and whole layers.
  - Visualizer: https://claude.ai/artifact/6hJQuG8Nikoks48jzDJ8vA
- **First measurements** (one prompt, "The capital of France is", target " Paris" at 46.9%):
  - 79 of 270 heads change P(target) by less than 1 point when removed alone.
  - 2 of 30 layers can be skipped with less than 1 point of change.
  - The logit lens shows " Paris" winning only at the last layer.
- **Machine:** Apple M5 Pro, 48 GB unified memory. Generating one token at a time is limited by
  memory bandwidth, so every skipped row is a row never read from memory.
  - No numpy installed. A nix flake is the sanctioned way to add tooling.
- **Prior work.** Each method gives up one of: cheap, accurate, training-free.
  - MoEfication (2022) and LLaMA-MoE (2024): cluster a trained FFN into experts and add a router.
    Needs retraining to recover accuracy.
  - Deja Vu (2023): learned predictors of contextual sparsity. About 2× faster on OPT, but the
    predictors cost compute.
  - PowerInfer: keeps often-firing ("hot") units on the GPU and rarely-firing ("cold") ones on
    the CPU.
  - ReLU Strikes Back (Apple, 2023): swap SiLU for ReLU so inactive units give exact zeros.
    Needs a fine-tune.
  - TEAL (2024): training-free magnitude threshold, about 40–50% sparsity with small loss.
    Crude.
  - SLIDE (2020): locality-sensitive hashing, binary match by XOR + popcount. Not shown working
    on pretrained transformers.
  - Product-key memory / PEER: 2√N comparisons to search N receptors.
- **Gap this targets:** a learned, binary, block-level docking code, fitted after training, that
  is cheaper than the work it skips and accurate. Not a crowded space.

## Agreed scope
- **In:** retrofit onto existing models, not train a new architecture from scratch (stated by
  Michael).
- **In:** the match signal is hard 0/1, not soft gating.
- **Leaning:** SmolLM2-135M as the first testbed, since it is small, verified and instrumented.
- **In:** feed-forward units first (Q7). Attention heads and whole layers come later.
- **In:** a light fine-tune of the model itself, to adapt to hard 0/1 gates (Q8).
- **In:** speed is counted work: rows read and multiply-adds, with the match's own cost
  included. Not wall-clock (Q9).
- **Deliverable this round:** a write-up, not code (Q6), as a visual HTML explainer artifact
  (Q12). It serves three readers at once (Q10):
  - precise enough to implement a later round from,
  - readable as Michael's own thinking aid,
  - framed like a proposal, with references and a clear statement of what is new.
- **The measurement suite** (grown from Q11 by later answers). All of it is on SmolLM2, before
  any fine-tune, so the cut numbers are a **worst case**:
  1. **Headroom (oracle bound):** the curve of units kept against output captured, by single
     unit and by blocks of 32. Blocks are formed both contiguous and clustered (Q14).
  2. **Per-domain docking-frequency heatmap** (Q21).
  3. **Sampled cut-and-test:**
     - two cut sets, target-domain and dead-everywhere (Q22);
     - both granularities (Q23);
     - with and without compensation (Q26);
     - measured by KL, top-1 agreement and perplexity (Q16);
     - cut in batches, giving a curve of fraction cut against KL.
  4. **Every result per context-length bucket**, and per position along growing chats (Q25).
  5. **The ceiling table per context-length bucket** (finding 14).
- **Code this needs:**
  - a KV-cached fast forward path, checked against the Node reference (which matches Ollama);
  - a dataset fetch and provenance pipeline;
  - the measurement runs above.
- **In: the most complete test (Michael, after Q26).** Coverage wins over speed:
  - all five domains,
  - the full range of context lengths up to 8,192,
  - growing chat histories,
  - the larger end of Q25's amounts.

  Run time is not a reason to shrink the set. A faster path is built instead.
- **In: full provenance of all source material (Michael).** Every text sample in the profiling
  set is documented with:
  - dataset name and publisher, source URL, and exact version, revision or commit;
  - licence;
  - retrieval date;
  - how samples were selected (seed, filters, offsets);
  - SHA-256 of each downloaded file and of each derived split.

  The write-up carries this as a sources appendix, so every number can be traced to the text
  it came from. Prior work cited in the write-up is listed the same way: authors, year, title,
  link.
- **In: sorting as a lossless step (Q3).**
  - Permute the 1536 FFN units so each docking block's `gate` rows, `up` rows and `down`
    columns are contiguous.
  - The permuted model must reproduce the original's output exactly. That doubles as a
    correctness check.
- **Future (not this round):**
  - the visualizer's re-sorting toggle;
  - algorithms that exploit sorted order: range checks, binary search, run-length
    compression, early-exit scans.
- **Non-goal:** training a new sparse model from scratch.
- **Non-goal this round:** a Metal/llama.cpp kernel or any real-runtime speedup.

## Approach directions
- **A. Block-level FFN docking:** group the 1536 units into blocks (for example 48 blocks of 32)
  by co-activation. Each block gets a docking code, and a token runs only the blocks it docks to.
- **B. Threshold baseline (TEAL-style):** keep units above a magnitude threshold. This is the
  bar any docking code must beat.
- **C. Binary hashing:** the docking code is a sign pattern. The match is a Hamming distance
  (XOR + popcount), nearly free.
- **D. Head docking:** the same idea for attention heads (skip whole heads per token).
- **E. Focused competency (added by Michael after Q12):**
  - Run many scenarios through the model, matched to what it is good at.
  - Use the docking signal as the instrument that finds rarely-docked ("dead") blocks per
    workload.
  - Record them as candidate waste, then devise strategies to **cut** them permanently.
  - The result is a smaller model that does most of one job without the edge cases.
  - Motto: "You don't ask a coder to write a poem."
  - This is static, done ahead of time. Docking (A) is dynamic, per token. They combine:
    profile → cut → dock what remains.
  - Caveats raised:
    - rare does not mean unimportant (closing-bracket blocks), so each block needs both
      docking frequency and ablation impact;
    - superposition means a cut can damage shared functions, so the cut model must be tested
      on the target workload;
    - gains grow with model size, so SmolLM2 will understate them.
  - Related work: calibration-based pruning (Wanda, SparseGPT), domain-specific pruning,
    per-task expert pruning in MoE.
  - Which of A or E is the core: **open (Q13)**.
- **Chosen:** A, block-level FFN docking (Q7), with B as the baseline it has to beat.
  - C is a candidate form for the docking code itself.
  - D is out for this round.

## Risks and tradeoffs
- The match must cost much less than the work it skips, or there is no gain.
- Retrofitting onto weights trained for dense compute loses some accuracy. Some training may be
  unavoidable (Q8).
- Single-unit granularity is too scattered for fast hardware. Blocks trade precision for speed.
- At large batch sizes, different tokens dock to different blocks, so the union shrinks the
  savings. Strongest at batch 1 (local use).
- A 135M model may not show the same sparsity as 7B+ models. Findings may not transfer.

## Edge cases / completeness gaps (adversarial pass)
Pass 1 by eve, a fresh agent briefed on the docking-only framing, before approach E was added.
Cost figures checked against the model shape.

1. **The headroom number is an oracle bound, and covers only a third of the FFN.**
   - In SwiGLU, `out = down · (silu(gate·x) * up·x)`. You only know which units carry 90% after
     computing all of `gate` and `up`.
   - So a hidden-unit magnitude threshold saves only `down`.
   - Pin baseline B to one real form:
     - TEAL thresholds the *input* `x` to every matrix;
     - CATS thresholds `silu(gate)`, which saves `up` and `down`.
   - Label the Q11 number an oracle bound. Only a predictor that never reads `gate` rows can
     reach it, and that predictor is the docking code. This is the core argument for docking
     and must be stated.
2. **The ceiling is low, and must sit next to the headroom figure.** Per token:

   | Part | Multiply-adds | Share |
   |---|---|---|
   | FFN | 79.6M | 59% |
   | attention projections | 26.5M | 20% |
   | tied LM head | 28.3M | 21% |

   - Skipping every FFN unit: at most about 2.45× fewer weights read.
   - Skipping half: about 1.4×.
3. **Binary is not needed for cost at blocks of 32.**
   - A float router over 48 blocks is about 27.6K multiply-adds per layer, about 1% of the FFN.
   - The real risk is **recall**: a missed block is lost output, and the losses add up over
     30 layers.
   - A sign-code Hamming match is SimHash: a 1-bit quantized linear router.
   - Binary only pays off on cost at fine granularity (hundreds of blocks or more), and it gives
     up matching accuracy.
4. **"Not a crowded space" is unbacked.**
   - Neighbours to check before citing: LTE (2024, routers fitted after training over grouped
     units, with hard thresholds and a fine-tune), CATS (2024), LLM in a Flash (Apple, 2023),
     GRIFFIN (2024), Hash Layers (2021).
   - The novelty is likely the **combination**, not a new mechanism.
   - The write-up needs a known / new / unverified table, with the novelty claim under
     "unverified" until checked.
5. **"90% of output" needs an exact definition.**
   - Unit contributions are 576-wide vectors that partly cancel. "90%" could mean the summed
     per-unit norms or the norm of the partial sum, and the two rank units differently.
   - The number is per layer and per token. It shows nothing about error adding up over layers,
     whether any predictor can reach it, or model quality.
   - Show the whole curve (units kept vs output captured), not one point. It comes at no extra
     cost.
6. **The hard-gate fine-tune needs a stated way to get gradients through a 0/1 decision.** One
   of:
   - (i) train the router as a classifier on oracle masks with the model frozen, then fine-tune
     the model with the router fixed (Deja Vu style);
   - (ii) a straight-through estimator for the gate and the sign code;
   - (iii) a soft or Gumbel-sigmoid gate annealed to hard.

   The fine-tune loss should be KL against the unmodified model.
7. **Edge cases for the spec.**
   - The match input is the layer's normalized hidden state, which depends on context, not just
     the token id. A static table by token id (Hash Layers style) costs nothing and is a floor
     baseline.
   - Define behaviour when a token docks to zero blocks or to all of them, and whether there is
     a minimum or maximum count.
   - Report counted work as a distribution per token, not a mean.
   - "Rows read" only holds after an offline re-layout that bundles each block's `gate` rows,
     `up` rows and `down` columns together.
8. **Out of scope (rejected by eve):**
   - union of blocks at large batch,
   - transfer from 135M to 7B (both already recorded as risks),
   - head docking and kernels (settled by Q7 and the non-goals).

Pass 2 by eve, a fresh agent briefed on both halves after Q13–Q19. Its questions are Q20–Q24
below. Its gaps:

9. **"One shared docking signal" is really two signals.**
   - Offline profiling can use the **oracle** mask: full compute, ranked by contribution.
   - Run time only has a **predictor**, with imperfect recall.
   - Profiling must use the oracle. If it used the router, every block the router wrongly misses
     would look dead and get cut, so the loop would confirm itself.
   - After a cut the block set has changed, so the router must be refitted, not reused.
10. **"Rarely docked" needs a statistical floor.**
    - A block that never docks in N tokens only bounds its rate below about 3/N at 95%
      confidence (rule of three).
    - About 500 measured tokens per domain bounds the rate only at about 0.6%, and a
      closing-bracket block can sit below that.
    - State the bound next to every "dead" claim, and set the sample size from the smallest rate
      a cut may ignore (Q25).
11. **Three disjoint text splits, not two:**
    1. form the blocks (Q14);
    2. choose the cut;
    3. test the cut model.

    Also test on held-out text from the **other** domains, so the write-up shows what the
    specialisation cost, not only what it kept.
12. **The static half needs its own ceiling table and must name what stays untouched.**
    - Cutting half the FFN leaves the tied embedding/output table (28.3M) and the attention
      projections (26.5M) whole, so the model is only about 30% smaller.
    - Pruning vocabulary rows changes the output's support, which breaks KL against the original
      unless it is renormalised.
    - Vocabulary and attention are **fixed this round**.
    - Attention still shifts indirectly, which the end-to-end KL captures.
13. **The cut needs a procedure, not a per-block score.**
    - Under superposition, single-block impacts do not add up when blocks are cut together.
    - Cut in rising order of score, in batches, re-measuring Q16 on held-out text after each
      batch.
    - Report a curve of fraction cut against KL. This fits Q17's rule of no threshold.
    - SiLU units never output exact zeros, so cutting a "dead" unit removes a small constant.
      The Llama-style FFN has no bias to absorb a correction. Name this, and say whether
      compensation (FLAP-style) is in scope.

Conflicts pass 2 found between decisions, both now asked:
- Q9 (counted work) against Q13c (equal halves): Q20.
- Q15's stated reason for breadth against approach E: Q22.

Neither blocks the write-up.

Pass 3 by eve, after Q20–Q26 and the two new requirements. Its questions are Q27–Q30. Its gaps:

14. **The ceiling falls sharply at long context.** Arithmetic verified. Per generated token at
    context length T, attention over the cache costs about 34,560·T multiply-adds and reads
    about 11,520·T cached values:

    | Context T | Attention multiply-adds | FFN share of all multiply-adds | Ceiling if all FFN is skipped |
    |---|---|---|---|
    | about 0 | — | 59% | about 2.45× |
    | about 2,300 | about 79.6M, equal to the FFN | — | — |
    | 8,192 | about 283M | about 19% | about 1.24× |

    The ceiling table must use the same context buckets as the headroom profiles.
15. **How a growing context is profiled.**
    - The model is causal, so one KV-cached pass over a full session gives the profile of every
      prefix. Record turn boundaries and absolute positions; no re-runs are needed.
    - The three disjoint splits are drawn per **whole document or conversation**, never per
      token or prefix.
    - N for the rule-of-three bound counts **independent documents** (or uses a cluster
      bootstrap), not tokens, because tokens within one context are correlated.
    - The first positions (BOS and attention-sink tokens) get their own bucket.
16. **Provenance must cover the whole chain from text to number:**
    - the GGUF digest;
    - the tokenizer and the exact ChatML template;
    - SHA-256 over the token IDs, not only the raw text;
    - the code commit and the nix `flake.lock`;
    - which forward path produced each number, plus the recorded equivalence check between the
      fast path and Node;
    - a per-sample licence where sets mix licences (The Stack, Stack Exchange CC BY-SA
      attribution);
    - pinned shards at a fixed revision, never streamed loads, so there is a file to hash.
17. **Scope text was stale and has now been corrected.** "No new code" and "one new measurement"
    were replaced by the actual measurement suite and code list above. Cut numbers come before
    any fine-tune, so they are a worst case.

Pass 4 by eve, the convergence check after Q27–Q33. It raised two conflicts (Q34, Q35) and one
gap. Its items were narrower than earlier passes, consequences of answers rather than new
directions:

18. **The chat generator's context window must be set to 8,192 explicitly and recorded.**
    - Ollama runs with a smaller window unless raised, and silently drops early turns.
    - SmolLM2's late replies would then be conditioned on a shortened history, while profiling
      replays the full one.
    - This would spoil the long-chat cells unseen, because the transcript files would not show
      it.
    - `num_ctx` goes into the recorded generation settings, and generation checks that no turn
      was truncated.

**Convergence.** After pass 4 and Michael's answers to Q34 and Q35, the loop is stopped:
- pass 4's items were consequences of answers, not new directions;
- no conflict between decisions remains open (Q36 is now answered).

## Open questions
| # | Question | Owner | Answer | Resolved |
|---|---|---|---|---|
| Q3 | Add a re-sorting toggle to the visualizer? | Michael | Keep it: "too many algorithms perform better on a sorted set". It is folded into the method: clustered blocks (Q14) are a lossless permutation of the FFN units, which also gives the re-layout that finding 7 requires. The visualizer toggle is a future item | yes |
| Q5 | Measure FFN headroom / threshold baseline on SmolLM2? | Michael | Answered by Q11 | yes |
| Q6 | What does "done" look like? | Michael | Q6d: a write-up, no new code | yes |
| Q7 | Where does docking apply first (FFN, heads, layers)? | Michael | Q7a: feed-forward units | yes |
| Q8 | May the retrofit include any training? | Michael | Q8c: a light fine-tune is allowed | yes |
| Q9 | Is the speed goal measured or real wall-clock? | Michael | Q9a: counted work, match cost included | yes |
| Q10 | Who is the write-up for? | Michael | A mix of design spec (a), own thinking aid (b) and paper-style proposal (d); mainly personal interest | yes |
| Q11 | Does the write-up include new SmolLM2 measurements? | Michael | Q11a: one headroom number (units, and blocks of 32, carrying 90% of FFN output) | yes |
| Q12 | What form does the write-up take? | Michael | Q12c: a visual HTML explainer (artifact) | yes |
| Q13 | Is the core idea focused competency (profile, cut, specialize) with docking as the instrument, or docking per token with specialization later, or both equally? | Michael | Q13c: both equally. One write-up, two halves (static cut and dynamic gate) sharing one docking signal | yes |
| Q14 | How are blocks of 32 formed for the headroom measurement (contiguous, clustered by co-activation, or both)? | Michael | Q14c: both, compared side by side | yes |
| Q15 | What text is the headroom measured on? | Michael | Q15c + Q15d: a broad set of thousands of tokens, split into separate domain sets, with blocks formed on one half and measured on the other. The breadth is there so a small but important block is not cut unseen | yes |
| Q16 | Which accuracy metric does the write-up commit to? | Michael | Q16d + Q16c: KL divergence from the original, top-1 agreement, and perplexity on held-out text | yes |
| Q17 | What headroom result means stop, and what means continue, fixed before measuring? | Michael | Q17c: no threshold. Report the numbers as found and discuss them without a verdict | yes |
| Q18 | Where does the domain text come from (public datasets, local files, or both)? | Michael | Q18a: small samples from public open datasets, so the results are reproducible and shareable | yes |
| Q19 | Which domains make up the broad set? | Michael | Code, prose, maths, chat (ChatML), plus **troubleshooting**, which Michael added | yes |
| Q20 | By counted work, the static cut adds little on top of the dynamic gate. How is the static half measured: counted work only, model size too, or both with a rule never to sum the two halves? | Michael | Q20a: counted work only. The write-up states plainly that by this measure the static half adds little on top of the dynamic half | yes |
| Q21 | Does the static half get data of its own: spec only, a free per-domain docking-frequency heatmap, or the heatmap plus ablation of sampled candidate blocks? | Michael | Q21c: the per-domain heatmap, plus cut-and-test on a small sample of candidate blocks | yes |
| Q22 | Is the cut decided on the target domain's profile (breadth only measures the cost elsewhere), only on blocks dead in every domain, or both reported? | Michael | Q22c: both cut sets, side by side, each with its cost | yes |
| Q23 | Does the static cut use blocks of 32, single units, or both? | Michael | Q23c: both, so the cost of block granularity shows | yes |
| Q24 | Where does the fine-tune sit in profile → cut → dock? | Michael | Q24b: two fine-tunes: cut → recovery fine-tune → re-profile → fit router → short gate fine-tune. Described in the write-up, not run this round | yes |
| Q25 | How much text per domain, given the rule-of-three floor and the cost of the Node forward pass? | Michael | "Different context sizes respond differently, so a mix of all", plus a **simulated growing context, like a chat history**. Profiles therefore span short prompts up to long contexts toward the 8,192-token limit, and include multi-turn chats whose history accumulates turn by turn. Amounts range from about 1k to 30k+ tokens per domain | yes |
| Q26 | Include FLAP-style compensation for the constant a cut SiLU unit removes? | Michael | Q26b: include it. The sampled cuts are reported with and without compensation | yes |
| Q27 | What makes the test set complete rather than just large: a smallest-ignorable rate, equal cells, or Q25's amounts with per-cell bounds? | Michael | Q27a: a floor of 1%. Every domain × context-length cell gets about 300 independent documents, so any block firing in more than about 1% of documents is caught. Long documents count for every shorter cell. Roughly 12M tokens in total | yes |
| Q28 | How do domains with naturally short text reach 8,192 tokens? | Michael | Q28d: naturally long sources where they exist; otherwise independent samples packed with separators and labelled as artificial length | yes |
| Q29 | What does "simulated" growing chat history mean? | Michael | 300 chats with Haiku workers on inane topics, with openings inspired by popular r/NoStupidQuestions and r/AskReddit questions. Details in Q31–Q33 | yes |
| Q30 | What gets published: identifiers only, or the derived texts too? | Michael | Q30a: identifiers only (dataset revisions, row IDs, offsets, hashes). The generated chats from Q29 have no external identifier, so they are kept as files with hashes | yes |
| Q31 | "anine topics": anime, inane (NoStupidQuestions-style), or any/varied topics? | Michael | Q31a: inane topics. Silly, everyday, curious questions across all kinds of subjects | yes |
| Q32 | In the 300 chats, who plays which role: Haiku as user and SmolLM2 answering, Haiku on both sides, or something else? | Michael | Q32a: Haiku asks and SmolLM2 answers, as in real local use. The write-up states the bias of self-generated history | yes |
| Q33 | Opening questions: real Reddit posts, or written by Haiku in the style of those subreddits? | Michael | Haiku searches the most popular questions on r/NoStupidQuestions and r/AskReddit (about 300) and **rewrites each in its own words**: inspired by the posts, not their sentences. The Haiku workers vary the chat length deliberately, from short to long back-and-forths, so responses can be measured by position in the conversation | yes |
| Q34 | The 1% floor needs 300 documents in the split that chooses the cut, but 300 split three ways leaves about 100 (a 3% floor). Which split holds 300? | Michael | Q34b: the deciding split gets 300, and the block-forming and test splits about 100 each. About 500 per cell and about 20M tokens | yes |
| Q35 | 300 chats of varied length leave few in the longest buckets. How do the chat cells meet the floor? | Michael | Generate long chats, and cut the shorter length items from them (prefixes at turn boundaries). Prefixes of one chat are not independent, so each chat still counts once toward the floor | yes |
| Q36 | Q35 says "a few long chats", but the floor (Q27, Q34) needs about 500 independent long chats for the chat domain. Is it as many as the floor needs, or literally a few, with the chat cells reported at a weaker bound? | Michael | Only long chats are generated, as many as the floor needs (about 500: 300 deciding + 100 + 100). No short chats are generated separately; every shorter length item is derived from the long chats by cutting at turn boundaries | yes |
| Q37 | Decision gate: a structured plan, or implement as is? | Michael | | no |

## Assumptions
- Accuracy is judged against the unmodified model's own output (it is the reference), not
  against ground truth. Perplexity (Q16) is the one measure taken against real held-out text.
- Q25's long contexts and 30k+ token sets are out of reach for the current naive Node forward
  pass:
  - it uses no KV cache and full O(T²) attention, at roughly 0.08 s per token on short prompts;
  - the measuring code therefore needs a faster path: a KV cache, plus a numpy/Accelerate route
    via a nix flake, which Michael sanctioned;
  - this is an implementation need, not a scope change.
- Context length becomes a reported dimension of the profiles. Docking frequency and headroom
  are shown per context-length bucket, and per position along a growing chat.
- **The chat pipeline (Q29–Q33).** Provenance for each chat records:
  - the source post URLs or IDs behind each rewritten opening question;
  - the Haiku model ID and the system prompts used for the user role;
  - SmolLM2's generation settings (temperature, seed) for the assistant role;
  - the full transcript file and its SHA-256.

  Generation through Claude is not seed-reproducible, so the stored transcripts are the
  reproducible artifact.
- Chat lengths are spread on purpose, from a single exchange to histories near the
  8,192-token limit.

## Undecided things
- Exactly which public datasets supply each of the five domains, and their licences.
- Block size beyond the first choice of 32.
- Which form of baseline B to use (TEAL input threshold or CATS gate threshold): finding 1.
- How gradients get through the hard gate: finding 6.
- Whether the novelty claim survives a check of LTE, CATS, LLM in a Flash, GRIFFIN and Hash
  Layers: finding 4.
