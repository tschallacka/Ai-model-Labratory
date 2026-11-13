# Lab book

Everything tried in the binary-docking-gates / petri-dish work: the question, the method, the numbers, the
conclusion. Dead ends and corrections stay in; this is throwing things at the wall to see what sticks.

Times are Europe/Amsterdam. Exact times come from git commits or file timestamps; `~` marks a time
reconstructed from the conversation order. Model throughout: SmolLM2-135M-Instruct as served by Ollama
(`smollm2:135m`, GGUF sha256 `f535f83e…`), D = 576, 30 layers, 9 query heads and 3 shared key/value heads
of 64, 1536 FFN units per layer, 49,152 tokens.

Earlier records, copied in as they were on 2026-11-12:
- [brainstorm-2026-11-12.md](brainstorm-2026-11-12.md): decisions Q3–Q37 and review findings 1–18 that
  shaped the measurement. Q37 shows as open there; it was answered 37b (implement directly, no plan).
- [related-work-2026-11-12.md](related-work-2026-11-12.md): Deja Vu, CATS, LTE, MoEfication, LLM in a Flash,
  Hash Layers, GRIFFIN, D-Pruner, Voita et al. Read at abstract level only.

---

## 2026-11-12

### ~16:10 · Seeing the model: heads and layers
- **Question:** how does a token flow through the layers and heads?
- **Method:** a Node forward pass straight from the GGUF (`head-viz/extract.mjs`, outside this repo) and a
  first visual artifact.
- **Result:** reference values for later checks: "The capital of France is" → " Paris" with p = 0.4693,
  continuation " Paris. Paris is the".

### ~16:50 · Brainstorm: binary docking gates
- **Idea (Michael):** tokens "dock" 0/1 onto blocks of FFN units, like RNA binding. Two equal halves: a
  per-token gate (skip undocked blocks) and "focused competency" (cut blocks a domain never uses: "you don't
  ask a coder to write a poem"), retrofitted onto an existing model.
- **Record:** brainstorm-2026-11-12.md.

### 17:30–17:41 · Environment and equivalence checks
- **Method:** a nix flake (`env/`) with numpy, pyarrow, tokenizers and torch. Three forward paths, each
  checked against the previous one.
- **Results (`data/checks/model-equivalence.json`):**
  - numpy vs Node: max relative difference 2.7e-4; continuations match Ollama;
  - the HF tokenizer's vocabulary equals the GGUF's, and its token counts match Ollama on 5 texts;
  - torch (Apple GPU) vs numpy: KL ≤ 1.2e-6, top-1 agreement 1.0 over 2048 tokens; 1.131 s per
    8192-token pass.

### 17:39–17:53 · Sources and documents
- **Method:** `fetch_sources.py`: pinned dataset shards, each with revision, URL, licence, time, size and
  sha256 (`sources/manifest.json`). `build_docs.py`: 500 documents of 8,192 tokens per domain (code, prose,
  maths, troubleshooting), with provenance per document.
- **Chat openers:** Michael pasted r/NoStupidQuestions search results. My first filter kept only 31 of
  104; Michael objected, and I revised to 97 kept (only meta/moderator/image-only/drama posts dropped,
  sensitive ones tagged rather than removed). Plus 553 r/AskReddit questions by score.

### 18:08 · Chat generation
- **Method:** `chat_gen.py`: an asker model plays the user, SmolLM2 answers, until the history nears 8,192
  tokens. ChatML, `num_ctx` 8192 set explicitly, token counts checked so nothing gets truncated.
- **Change:** Haiku as the asker needed ~51 calls per chat; Michael stopped that. Switched to local
  gemma3:27b. 3 Haiku chats kept as samples.
- **Observation:** SmolLM2 falls into repetition in long chats: from ~3,000 tokens on, replies settle at
  75–80 tokens each.
- **Rate:** 45 of 500 chats after 2 h 40 min (~3–3.5 min per chat).

### ~18:10 · First docking numbers (8 documents)
- **Method:** `profile_ffn.py`: docking set = the fewest units (or blocks of 32) whose
  |activation| × ‖W_down column‖ carries ≥ 90% of the FFN output at a token.
- **Results:** 550–855 of 1536 units per token; clustered blocks need ~39 of 48, contiguous ~41; output
  error with only the docking set 3–10%.

### 18:15 · Measurement chain
- `run_measurements.sh` waits for the chats, then runs block clustering → profiling → `cut_test.py`.
- Smoke test of `cut_test.py` on 2 documents: works end to end. At a 20% cut, adding the cut units' mean
  output back as a bias came out slightly *worse* than not (2 documents: not meaningful yet).

### ~18:20 · Related work
- Closest per-token work: Deja Vu, CATS, LTE; MoEfication forms blocks by co-activation exactly as our
  clustering does. Closest to focused competency: GRIFFIN, D-Pruner; Voita et al. found "dead" neurons in
  OPT. What looks new: one docking record serving both halves, measured per domain and per context length.

### 20:51 · Pause scheduled
- `pause_at.sh` stops the runner and the chat generator at 2026-11-13 08:45 (Michael needs the laptop);
  resume only on Michael's word.

### ~21:00 · Tangent: why this session was missing from `claude --resume`
- **Finding:** the resume picker only lists sessions whose recorded `entrypoint` matches its own. `-p` and
  stream-json sessions (MonoCode) record `sdk-cli`; an interactive terminal is `cli`. Setting
  `CLAUDE_CODE_ENTRYPOINT=cli` on a `-p` run does not change what is recorded. `claude --resume <id>` opens
  any session regardless.

### 22:03 · Petri viewer, first version
- **Method:** `petri/trace.py` repeats the forward pass with explicit attention to record its weights, and
  refuses any trace that differs from the verified pass (agreement ~1e-6). Blocks are provisional: clustered
  from 20 documents, no chats.
- **First reading:** the code sample's last token docks 38 of 48 blocks in layer 1, 8.7% output error with
  only those. Consistent with the ~39 above.

### ~22:20 · Publishing quirks
- The claude.ai artifact host serves no binary file types: traces now ship as base64 text (`.b64.txt`).
- Artifacts were off in the CLI session (it lacked `CLAUDE_CODE_ARTIFACT=1`). A headless
  `CLAUDE_CODE_ARTIFACT=1 claude -p … --allowedTools "Read,Artifact"` published it.

### ~22:45 · Finding: giant outlier values in the stream
- **Trigger:** the viewer's by-hand head calculation was 42% off from the recorded value.
- **Cause:** per-vector 8-bit storage of the trace. In one stream vector the largest value is **1,440×**
  the median, so 8 bits leave the typical values almost nothing.
- **Measured 8-bit error on values ≥ the vector's median (median / worst 5%):**

  | Section | Largest ÷ median | Median error | 95th percentile |
  |---|---|---|---|
  | stream | 1440 | 3.8% | 17.6% |
  | attention output | 213 | 1.0% | 8.7% |
  | detector output | 1313 | 1.1% | 5.6% |
  | keys | 516 | 1.0% | 3.9% |
  | queries | 59 | 1.0% | 3.8% |
  | values / head outputs | 100 | 0.5–0.6% | 1.7–4.1% |
- **Fix:** stream, both updates and keys stored as float16. With the exact stream, the hand calculation
  (rescale, 576 multiplies, sum, rotate) matches the model to 0.06%.
- **Viewer consequence:** colours now scale to each vector's 98th percentile, not its maximum.

### 23:21–23:36 · Repository and site
- Public repo `tschallacka/Ai-model-Labratory` (viewer and server only: allow-list `.gitignore`), deployed
  to petri.tschallacka.dev. The static page lacked the viewport tag (the artifact host adds it, a plain host
  does not), so phones got a desktop layout: fixed. `version.json` (a hash of all served files) to poll
  after a deploy; deploys take ~10–15 s.

### 23:40 · Rotation speeds
- Pair p of a head turns 100000^(−2p/64) radians per token: pair 0 turns 57°, pair 4 14°, pair 8 3.2°,
  pair 31 under 0.001°. The pair with the biggest weights tends to be a slow one (block 0, layer 1: 0.82°
  per token), so the viewer now offers pairs 0, 4 and 8 too.

---

## 2026-11-13

### 00:13 · Finding: the layer-1 multipliers mute, boost and flip
- **Trigger:** Michael saw a value that was not a spike become one after "evening out".
- **Method:** compare the stream before and after RMSNorm for "def" in layer 1, then the gains of all
  layers.
- **Results:** the volume knob (×9.41) keeps the shape; the per-position gains (`attn_norm.weight`) change it:

  | Position | On arrival | Gain | After |
  |---|---|---|---|
  | #471 | −0.198 (31st) | −0.305 | +0.569, biggest, sign flipped |
  | #312 | +0.095 (202nd) | +0.315 | +0.280, 4th |
  | #322 | −0.440 (biggest) | +0.0125 | −0.052, 61st |
  | #199 | +0.434 (2nd) | −0.006 | −0.026, 153rd, flipped |

  Gains per layer (median |g|, max, negatives of 576):

  | Layer | Median | Max | Negative |
  |---|---|---|---|
  | 1 | 0.022 | 0.314 | **271** |
  | 2 | 0.159 | 0.617 | 3 |
  | 15 | 0.388 | 0.816 | 0 |
  | 30 | 0.408 | 0.910 | 0 |
  | final | 1.781 | | 2 |
- **Conclusion:** the heavy muting and the mass sign-flipping happen almost only in layer 1.
  **Unverified guess:** layer 1 is the only layer reading raw token-table rows, and that table is tied to
  the output layer, so its values suit the output more than layer 1.
- **Correction:** the viewer had called the gains "a small boost or damping"; they can be large and
  negative. The viewer now shows the volume knob and the multipliers as separate parts.

### ~00:20 · Where the parameters and the bytes are
- **Parameters:** token table 28,311,552 + 30 × 3,540,096 per layer (attention 884,736; detectors
  2,654,208; two gain sets 1,152) + 576 final gains = **134,515,008**. The table is ~21%. The layers are
  shared by every token: one set of weights per layer, not per token.
- **File (270,885,952 bytes):** 134,479,872 numbers in F16 = 269.0 MB; 35,136 gains in F32 = 0.1 MB;
  tokenizer strings and 48,900 merges ≈ 1.8 MB.
- **RAM while serving:** ~624 MB. The extra is mostly the KV cache: Ollama reserved 3 parallel slots ×
  8,192 tokens (q8_0 cache). Cheapest RAM lever: a smaller context.

### ~00:30 · Tangent: rows look alike once sorted
- **Question (Michael):** if the numbers are all similar, why store them all?
- **Measured on layer-1 W_q (576 rows):**
  - two rows as stored: similarity −0.005; after sorting each row: **+0.991**;
  - one shared sorted "shape" + each row's reshuffle + one scale per row: **10.7% error** at 7.7 bits per
    weight (log2(576!)/576); layer-15 gate 9.5%, layer-30 down 7.5%;
  - plain 8-bit (one scale per row): ~0.8% error. Plain 4-bit: 15–18%. A crude shape-based 4-bit codebook:
    17–23%;
  - where the reshuffle error sits: 30–37% in the biggest 1% of weights; the rest spread over the middle,
    where each value is off by ~5% of the row scale;
  - zlib / xz on raw F16 weights: 77% / 72% of the size.
- **Conclusion:** the set of values is shared; the order is the information, and storing an order costs
  about as much as storing the values. Related known work: NF4 (QLoRA) places 4-bit levels at bell-curve
  quantiles.

### ~00:40 · Tangent: sort the token rows offline, keep the shift
- **Idea (Michael):** a one-time pass sorts each token row and stores values in sorted order, making
  positions implicit.
- **Measured, lossless, on 2,000 token-table rows at exact F16 values:**

  | Encoding | Bits per value |
  |---|---|
  | as stored | 16 |
  | sorted values (steps between neighbours) + shifts as stored | 3.42 + 9.89 = 13.3 |
  | same, with the theoretical minimum for an order | 3.42 + 7.74 = 11.2 |
  | plain entropy coding, no sorting | **10.6** |

  Shifts average 194 places; only 0.8% are within ±2.
- **Follow-up:** the shifts can only be absorbed into the weights if one reordering is used for all tokens
  (a valid, output-preserving change of coordinates). The best single reordering (positions by average rank
  over 3,000 rows) makes rows **+0.536** sorted on average (best row +0.776). The remaining shifts still
  cost 9.33 bits. **There is real shared structure** (positions systematically high or low across tokens),
  but it does not pay for itself as compression.
- **Also measured:** two weight rows' orders agree −0.006 / +0.002 / +0.000 (layers 1, 15, 30); a weight
  row's order vs the token row's order: 0.078 / 0.039 / 0.035. Unrelated, so a runtime sort of the input
  cannot stand in for stored positions.

### 00:53 · Petri dishes
- **Method:** `petri/mini.py` (see `petri/DISHES.md`).
- **`code` dish:** output " fibonacci(n - 2)\n\n# Example usage:\nprint" (16 tokens).
  - 24 of 49,152 token rows; **38,019 of 46,080 detectors (82.5%)**; 92.3M of 134.5M numbers (68.6%);
    185 MB.
  - Kept per layer: 991, 342, 435, 526, 717, 831, 1008, 1134, 1277, 1359, 1375, 1434, 1403, 1327, 1348,
    1389, 1402, 1444, 1471, 1497, 1521, 1517, 1527, 1531, 1535, 1536, 1535, 1536, 1536, 1535.
  - Closest decision margin 0.01 at the cut (0.25 uncut).
- **Conclusion:** even one fixed sentence needs most detectors in total. Early layers (2–4) can lose two
  thirds; the last ten almost nothing.

### ~01:00 · Changing the model without changing the output
- `permute.py`: reordering the 576 positions everywhere moves the scores ~2e-6 (rounding); output unchanged.
  The sorted order makes this dish's 24 rows +0.32 sorted.
- Measured edits on copies of `code`: see the table in `petri/DISHES.md` §6. Corrections to my own
  expectations: scaling a layer's gains by 2 *does* change the output (score change 14.1); one more
  detector removed (the first kept one in layer 21) does *not* flip a token, so the rank order left room;
  8-bit rounding of all matrices flips the output on this knife's-edge dish (score change 1.03).

### 01:05 · Lab notes in the repository
- This lab book, plus copies of the brainstorm and related-work notes, committed to `notes/` (Michael: "It is
  all gathering data, throwing stuff against the wall to see what's useful"). Entries are written as findings
  are made, then committed and pushed.

### 01:15 · Run log for every experiment script (Q54a)
- **What:** `notes/runs.jsonl`, append-only, one JSON line per run of `mini.py`, `permute.py` and a dish's
  `run.py`, failed runs included. A run's id is its line number, so a single run can be read without scanning
  the file. Each script ends with a reminder on stdout: write a lab-book entry? keep the log line, whole or in
  part? (`petri/runlog.py keep|drop <id> "why" [part ...]`). Details in `petri/DISHES.md` §7.
- **Caught while testing:** the first version wrote absolute local paths (with the user name) into the log,
  which goes to a public repository. Paths are now relative to `docking/`; the three test lines were discarded
  before any commit and the log restarted.
- **Runs 1–3** re-made `code` and `code-sorted` under the new log (same numbers as at 00:53 and ~01:00); run 2 is
  marked keep (meter, unchanged, sortedness_after) in line 4. Saved at 06:34 as
  [batches/2026-11-13-0634-run-log-introduction.jsonl](batches/2026-11-13-0634-run-log-introduction.jsonl).

### 06:34 · The run log becomes per-batch scratch
- **Michael:** the log is git-ignored and started fresh before each batch (a tangent, idea, experiment,
  proof or falsification attempt), with the relevant parts kept next to the notes.
- **What changed:** `runlog.py start "…"` (refuses while the previous batch is open) and
  `runlog.py close [name]`, which writes only the kept runs, and only their kept parts, to
  `notes/batches/<date-time>-<name>.jsonl`. `notes/runs.jsonl` left git. Manual: `petri/DISHES.md` §7.
- **Tested:** the refusal while a batch is open, a drop, and a close with 0 kept (that test batch file was
  deleted). The 01:14 log was closed as the first batch file (1 kept, 2 unmarked).
- **Not done:** a peer session asked for a review of an ai-text-editor structured-edit RFC in chat
  `#ai-skills`. Joining was blocked by Claude Code's permission check, and the chat server refused the
  connection. Later sent directly to that session with SendMessage, at Michael's suggestion, without having
  read the RFC itself. The view: locate edits structurally but replace only the touched bytes; address JSONL as
  line + key path; make append a first-class verb; guard each edit with the expected value; parse only the
  addressed line in huge files; leave YAML in text mode unless comments survive.

---

## Open threads
- **Q50:** measure 8-bit / 4-bit SmolLM2 (size, RAM, speed, perplexity).
- **Q51:** input-side skipping (skip weight columns for small inputs), next to docking.
- **Q52a (chosen, next):** which positions are systematically high or low across tokens; how they relate to
  the outliers and the layer-1 gains; how they behave through the layers.
- Dishes: cut attention heads too; a per-detector greedy pass (the rank order left room); dishes for the
  other samples.
- The measurement run: resume after 08:45 on Michael's word; then rebuild the viewer with the final blocks.
