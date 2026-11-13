# Petri dishes: manual

A petri dish is SmolLM2-135M cut down to exactly what **one input text** needs to produce **its own output**.
It knows that one input and nothing else. That makes it a small, fixed sample you can copy, change and run
again, to see which changes to a model leave its output alone and which do not.

All commands run from the `docking/` folder.

## 1. Before you start

- Build the Python environment once: `nix build path:./env --out-link env/result`.
- The model must be in Ollama: `ollama pull smollm2:135m`. `mini.py` reads the model file straight from
  Ollama's storage (the path is `BLOB` in `profile_ffn.py`).
- A finished dish needs only Python 3 with numpy. Your Mac's own `python3` has no numpy, so use
  `env/result/bin/python3`, or any Python that has numpy.

## 2. Make a dish

```sh
env/result/bin/python3 petri/mini.py code                              # one of the five samples
env/result/bin/python3 petri/mini.py fox "The quick brown fox" 12      # your own text, up to 12 new tokens
```

The first argument names the dish. With only a name, the text comes from `petri/prompts.json` (`code`, `prose`,
`maths`, `troubleshooting`, `chat`). The last argument is the most tokens it may add (default 16); it stops
earlier at an end-of-text token. It takes about 15 seconds and writes `petri/dishes/<name>/`.

What `mini.py` does:

1. **Runs the full model** on the input and records its output, always taking the most likely next token.
2. **Shrinks the vocabulary** to only the tokens in the input and the output. The full model's choice is the top
   token over all 49,152, so it is also the top token among these few, and the output stays the same.
3. **Cuts detectors.** Every detector (FFN unit, 30 layers × 1536) gets a score: its biggest contribution
   anywhere in this text. The dish keeps the fewest top-scoring detectors that still pick every output token,
   found by binary search. Attention is kept whole.
4. **Saves and checks.** It saves the dish, runs the dish's own `run.py`, and requires the same output. Then it
   records the dish's exact scores as the baseline for the change meter (section 4).

What the `code` dish came out as: 24 of 49,152 token rows, 38,019 of 46,080 detectors (82.5%), 68.6% of all
numbers, 185 MB. Early layers lose the most detectors (layer 2 keeps 342 of 1536); the last ten keep almost all.

**A dish is cut to the bone on purpose.** At the most aggressive cut, the closest decision is won by about 0.01
points (the uncut model wins it by 0.25). Removing the next detector in rank order flips a token, and small
edits such as rounding the weights to 8 bits can too. That makes the dish a sensitive instrument: if an edit
leaves the output *and* the scores alone (see the change meter), it really did not change what the model does.

## 3. Run a dish

```sh
env/result/bin/python3 petri/dishes/code/run.py
```

It prints the output, then the change meter, then `output unchanged` or `OUTPUT CHANGED`; the latter also
exits with an error. Given any other input text, it refuses: the tokens and detectors that text would need were
removed.

## 4. The change meter

```
largest score change vs the original: 2.17e-06 · closest decision wins by 0.0098
output unchanged
```

- **Largest score change:** at every output position the dish gives each of its tokens a score. This is the
  biggest difference from the scores recorded when the dish was made (`reference_scores.npy`). Ordinary computer
  rounding stays around 1e-5 or below; anything larger means the edit changed something.
- **Closest decision:** the smallest gap, at any output position, between the winning token and the runner-up.
  When it reaches 0 a token flips and the output changes.

A copy keeps the original's `reference_scores.npy`, so the meter always compares against the dish it came from.

## 5. What is in a dish

| File | What it is |
|---|---|
| `run.py` | the whole model in about 60 lines of numpy, plus the change meter |
| `model.npz` | the kept weights, as 16-bit floats (32-bit if 16 did not reproduce the output) |
| `dish.json` | the input and output, the token list, the model settings, what was kept per layer, sizes, checks |
| `reference_scores.npy` | the scores the meter compares against |

`model.npz`, with `K` the number of detectors kept in that layer (`dish.json` → `ffn_units_kept_per_layer`):

| Key | Shape | Role |
|---|---|---|
| `emb` | (tokens, 576) | token table: one row per token this dish knows, in the order of `dish.json` → `vocab`; also used at the end to score tokens |
| `out_norm` | (576) | the final multipliers, before the token scores |
| `<l>.an` | (576) | layer `l`'s multipliers before attention (`l` = 0 … 29) |
| `<l>.q` | (576, 576) | question weights: 9 heads × 64 rows |
| `<l>.k`, `<l>.v` | (192, 576) | label and content weights: 3 shared heads × 64 rows |
| `<l>.o` | (576, 576) | mixes the 9 head outputs back into the stream |
| `<l>.fn` | (576) | multipliers before the detectors |
| `<l>.gate`, `<l>.up` | (K, 576) | the kept detectors' two input rows |
| `<l>.down` | (576, K) | the kept detectors' outputs, one column per detector |

Every matrix is (out, in): each row produces one output number from all its inputs, `y = x @ W.T`.

## 6. Change it without changing the output

Edit a copy, then run it:

```sh
cp -r petri/dishes/code petri/dishes/code-test
env/result/bin/python3 - <<'EOF'
import numpy as np
p = 'petri/dishes/code-test/model.npz'
W = dict(np.load(p))
W['3.up'][0] *= 2       # layer 4, first kept detector: double its "up" input row ...
W['3.down'][:, 0] /= 2  # ... and halve its output column: the detector's output is unchanged
np.savez(p, **W)
EOF
env/result/bin/python3 petri/dishes/code-test/run.py
```

`permute.py` does a bigger change of this kind automatically. It reorders the 576 stream positions everywhere at
once: the token table, all multipliers, and the stream side of every weight matrix.

```sh
env/result/bin/python3 petri/permute.py code sorted     # positions ordered by average rank -> dishes/code-sorted
env/result/bin/python3 petri/permute.py code random     # a random order, as a control -> dishes/code-random
```

Measured on `code`: both copies keep the output, the scores move by about 2e-6 (rounding), and the sorted order
makes the 24 token rows about a third sorted (+0.32, where 1 would be fully sorted).

Edits tried on copies of the `code` dish (measured 2026-11-13, weights saved as 16-bit like the dish):

| Edit | Output | Largest score change |
|---|---|---|
| Reorder the 576 positions (`permute.py`, sorted or random) | unchanged | 0.000002 (rounding) |
| A detector's `up` row × 2 and its `down` column ÷ 2 | unchanged | 0 (exact) |
| Reorder all kept detectors in one layer (rows of `gate`/`up`, columns of `down`, the same way) | unchanged | 0.0000000000001 |
| A detector's `gate` row × 2 | unchanged | 0.29: a real change inside; the gate passes through a curve (silu), so it does not scale along |
| A layer's multipliers (`an`) × 1.01 | unchanged | 0.075: a real change inside |
| A layer's multipliers (`an`) × 2 | **changed** | 14.1: the multipliers come after the volume knob, so the heads see a louder stream |
| Round every weight matrix to 8 bits (one scale per row) | **changed** | 1.03: only ~0.8% error per weight, but the dish has almost no margin |
| Remove one more detector (the first kept one in layer 21) | unchanged | 0.06 |

The first three are the "change the model, not its output" kind: the scores do not move beyond rounding. The
middle ones change the model's insides while the output happens to survive; the meter shows by how much. The
last row shows the dish is on the edge for the *next detector in rank order*, not for every detector: single
detectors can often still go, so the ranking left some room.

## 7. Limits

- A dish knows one input text. It cannot answer anything else, by design.
- The cut is checked token by token on the recorded output ("teacher-forced"). That guarantees the same
  output from the same input; it says nothing about how the cut model would behave on other text.
- Attention heads are not cut; only detectors and token rows are.
- Dishes are not in git: `model.npz` is about 185 MB, over GitHub's 100 MB file limit. Make them with `mini.py`.
