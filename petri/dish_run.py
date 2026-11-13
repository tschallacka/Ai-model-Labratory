"""Run this petri-dish model. Needs only Python 3 and numpy.

  python3 run.py                 # runs the one input this dish knows (dish.json "input")
  python3 run.py "some text"     # refused unless it is exactly that input

After the output it prints a change meter: how far the scores moved from the ones recorded when the dish
was made (reference_scores.npy), and how close the tightest decision is. Use it to change the weights
without changing the output: a harmless edit moves the scores by ~1e-5 (float rounding) or less.

The dish is SmolLM2-135M cut down for one input string: only the tokens that input and its output use,
and only the FFN units that the output needs. Any other input would need tokens and units that were
removed, so it is refused. Edit model.npz and run again to experiment; the last line says whether the
output still matches the one recorded when the dish was made.
"""
import json
import os
import sys

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(os.path.join(D, 'dish.json')))
W = {k: v.astype(np.float32) for k, v in np.load(os.path.join(D, 'model.npz')).items()}
RECORD = sys.argv[1:] == ['--record']                     # used once, by mini.py, to store the reference scores
if len(sys.argv) > 1 and not RECORD and sys.argv[1] != M['input']:
    sys.exit('this dish only knows its own input: ' + repr(M['input']))

d = M['model']
L, NH, NKV, HD, EPS = d['L'], d['NH'], d['NKV'], d['HD'], d['eps']
G = NH // NKV                                             # query heads per shared key/value head
VOCAB = M['vocab']
ROW = {t: i for i, t in enumerate(VOCAB)}                 # token id -> row in this dish's token table
INV = d['rope_base'] ** (-np.arange(0, HD, 2) / HD)       # turning speed of each pair


def rms(x, w):                                            # evening out: volume knob, then the multipliers
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def rope(x, heads):                                       # turn each adjacent pair by position x speed
    T = x.shape[0]
    ang = np.arange(T)[:, None] * INV[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    x = x.reshape(T, heads, HD // 2, 2)
    x0, x1 = x[..., 0], x[..., 1]
    return np.stack([x0 * c - x1 * s, x0 * s + x1 * c], -1).reshape(T, heads, HD)


def forward(ids):
    X = W['emb'][[ROW[t] for t in ids]]                   # text -> numbers: one row per token
    T = len(ids)
    future = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
    for l in range(L):
        p = f'{l}.'
        n = rms(X, W[p + 'an'])
        q = rope(n @ W[p + 'q'].T, NH)
        k = np.repeat(rope(n @ W[p + 'k'].T, NKV), G, 1)
        v = np.repeat((n @ W[p + 'v'].T).reshape(T, NKV, HD), G, 1)
        s = np.einsum('qhd,khd->hqk', q, k) / np.sqrt(HD) + future
        a = np.exp(s - s.max(-1, keepdims=True))
        a /= a.sum(-1, keepdims=True)                      # attention weights
        X = X + np.einsum('hqk,khd->qhd', a, v).reshape(T, NH * HD) @ W[p + 'o'].T
        n = rms(X, W[p + 'fn'])
        g = n @ W[p + 'gate'].T
        H = g / (1 + np.exp(-g)) * (n @ W[p + 'up'].T)     # the kept detectors
        X = X + H @ W[p + 'down'].T
    return rms(X, W['out_norm']) @ W['emb'].T             # a score for every token this dish knows


def scores():
    """The dish's scores at every output position, given the recorded output (one pass)."""
    P, out = len(M['input_ids']), M['output_ids']
    return forward(M['input_ids'] + out[:-1])[P - 1:]


REF = os.path.join(D, 'reference_scores.npy')
if RECORD:
    np.save(REF, scores())
    sys.exit(0)

ids, out = list(M['input_ids']), []
for _ in range(len(M['output_ids'])):
    nxt = VOCAB[int(forward(ids + out)[-1].argmax())]
    out.append(nxt)
    if nxt in (0, 2):                                     # <|endoftext|>, <|im_end|>
        break
text = ''.join(M['vocab_strings'][ROW[t]] for t in out)
print(text)
# The change meter: how far the scores moved from the ones recorded when the dish was made, and how close
# the tightest decision is. A truly harmless edit moves nothing (up to float rounding, ~1e-5).
s = scores()
top2 = np.sort(s, -1)[:, -2:]
margin = float((top2[:, 1] - top2[:, 0]).min())
change = float(np.abs(s - np.load(REF)).max()) if os.path.exists(REF) else None
meter = (f'largest score change vs the original: {change:.2e} · ' if change is not None else '') + f'closest decision wins by {margin:.4f}'
print(meter, file=sys.stderr)
unchanged = out == M['output_ids']
print('output unchanged' if unchanged else 'OUTPUT CHANGED: differs from the one recorded when this dish was made', file=sys.stderr)

# The run log (see petri/runlog.py): one JSON line per run, its line number is its id. Written to the lab's
# notes/runs.jsonl when this dish still sits in the lab, otherwise to runs.jsonl inside the dish.
if not os.environ.get('PETRI_NO_LOG'):
    import hashlib
    import time
    lab = os.path.abspath(os.path.join(D, '..', '..', '..', 'notes', 'runs.jsonl'))
    log = lab if os.path.isdir(os.path.dirname(lab)) else os.path.join(D, 'runs.jsonl')
    h = hashlib.sha256()
    with open(os.path.join(D, 'model.npz'), 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    entry = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'script': 'run.py', 'argv': sys.argv[1:],
             'dish': M['name'], 'dish_path': os.path.relpath(D, os.path.dirname(os.path.dirname(lab))) if log == lab else os.path.basename(D),
             'model_npz_sha256': h.hexdigest()[:16],
             'output': text, 'unchanged': unchanged, 'largest_score_change': change, 'closest_margin': margin}
    with open(log, 'a') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    with open(log) as f:
        n = sum(1 for _ in f)
    print(f'\n[run log] written: {log} line {n} (run id {n}).')
    print('[run log] REMINDER: is this worth a lab-book entry in notes/LABBOOK.md (question, method, numbers, '
          'conclusion)? Is the log line worth keeping, whole or in part?')
    print(f'[run log]   keep: petri/runlog.py keep {n} "why" [part ...]   ·   drop: petri/runlog.py drop {n} "why"')
if not unchanged:
    sys.exit(1)
