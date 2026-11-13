"""Reorder the 576 stream positions of a petri dish everywhere at once: change the model, not its output.

  env/result/bin/python3 petri/permute.py <dish> [sorted|random] [new dish name]

Copies petri/dishes/<dish> to a new dish (default <dish>-<mode>) and moves position perm[j] to position j in
everything that touches the stream: the token table's columns, the three multiplier sets, the stream side
of every weight matrix (the columns of q, k, v, gate and up, which read the stream; the rows of o and down,
which write to it). Head dimensions and detector order are untouched. Evening out averages over all 576
positions, so the order does not matter to it, and every sum over positions is the same sum in another
order. The output therefore cannot change; only float rounding can, and the change meter shows how much.
The copy keeps the original's reference_scores.npy, so the meter compares against the original dish.

  sorted: Michael's idea, positions ordered by their average rank across this dish's token rows, so the
          token rows come out as sorted as one shared order allows.
  random: a seeded random order, as a control.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    src = os.path.join(HERE, 'dishes', sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else 'sorted'
    dst = os.path.join(HERE, 'dishes', sys.argv[3] if len(sys.argv) > 3 else f'{sys.argv[1]}-{mode}')
    W = dict(np.load(os.path.join(src, 'model.npz')))
    M = json.load(open(os.path.join(src, 'dish.json')))
    L, D = M['model']['L'], M['model']['D']
    emb = W['emb'].astype(np.float64)
    if mode == 'sorted':
        ranks = np.argsort(np.argsort(emb, 1), 1)
        perm = np.argsort(ranks.mean(0), kind='stable')
    elif mode == 'random':
        perm = np.random.default_rng(20261113).permutation(D)
    else:
        raise SystemExit('mode is sorted or random')

    W['emb'] = W['emb'][:, perm]
    W['out_norm'] = W['out_norm'][perm]
    for l in range(L):
        p = f'{l}.'
        for vec in ('an', 'fn'):
            W[p + vec] = W[p + vec][perm]
        for reads in ('q', 'k', 'v', 'gate', 'up'):
            W[p + reads] = W[p + reads][:, perm]
        for writes in ('o', 'down'):
            W[p + writes] = W[p + writes][perm, :]

    shutil.copytree(src, dst, dirs_exist_ok=True)
    np.savez(os.path.join(dst, 'model.npz'), **W)
    M['name'] = os.path.basename(dst)
    M['permuted'] = {'from': sys.argv[1], 'mode': mode, 'perm': perm.tolist(),
                     'how': 'new position j holds old position perm[j], in everything that touches the stream'}
    json.dump(M, open(os.path.join(dst, 'dish.json'), 'w'), indent=1, ensure_ascii=False)

    def sortedness(e):                                   # 1 = every row fully sorted, 0 = random
        r = np.argsort(np.argsort(e, 1), 1).astype(np.float64)
        r -= r.mean(1, keepdims=True)
        x = np.arange(e.shape[1]) - (e.shape[1] - 1) / 2
        return float(((r * x).sum(1) / np.sqrt((r * r).sum(1) * (x * x).sum())).mean())
    print(f'{os.path.basename(dst)}: token rows sortedness {sortedness(emb):+.3f} -> {sortedness(W["emb"].astype(np.float64)):+.3f}')
    r = subprocess.run([sys.executable, os.path.join(dst, 'run.py')], capture_output=True, text=True)
    print('output:', repr(r.stdout.strip()))
    print(r.stderr.strip())


if __name__ == '__main__':
    main()
