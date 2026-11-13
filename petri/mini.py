"""Petri-dish mini model: SmolLM2 cut down to what one input string needs to produce its exact output.

  env/result/bin/python3 petri/mini.py [name] ["input text"] [max new tokens]

With only a name from petri/prompts.json, that sample's text is used. Writes petri/dishes/<name>/:
model.npz (the kept weights), dish.json (what was kept, the sizes, the checks) and run.py (standalone,
numpy only). The steps:
  1. greedy-generate the output with the full, verified model;
  2. keep only the tokens the input and output use: the full model's pick is the top token over the
     whole vocabulary, so it is also the top token over this subset;
  3. rank every FFN unit (30 x 1536) by its largest contribution |activation| x ||W_down column|| on
     this exact text, and binary-search the fewest units, taken globally in that order, that keep
     every output token identical (checked teacher-forced: one pass over input + output);
  4. save, run the saved dish with its own run.py, and require the same output.
Attention heads are kept whole.
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace import HERE, ROOT, TOKENIZER  # noqa: E402  (petri/trace.py; also chdirs to the docking root)
from model import GGUF  # noqa: E402
from model_torch import TorchModel  # noqa: E402
from profile_ffn import BLOB  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

STOP = {0, 2}   # <|endoftext|>, <|im_end|>


@torch.no_grad()
def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'code'
    prompts = {p['name']: p['text'] for p in json.load(open(os.path.join(HERE, 'prompts.json')))['prompts']}
    text = sys.argv[2] if len(sys.argv) > 2 else prompts[name]
    max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    tm, tok = TorchModel(BLOB), Tokenizer.from_file(TOKENIZER)
    L, F, D = tm.L, tm.F, tm.D
    ids = tok.encode(text, add_special_tokens=False).ids
    P = len(ids)

    # 1. the full model's own output
    seq = list(ids)
    for _ in range(max_new):
        nxt = int(torch.argmax(tm.final_hidden(tm.forward(seq))[-1] @ tm.emb.T))
        seq.append(nxt)
        if nxt in STOP:
            break
    out = seq[P:]
    print(f'input {P} tokens -> output {len(out)} tokens: {tok.decode(out)!r}', flush=True)

    # 2. the dish's vocabulary
    vocab = sorted(set(seq))
    V = torch.as_tensor(vocab, device=tm.device)
    target = torch.as_tensor(out, device=tm.device)

    def check(keep):
        """Teacher-forced: does every output position still pick the recorded token? And by what margin?"""
        logits = tm.final_hidden(tm.forward(seq[:-1], keep=keep))[P - 1:] @ tm.emb[V].T
        top2 = logits.topk(min(2, len(vocab)), -1).values
        margin = float((top2[:, 0] - top2[:, -1]).min())
        return bool((V[logits.argmax(-1)] == target).all()), margin

    full_ok, full_margin = check(None)
    if not full_ok:
        raise SystemExit('the uncut model does not reproduce its own output teacher-forced')

    # 3. rank units by their largest contribution on this text, then cut as many as possible
    scores = torch.zeros(L, F)

    def hook(l, H):
        scores[l] = (H.abs().max(0).values * tm.down_norm[l]).float().cpu()
    tm.forward(seq[:-1], on_ffn=hook)
    order = torch.argsort(scores.flatten(), descending=True)

    def keep_for(k):
        m = torch.zeros(L * F, dtype=torch.bool)
        m[order[:k]] = True
        return [torch.where(m.view(L, F)[l])[0].tolist() for l in range(L)]

    t0 = time.time()
    lo, hi = 0, L * F                       # check(keep_for(hi)) passes (that is the uncut model)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if check(keep_for(mid))[0]:
            hi = mid
        else:
            lo = mid
    keep = keep_for(hi)
    ok, margin = check(keep)
    if not ok:
        raise SystemExit('binary search ended on a failing cut (the pass/fail boundary is not monotonic here)')
    kept = sum(len(k) for k in keep)
    print(f'kept {kept} of {L * F} FFN units ({kept / (L * F) * 100:.1f}%) in {time.time() - t0:.0f}s; '
          f'smallest winning margin {margin:.2f} (uncut {full_margin:.2f})', flush=True)

    # 4. save, then prove the saved dish gives the same output
    m = tm.np_model
    dish = os.path.join(HERE, 'dishes', name)
    os.makedirs(dish, exist_ok=True)
    attn = L * (2 * D * D + 2 * D * tm.NKV * tm.HD + 2 * D)
    params = {'token_table': len(vocab) * D, 'attention_and_multipliers': attn, 'ffn': kept * 3 * D, 'final_multipliers': D}
    full_params = {'token_table': m.emb.shape[0] * D, 'attention_and_multipliers': attn, 'ffn': L * F * 3 * D, 'final_multipliers': D}
    for dtype in (np.float16, np.float32):
        arrays = {'emb': m.emb[vocab], 'out_norm': m.out_norm}
        for l, W in enumerate(m.layers):
            k = keep[l]
            arrays.update({f'{l}.an': W['an'], f'{l}.q': W['q'], f'{l}.k': W['k'], f'{l}.v': W['v'], f'{l}.o': W['o'],
                           f'{l}.fn': W['fn'], f'{l}.gate': W['gate'][k], f'{l}.up': W['up'][k], f'{l}.down': W['down'][:, k]})
        np.savez(os.path.join(dish, 'model.npz'), **{a: v.astype(dtype) for a, v in arrays.items()})
        meta = {
            'name': name, 'made': time.strftime('%Y-%m-%d %H:%M:%S'),
            'input': text, 'input_ids': ids, 'output_ids': out, 'output': tok.decode(out),
            'vocab': vocab, 'vocab_strings': [tok.decode([i]) for i in vocab],
            'model': {'source': 'SmolLM2-135M-Instruct (Ollama smollm2:135m)', 'gguf_sha256': tm.gguf_sha256,
                      'L': L, 'D': D, 'NH': tm.NH, 'NKV': tm.NKV, 'HD': tm.HD, 'eps': float(tm.eps),
                      'rope_base': float(GGUF(BLOB).kv['llama.rope.freq_base'])},
            'ffn_units_kept_per_layer': [len(k) for k in keep],
            'ffn_units_kept': kept, 'ffn_units_full': L * F,
            'params': params, 'params_total': sum(params.values()),
            'params_full': full_params, 'params_full_total': sum(full_params.values()),
            'dtype': np.dtype(dtype).name,
            'checks': {'smallest_winning_margin': margin, 'uncut_margin': full_margin,
                       'how': 'teacher-forced: every output position picks the recorded token over the dish vocabulary'},
            'how_cut': __doc__.split('The steps:')[1].strip(),
        }
        json.dump(meta, open(os.path.join(dish, 'dish.json'), 'w'), indent=1, ensure_ascii=False)
        shutil.copy(os.path.join(HERE, 'dish_run.py'), os.path.join(dish, 'run.py'))
        r = subprocess.run([sys.executable, os.path.join(dish, 'run.py')], capture_output=True, text=True)
        if r.returncode == 0:
            # the change meter's baseline: this dish's own scores, as saved
            subprocess.run([sys.executable, os.path.join(dish, 'run.py'), '--record'], check=True)
            break
        print(f'{np.dtype(dtype).name} weights did not reproduce the output ({r.stderr.strip()}); trying float32', flush=True)
    else:
        raise SystemExit('the saved dish does not reproduce the output')
    size = os.path.getsize(os.path.join(dish, 'model.npz'))
    print(f'saved {dish}: {meta["params_total"]:,} of {meta["params_full_total"]:,} numbers '
          f'({meta["params_total"] / meta["params_full_total"] * 100:.1f}%), model.npz {size / 1e6:.1f} MB as {meta["dtype"]}')
    print(f'run.py says: {r.stdout.strip()!r}')


if __name__ == '__main__':
    main()
