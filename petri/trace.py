"""Step-by-step trace of one short prompt through SmolLM2, for the petri viewer.

The same computation as model_torch.TorchModel.forward, but attention is computed explicitly so its
weights can be recorded (SDPA does not return them). trace() checks its own final hidden state against
TorchModel.forward on every call and refuses to return a trace that disagrees.

Docking is the definition from profile_ffn.py: the fewest clustered blocks of 32 FFN units whose summed
contribution |H|*||W_down col|| reaches 90% at this token. The block clustering comes from
data/results/blocks.json when the profile run has made it; until then from petri/blocks_provisional.json
(fewer documents, no chats), and the trace says which one it used.
"""
import json
import math
import os
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from profile_ffn import BLOB, BLOCK, FRAC, RES, docking  # noqa: E402

FINAL_BLOCKS = f'{RES}/blocks.json'
PROVISIONAL_BLOCKS = os.path.join(HERE, 'blocks_provisional.json')
TOKENIZER = 'sources/hf/SmolLM2-135M-Instruct-tokenizer.json'


def load_blocks():
    for path, kind in [(FINAL_BLOCKS, 'final'), (PROVISIONAL_BLOCKS, 'provisional')]:
        if os.path.exists(path):
            b = json.load(open(path))
            return b['perm'], {'kind': kind, 'path': os.path.relpath(path, ROOT),
                               'docs_per_domain': b['docs_per_domain'], 'tokens': b['tokens']}
    raise SystemExit('no block clustering yet: run petri/trace.py provisional-blocks first')


@torch.no_grad()
def trace(tm, ids, perm):
    dev, T, L, D, NH, NKV, HD, G = tm.device, len(ids), tm.L, tm.D, tm.NH, tm.NKV, tm.HD, tm.G
    nb = tm.F // BLOCK
    ids_t = torch.as_tensor(ids, device=dev, dtype=torch.long)
    future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=dev), 1)
    X = tm.emb[ids_t].clone()
    keys = ['resid', 'q', 'k', 'v', 'att', 'ho', 'attn_delta', 'ffn_delta', 'bshare', 'bdock', 'bunits',
            'k_blocks', 'k_units', 'err_blocks', 'err_units']
    r = {k: [] for k in keys}
    for l, W in enumerate(tm.layers):
        r['resid'].append(X.clone())
        n = tm.rms(X, W['an'])
        q = tm.rope(n @ W['q'].T, NH)                       # (T, 9, 64)
        k = tm.rope(n @ W['k'].T, NKV)                      # (T, 3, 64)
        v = (n @ W['v'].T).view(T, NKV, HD)
        kk, vv = k.repeat_interleave(G, 1), v.repeat_interleave(G, 1)   # head h reads kv head h // 3
        s = torch.einsum('qhd,khd->hqk', q, kk) / math.sqrt(HD)
        a = torch.softmax(s.masked_fill(future, float('-inf')), -1)     # (9, T, T)
        ho = torch.einsum('hqk,khd->qhd', a, vv)            # (T, 9, 64)
        ad = ho.reshape(T, D) @ W['o'].T
        X = X + ad
        n2 = tm.rms(X, W['fn'])
        H = F.silu(n2 @ W['gate'].T) * (n2 @ W['up'].T)     # (T, 1536)
        y = H @ W['down'].T
        X = X + y
        c = H.abs() * tm.down_norm[l]
        p = torch.as_tensor(perm[l], device=dev)
        cb = c[:, p].view(T, nb, BLOCK).sum(-1)             # clustered blocks, in perm order
        kb, mb, _ = docking(cb)
        ku, mu, _ = docking(c)
        unit_mask = torch.zeros_like(mu)
        unit_mask[:, p] = mb.repeat_interleave(BLOCK, 1)
        ny = y.norm(dim=-1).clamp_min(1e-30)
        r['q'].append(q); r['k'].append(k); r['v'].append(v); r['att'].append(a); r['ho'].append(ho)
        r['attn_delta'].append(ad); r['ffn_delta'].append(y)
        r['bshare'].append(cb / cb.sum(-1, keepdim=True).clamp_min(1e-30))
        r['bdock'].append(mb)
        r['bunits'].append(mu[:, p].view(T, nb, BLOCK).sum(-1))
        r['k_blocks'].append(kb); r['k_units'].append(ku)
        r['err_blocks'].append((y - (H * unit_mask) @ W['down'].T).norm(dim=-1) / ny)
        r['err_units'].append((y - (H * mu) @ W['down'].T).norm(dim=-1) / ny)
    r['resid'].append(X.clone())
    ref = tm.forward(ids)
    diff = ((X - ref).norm() / ref.norm()).item()
    if diff > 1e-4:
        raise RuntimeError(f'trace disagrees with TorchModel.forward: relative diff {diff:.2e}')
    # Logit lens: what the model would predict if it stopped after each layer.
    lens = torch.softmax(tm.final_hidden(torch.stack(r['resid'][1:])) @ tm.emb.T, -1)   # (L, T, vocab)
    lp, li = lens.topk(3, -1)
    out = {k: torch.stack(v).cpu().numpy() for k, v in r.items()}
    out['lens_p'], out['lens_ids'] = lp.cpu().numpy(), li.cpu().numpy()
    out['check_rel_diff'] = diff
    return out


def q8(a, axis_keep):
    """int8 with one float32 scale per vector over the trailing axes beyond axis_keep."""
    lead = a.shape[:axis_keep]
    flat = a.reshape(*lead, -1)
    s = np.abs(flat).max(-1) / 127.0
    s[s == 0] = 1.0
    return np.round(flat / s[..., None]).astype(np.int8).reshape(a.shape), s.astype(np.float32)


def pack(tr, tokens, meta):
    """Binary trace: 'PTRI', u32 header length, JSON header, then 4-byte aligned arrays the header indexes."""
    arrays = {}
    # The stream, both sublayer outputs and the keys carry outlier dimensions (one value up to ~1400x the
    # vector median), which per-vector int8 crushes (p95 error 18% on the stream). They go as float16
    # (~3 significant digits at every size). q, v and head outputs stay int8: p95 error under 5%.
    for name in ['resid', 'attn_delta', 'ffn_delta', 'k']:
        a = tr[name]
        if np.abs(a).max() >= 65504:
            raise ValueError(f'{name} exceeds the float16 range')
        arrays[name] = a.astype(np.float16)
    for name in ['q', 'v', 'ho']:
        arrays[name], arrays[name + '_scale'] = q8(tr[name], 3)
    arrays['att'] = np.round(tr['att'] * 255).astype(np.uint8)
    arrays['bshare'] = np.round(tr['bshare'] * 255).astype(np.uint8)
    arrays['bdock'] = tr['bdock'].astype(np.uint8)
    arrays['bunits'] = tr['bunits'].astype(np.uint8)
    arrays['k_blocks'] = tr['k_blocks'].astype(np.uint8)
    arrays['k_units'] = tr['k_units'].astype(np.uint16)
    for name in ['err_blocks', 'err_units', 'lens_p']:
        arrays[name] = tr[name].astype(np.float32)
    arrays['lens_ids'] = tr['lens_ids'].astype(np.int32)
    header = dict(meta, tokens=tokens, sections={}, check_rel_diff=tr['check_rel_diff'])
    return pack_arrays(arrays, header)


def pack_arrays(arrays, header):
    """'PTRI', u32 header length, JSON header (sections filled in here), then 4-byte aligned arrays."""
    blobs, off = [], 0
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        header['sections'][name] = {'dtype': a.dtype.name, 'shape': list(a.shape), 'offset': off}
        b = a.tobytes()
        b += b'\0' * (-len(b) % 4)
        blobs.append(b)
        off += len(b)
    h = json.dumps(header, ensure_ascii=False).encode()
    h += b' ' * (-(len(h) + 8) % 4)
    return b'PTRI' + struct.pack('<I', len(h)) + h + b''.join(blobs)


class Tracer:
    def __init__(self):
        from tokenizers import Tokenizer
        from model_torch import TorchModel
        self.tm = TorchModel(BLOB)
        self.tok = Tokenizer.from_file(TOKENIZER)
        self.perm, self.blocks_meta = load_blocks()

    def run(self, text, name='custom', domain='custom', source='typed in the live viewer'):
        ids = self.tok.encode(text, add_special_tokens=False).ids
        if not 2 <= len(ids) <= 96:
            raise ValueError(f'prompt is {len(ids)} tokens; the viewer takes 2 to 96')
        tr = trace(self.tm, ids, self.perm)
        lens_vocab = sorted(set(tr['lens_ids'].ravel().tolist()))
        meta = {'format': 1, 'name': name, 'domain': domain, 'text': text, 'source': source, 'ids': ids,
                'model': self.model_meta(),
                'blocks': self.blocks_meta,
                'lens_vocab': {str(i): self.tok.decode([i]) for i in lens_vocab}}
        return pack(tr, [self.tok.decode([i]) for i in ids], meta)

    def model_meta(self):
        from model import GGUF
        tm = self.tm
        return {'name': 'SmolLM2-135M-Instruct (Ollama smollm2:135m)', 'gguf_sha256': tm.gguf_sha256,
                'L': tm.L, 'D': tm.D, 'NH': tm.NH, 'NKV': tm.NKV, 'HD': tm.HD, 'F': tm.F, 'block': BLOCK,
                'frac': FRAC, 'vocab': len(tm.vocab), 'eps': float(tm.eps),
                'rope_base': float(GGUF(BLOB).kv['llama.rope.freq_base'])}

    def weights(self):
        """Weights for the viewer's 'inside a head' demo, shared by every trace.

        Per layer: the attention RMS-norm weights (576), and per query head the two W_q rows of one
        rotation pair, the pair whose rows have the largest norm. That is enough to redo two of the
        64 values of every head by hand (rescale, 576 multiplies, sum, rotate) in the browser.
        Stored as int16 with one float32 scale per row: 1/32767 resolution, about 0.6 MB.
        """
        tm, HD = self.tm, self.tm.HD
        an, rows, pairs = [], [], []
        for W in tm.np_model.layers:
            an.append(W['an'])
            lr, lp = [], []
            for h in range(tm.NH):
                wq = W['q'][h * HD:(h + 1) * HD]                    # (64, 576), rows in rotation-pair order
                norms = np.linalg.norm(wq.reshape(HD // 2, 2, -1), axis=(1, 2))
                p = int(np.argmax(norms))
                lp.append(p)
                lr.append(wq[2 * p:2 * p + 2])
            rows.append(lr)
            pairs.append(lp)
        an, rows = np.array(an, np.float32), np.array(rows, np.float32)    # (L, 576), (L, 9, 2, 576)
        arrays = {}
        for name, a in [('an', an), ('wq', rows)]:
            s = np.abs(a).max(-1) / 32767.0
            s[s == 0] = 1.0
            arrays[name] = np.round(a / s[..., None]).astype(np.int16)
            arrays[name + '_scale'] = s.astype(np.float32)
        arrays['pair'] = np.array(pairs, np.uint8)
        header = {'format': 1, 'kind': 'weights', 'model': self.model_meta(), 'sections': {},
                  'what': 'attn_norm weights per layer; per query head the two W_q rows of the largest-norm rotation pair'}
        return pack_arrays(arrays, header)


if __name__ == '__main__':
    if sys.argv[1:] == ['provisional-blocks']:
        from model_torch import TorchModel
        from profile_ffn import cmd_blocks
        cmd_blocks(TorchModel(BLOB), domains=['code', 'prose', 'maths', 'troubleshooting'], max_docs=5,
                   out=PROVISIONAL_BLOCKS)
    else:
        t = Tracer()
        b = t.run(sys.argv[1] if len(sys.argv) > 1 else 'The capital of France is')
        print(len(b), 'bytes')
