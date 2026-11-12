"""Measurement suite for binary-docking-gates on SmolLM2 (brainstorm Q11, Q14, Q16, Q21-Q27, Q34).

Definitions used throughout:
  contribution  c[t,u] = |H[t,u]| * ||W_down[:,u]||, unit u's share of the FFN output at token t
                (H = silu(gate x) * up x). This is the summed-norm proxy of finding 5.
  docking set   the fewest units (or blocks of 32) whose contributions sum to >= 90% of the total at
                that token: an oracle set, computable only after gate and up (finding 1).
  rel_err       ||y - y_dock|| / ||y||, the true FFN output vector y against the output of the
                docking set alone: how well the proxy set reproduces the real output vector.
  buckets       token position (prefix length - 1) in [0,16) [16,128) [128,512) [512,2048)
                [2048,4096) [4096,8192): the context-length dimension of Q25.

Commands:
  blocks   co-docking over the 'blocks' split of every domain -> 48 clustered blocks of 32 per layer
  profile  every document of every domain -> per (domain, split) aggregates + per-document docking counts
"""
import json
import os
import sys
import time

import numpy as np
import torch

from model_torch import TorchModel

BLOB = os.path.expanduser('~/.ollama/models/blobs/sha256-f535f83ec568d040f88ddc04a199fa6da90923bbb41d4dcaed02caa924d6ef57')
DOMAINS = ['code', 'prose', 'maths', 'troubleshooting', 'chat']
EDGES = [0, 16, 128, 512, 2048, 4096, 8192]
NB = len(EDGES) - 1
BLOCK, FRAC = 32, 0.9
RES = os.environ.get('DOCKING_RES', 'data/results')   # override only for smoke tests


def load_domain(name):
    """[(doc id, split, int32 token ids)] for one domain."""
    if name != 'chat':
        ids = np.load(f'data/docs/{name}.npy')
        recs = [json.loads(l) for l in open(f'data/docs/{name}.jsonl')]
        return [(r['doc'], r['split'], ids[i]) for i, r in enumerate(recs)]
    from tokenizers import Tokenizer
    import hashlib
    tok = Tokenizer.from_file('sources/hf/SmolLM2-135M-Instruct-tokenizer.json')
    out = []
    d = 'data/chats/transcripts'
    for f in sorted(os.listdir(d)):
        if not f.endswith('.json'):
            continue
        t = json.load(open(os.path.join(d, f)))
        if t.get('declined'):
            continue
        sp = t['smollm2']['system_prompt']
        s = f'<|im_start|>system\n{sp}<|im_end|>\n' if sp else ''
        for m in t['messages']:
            s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        ids = tok.encode(s, add_special_tokens=False).ids
        if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != t['final_token_sha256']:
            raise SystemExit(f'{f}: token ids do not match the hash recorded at generation')
        out.append((t['chat'], t['split'], np.array(ids, dtype=np.int32)))
    return out


def bucket_index(T, device):
    b = torch.zeros(T, dtype=torch.long, device=device)
    for i, e in enumerate(EDGES[1:-1]):
        b[e:] = i + 1
    return b


def docking(c):
    """c (T, n) contributions -> (k, mask, captured): docking-set size, membership, cumulative share."""
    tot = c.sum(-1, keepdim=True).clamp_min(1e-30)
    s, _ = torch.sort(c, -1, descending=True)
    cap = torch.cumsum(s, -1) / tot
    k = (cap < FRAC).sum(-1) + 1
    k = k.clamp(max=c.shape[1])
    thr = s.gather(1, (k - 1)[:, None])
    return k, c >= thr, cap


def cmd_blocks(tm, domains=None, max_docs=None, out=None):
    """Greedy balanced clustering of units by co-docking on the 'blocks' split of all domains.
    domains/max_docs/out exist for the petri viewer's provisional blocks; the recorded run uses the defaults."""
    F, L, dev = tm.F, tm.L, tm.device
    co = [torch.zeros(F, F, device=dev) for _ in range(L)]
    n_tok = 0
    used = {}
    for dom in (domains or DOMAINS):
        docs = [d for d in load_domain(dom) if d[1] == 'blocks'][:max_docs]
        used[dom] = len(docs)
        for _, _, ids in docs:
            def hook(l, H):
                c = H.abs() * tm.down_norm[l]
                _, mask, _ = docking(c)
                m = mask.float()
                co[l] += m.T @ m
            tm.forward(ids, on_ffn=hook)
            n_tok += len(ids)
        print(f'blocks: {dom} {used[dom]} docs', flush=True)
    perms = []
    for l in range(L):
        C = co[l].cpu().double().numpy()
        freq = np.diag(C).copy()
        sim = C / np.sqrt(np.outer(freq, freq).clip(min=1))
        unassigned = np.ones(F, bool)
        perm = []
        for u in np.argsort(-freq):
            if not unassigned[u]:
                continue
            cand = np.where(unassigned)[0]
            cand = cand[np.argsort(-sim[u, cand])][:BLOCK]   # u itself has similarity 1 and comes first
            perm.extend(cand.tolist())
            unassigned[cand] = False
        perms.append(perm)
    out = out or f'{RES}/blocks.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({'method': 'greedy balanced: seed = most-docking unassigned unit, block = it plus the 31 '
                         'unassigned units with the highest co-docking cosine to it',
               'docs_per_domain': used, 'tokens': n_tok, 'block': BLOCK, 'perm': perms},
              open(out, 'w'))
    print('wrote', out)


def cmd_profile(tm, only=None):
    F, L, dev = tm.F, tm.L, tm.device
    nb_c = F // BLOCK
    perms = [torch.tensor(p, device=dev) for p in json.load(open(f'{RES}/blocks.json'))['perm']]
    os.makedirs(RES, exist_ok=True)
    for dom in (only or DOMAINS):
        docs = load_domain(dom)
        n = len(docs)
        doc_unit = np.lib.format.open_memmap(f'{RES}/dock_units_{dom}.npy', 'w+', np.uint16, (n, L, NB, F))
        doc_blk = np.lib.format.open_memmap(f'{RES}/dock_blocks_{dom}.npy', 'w+', np.uint16, (n, 2, L, NB, nb_c))
        doc_tok = np.zeros((n, NB), np.int32)
        agg = {}
        chat_k = {}
        t0 = time.time()
        for i, (doc, split, ids) in enumerate(docs):
            T = len(ids)
            bidx = bucket_index(T, dev)
            a = agg.setdefault(split, {
                'tokens': torch.zeros(NB, device=dev),
                'curve_units': torch.zeros(L, NB, F, device=dev), 'curve_contig': torch.zeros(L, NB, nb_c, device=dev),
                'curve_clust': torch.zeros(L, NB, nb_c, device=dev),
                'hist_units': torch.zeros(L, NB * (F + 1), device=dev), 'hist_contig': torch.zeros(L, NB * (nb_c + 1), device=dev),
                'hist_clust': torch.zeros(L, NB * (nb_c + 1), device=dev),
                'err_units': torch.zeros(L, NB, device=dev), 'err_clust': torch.zeros(L, NB, device=dev),
                'h_sum': torch.zeros(L, F, device=dev), 'docs': 0})
            a['docs'] += 1
            a['tokens'] += torch.bincount(bidx, minlength=NB).float()
            du = torch.zeros(L, NB, F, device=dev)
            db = torch.zeros(2, L, NB, nb_c, device=dev)
            ks = torch.zeros(L, T, dtype=torch.int16, device=dev)

            def hook(l, H):
                c = H.abs() * tm.down_norm[l]
                k, mask, cap = docking(c)
                a['curve_units'][l].index_add_(0, bidx, cap)
                a['hist_units'][l] += torch.bincount(bidx * (F + 1) + k, minlength=NB * (F + 1)).float()
                du[l].index_add_(0, bidx, mask.float())
                ks[l] = k.to(torch.int16)
                W = tm.layers[l]['down']
                y = H @ W.T
                ny = y.norm(dim=-1).clamp_min(1e-30)
                a['err_units'][l].index_add_(0, bidx, (y - (H * mask) @ W.T).norm(dim=-1) / ny)
                for j, cb in enumerate([c.view(T, nb_c, BLOCK).sum(-1), c[:, perms[l]].view(T, nb_c, BLOCK).sum(-1)]):
                    kb, mb, capb = docking(cb)
                    name = ['contig', 'clust'][j]
                    a['curve_' + name][l].index_add_(0, bidx, capb)
                    a['hist_' + name][l] += torch.bincount(bidx * (nb_c + 1) + kb, minlength=NB * (nb_c + 1)).float()
                    db[j, l].index_add_(0, bidx, mb.float())
                    if j == 1:
                        unit_mask = torch.zeros_like(mask)
                        unit_mask[:, perms[l]] = mb.repeat_interleave(BLOCK, 1)
                        a['err_clust'][l].index_add_(0, bidx, (y - (H * unit_mask) @ W.T).norm(dim=-1) / ny)
                a['h_sum'][l] += H.sum(0)

            tm.forward(ids, on_ffn=hook)
            doc_unit[i] = du.cpu().numpy().astype(np.uint16)
            doc_blk[i] = db.cpu().numpy().astype(np.uint16)
            doc_tok[i] = torch.bincount(bidx, minlength=NB).cpu().numpy()
            if dom == 'chat':
                chat_k[str(doc)] = ks.cpu().numpy()
            if (i + 1) % 50 == 0 or i + 1 == n:
                print(f'profile {dom}: {i + 1}/{n} docs, {time.time() - t0:.0f}s', flush=True)
        doc_unit.flush()
        doc_blk.flush()
        meta = {'docs': [[d, s] for d, s, _ in docs], 'doc_tokens_per_bucket': doc_tok.tolist(), 'edges': EDGES,
                'block': BLOCK, 'frac': FRAC}
        json.dump(meta, open(f'{RES}/profile_{dom}_meta.json', 'w'))
        np.savez_compressed(f'{RES}/profile_{dom}.npz', **{f'{s}__{k}': (v.cpu().numpy() if torch.is_tensor(v) else np.array(v))
                                                           for s, a in agg.items() for k, v in a.items()})
        if chat_k:
            np.savez_compressed(f'{RES}/chat_k_per_token.npz', **chat_k)
        print(f'profile {dom}: done', flush=True)


if __name__ == '__main__':
    tm = TorchModel(BLOB)
    cmd = sys.argv[1]
    if cmd == 'blocks':
        cmd_blocks(tm)
    elif cmd == 'profile':
        cmd_profile(tm, sys.argv[2:] or None)
