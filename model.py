"""SmolLM2-135M (llama architecture) forward pass in numpy, read straight from Ollama's GGUF blob.

Processes a whole sequence at once with causal attention, so one pass over a document gives
every prefix's activations. Optional hooks expose each layer's feed-forward hidden state, and
optional per-layer unit masks and bias vectors apply a static cut with compensation.
"""
import hashlib
import struct

import numpy as np

_SCALAR = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}


class GGUF:
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.b = f.read()
        self.sha256 = hashlib.sha256(self.b).hexdigest()
        if self.b[:4] != b'GGUF':
            raise ValueError(f'{path} is not a GGUF file')
        self.p = 8
        n_tensors, n_kv = self._r('<Q'), self._r('<Q')
        self.kv = {}
        for _ in range(n_kv):
            k = self._s()
            self.kv[k] = self._v(self._r('<I'))
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._s()
            dims = [self._r('<Q') for _ in range(self._r('<I'))]
            self.tensors[name] = (dims, self._r('<I'), self._r('<Q'))
        align = self.kv.get('general.alignment', 32)
        self.data_start = (self.p + align - 1) // align * align

    def _r(self, fmt):
        v = struct.unpack_from(fmt, self.b, self.p)[0]
        self.p += struct.calcsize(fmt)
        return v

    def _s(self):
        n = self._r('<Q')
        s = self.b[self.p:self.p + n].decode('utf-8', 'replace')
        self.p += n
        return s

    def _v(self, t):
        if t in _SCALAR:
            return self._r(_SCALAR[t])
        if t == 8:
            return self._s()
        if t == 9:
            it, n = self._r('<I'), self._r('<Q')
            return [self._v(it) for _ in range(n)]
        raise ValueError(f'unknown metadata type {t}')

    def load(self, name):
        """A GGUF weight with dims [in, out] is `out` rows of `in` values: returned as (out, in)."""
        dims, typ, off = self.tensors[name]
        dtype = {0: np.float32, 1: np.float16}.get(typ)
        if dtype is None:
            raise ValueError(f'{name}: tensor type {typ} not handled')
        n = int(np.prod(dims))
        a = np.frombuffer(self.b, dtype=dtype, count=n, offset=self.data_start + off).astype(np.float32)
        return a.reshape(dims[::-1]) if len(dims) > 1 else a


class Model:
    def __init__(self, path):
        g = GGUF(path)
        kv = g.kv
        self.gguf_sha256 = g.sha256
        self.D = kv['llama.embedding_length']
        self.L = kv['llama.block_count']
        self.NH = kv['llama.attention.head_count']
        self.NKV = kv['llama.attention.head_count_kv']
        self.F = kv['llama.feed_forward_length']
        self.eps = kv['llama.attention.layer_norm_rms_epsilon']
        self.ctx = kv['llama.context_length']
        self.HD = self.D // self.NH
        self.G = self.NH // self.NKV
        self.vocab = kv['tokenizer.ggml.tokens']
        self.chat_template = kv.get('tokenizer.chat_template')
        self.emb = g.load('token_embd.weight')
        self.out_norm = g.load('output_norm.weight')
        names = dict(an='attn_norm', q='attn_q', k='attn_k', v='attn_v', o='attn_output',
                     fn='ffn_norm', gate='ffn_gate', up='ffn_up', down='ffn_down')
        self.layers = [{k: g.load(f'blk.{l}.{n}.weight') for k, n in names.items()} for l in range(self.L)]
        # Column norm of ffn_down per unit: the size of what a unit writes per unit of activation.
        self.down_norm = [np.linalg.norm(w['down'], axis=0) for w in self.layers]
        inv = kv['llama.rope.freq_base'] ** (-np.arange(0, self.HD, 2) / self.HD)
        ang = np.arange(self.ctx)[:, None] * inv[None, :]
        self.cos, self.sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    def rms(self, x, w):
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + self.eps) * w

    def rope(self, x, heads):
        """llama.cpp 'normal' RoPE: rotate adjacent pairs inside each head."""
        T = x.shape[0]
        x = x.reshape(T, heads, self.HD // 2, 2)
        c, s = self.cos[:T, None, :], self.sin[:T, None, :]
        x0, x1 = x[..., 0], x[..., 1]
        return np.stack([x0 * c - x1 * s, x0 * s + x1 * c], -1).reshape(T, heads * self.HD)

    def attention(self, Q, K, V, chunk=1024):
        T = Q.shape[0]
        Q = Q.reshape(T, self.NH, self.HD)
        K = K.reshape(T, self.NKV, self.HD)
        V = V.reshape(T, self.NKV, self.HD)
        out = np.empty((T, self.NH, self.HD), np.float32)
        scale = np.float32(1 / np.sqrt(self.HD))
        for g in range(self.NKV):
            hs = slice(g * self.G, (g + 1) * self.G)
            for s in range(0, T, chunk):
                e = min(T, s + chunk)
                q = Q[s:e, hs, :].transpose(1, 0, 2)                 # (G, c, HD)
                sc = (q @ K[:e, g, :].T) * scale                       # (G, c, e)
                future = np.arange(s, e)[:, None] < np.arange(e)[None, :]
                sc[:, future] = -np.inf
                sc -= sc.max(-1, keepdims=True)
                np.exp(sc, out=sc)
                sc /= sc.sum(-1, keepdims=True)
                out[s:e, hs, :] = (sc @ V[:e, g, :]).transpose(1, 0, 2)
        return out.reshape(T, self.D)

    def forward(self, ids, keep=None, bias=None, on_ffn=None):
        """Residual stream after the last layer, shape (T, D).

        keep[l]: indices of the FFN units kept in layer l (None = all).
        bias[l]: vector added to layer l's FFN output (compensation for cut units).
        on_ffn(l, H): called with layer l's FFN hidden state H = silu(gate)*up, shape (T, F).
        """
        ids = np.asarray(ids)
        X = self.emb[ids].copy()
        with np.errstate(over='ignore'):
            for l, W in enumerate(self.layers):
                n = self.rms(X, W['an'])
                Q = self.rope(n @ W['q'].T, self.NH)
                K = self.rope(n @ W['k'].T, self.NKV)
                X += self.attention(Q, K, n @ W['v'].T) @ W['o'].T
                n = self.rms(X, W['fn'])
                idx = None if keep is None else keep[l]
                if idx is None:
                    Gt, U, Dn = W['gate'], W['up'], W['down']
                else:
                    Gt, U, Dn = W['gate'][idx], W['up'][idx], W['down'][:, idx]
                Gv = n @ Gt.T
                H = Gv / (1 + np.exp(-Gv)) * (n @ U.T)
                if on_ffn is not None:
                    on_ffn(l, H)
                X += H @ Dn.T
                if bias is not None and bias[l] is not None:
                    X += bias[l]
        return X

    def final_hidden(self, X):
        return self.rms(X, self.out_norm)

    def log_probs(self, h, chunk=512):
        """Log-softmax over the vocabulary for final hidden states h (T, D), in chunks."""
        out = []
        for s in range(0, h.shape[0], chunk):
            z = h[s:s + chunk] @ self.emb.T
            z -= z.max(-1, keepdims=True)
            z -= np.log(np.exp(z).sum(-1, keepdims=True))
            out.append(z)
        return np.concatenate(out)
