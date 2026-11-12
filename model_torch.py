"""The same SmolLM2 forward pass as model.py, in PyTorch on Apple's GPU (MPS).

Weights come from model.Model (the numpy GGUF reader), so both paths read identical numbers.
check_torch.py records the equivalence between the two before any measurement uses this one.
"""
import torch
import torch.nn.functional as F

from model import Model


class TorchModel:
    def __init__(self, path, device='mps'):
        m = Model(path)
        self.np_model = m
        self.device = device
        t = lambda a: torch.from_numpy(a).to(device)
        self.gguf_sha256 = m.gguf_sha256
        self.D, self.L, self.NH, self.NKV, self.F, self.HD, self.G = m.D, m.L, m.NH, m.NKV, m.F, m.HD, m.G
        self.eps, self.ctx, self.vocab = m.eps, m.ctx, m.vocab
        self.emb = t(m.emb)
        self.out_norm = t(m.out_norm)
        self.layers = [{k: t(v) for k, v in w.items()} for w in m.layers]
        self.down_norm = [t(d) for d in m.down_norm]
        self.cos = t(m.cos)
        self.sin = t(m.sin)

    def rms(self, x, w):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps) * w

    def rope(self, x, heads):
        T = x.shape[0]
        x = x.view(T, heads, self.HD // 2, 2)
        c, s = self.cos[:T, None, :], self.sin[:T, None, :]
        x0, x1 = x[..., 0], x[..., 1]
        return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], -1).view(T, heads, self.HD)

    @torch.no_grad()
    def forward(self, ids, keep=None, bias=None, on_ffn=None):
        """Same contract as model.Model.forward; tensors stay on the device."""
        ids = torch.as_tensor(ids, device=self.device)
        T = ids.shape[0]
        X = self.emb[ids].clone()
        for l, W in enumerate(self.layers):
            n = self.rms(X, W['an'])
            q = self.rope(n @ W['q'].T, self.NH).transpose(0, 1)[None]
            k = self.rope(n @ W['k'].T, self.NKV).transpose(0, 1)
            v = (n @ W['v'].T).view(T, self.NKV, self.HD).transpose(0, 1)
            k = k.repeat_interleave(self.G, 0)[None]
            v = v.repeat_interleave(self.G, 0)[None]
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)[0]
            X += a.transpose(0, 1).reshape(T, self.D) @ W['o'].T
            n = self.rms(X, W['fn'])
            idx = None if keep is None else keep[l]
            if idx is None:
                Gt, U, Dn = W['gate'], W['up'], W['down']
            else:
                idx = torch.as_tensor(idx, device=self.device, dtype=torch.long)
                Gt, U, Dn = W['gate'][idx], W['up'][idx], W['down'][:, idx]
            H = F.silu(n @ Gt.T) * (n @ U.T)
            if on_ffn is not None:
                on_ffn(l, H)
            X += H @ Dn.T
            if bias is not None and bias[l] is not None:
                X += bias[l]
        return X

    def final_hidden(self, X):
        return self.rms(X, self.out_norm)

    @torch.no_grad()
    def log_probs(self, h):
        return torch.log_softmax(h @ self.emb.T, -1)
