# Petri Dish

A step-by-step 3D view of what SmolLM2-135M does to one token, layer by layer: how text becomes numbers,
how one number inside an attention head is computed by hand, where attention looks, and which blocks of
FFN units a token "docks" onto (the fewest blocks carrying 90% of the FFN output). Every step has a
plain-language explanation and a technical one.

## Run it live (any prompt)

Needs Nix and the model pulled in Ollama (`ollama pull smollm2:135m`).

```sh
nix build path:./env --out-link env/result
env/result/bin/python3 petri/server.py          # http://127.0.0.1:8765/
```

The server reads the GGUF straight from Ollama's blob store; the path is `BLOB` in `profile_ffn.py`.
It listens on 127.0.0.1 only and computes each trace on the GPU (MPS) in well under a second.

## Static build (for sharing)

```sh
env/result/bin/python3 petri/build_static.py   # writes petri/static/
```

`petri/static/` is a self-contained site with five fixed sample prompts: host the folder as-is on any web
server (`index.html` is a complete document). `artifact.html` is the same page without the document
skeleton, for a claude.ai artifact, whose host adds it. Traces ship as base64 text because that host
serves no binary file types.

`petri/static/version.json` fingerprints the build: `version` is a hash over every served file, so it
changes exactly when the content does. After a deploy, the live `version.json` matching the local one
means the new build is being served.

## Files

- `petri/viewer.html`: the page (three.js), used by both the live server and the static build.
- `petri/trace.py`: the traced forward pass. It recomputes attention explicitly to record its weights, and
  refuses to return a trace that disagrees with the verified forward pass in `model_torch.py`.
- `petri/server.py`, `petri/build_static.py`, `petri/prompts.json`: the live server, the static build, the samples.
- `petri/blocks_provisional.json`: the FFN block clustering used for docking, built from 20 documents;
  provisional until the full profile run replaces it.
- `model.py` / `model_torch.py`: a GGUF reader and the SmolLM2 forward pass (numpy and PyTorch).
- `sources/hf/SmolLM2-135M-Instruct-tokenizer.json`: the model's tokenizer (Apache-2.0, from
  HuggingFaceTB/SmolLM2-135M-Instruct, revision 12fd25f77366fa6b3b4b768ec3050bf629380bac).
- `petri/vendor/three-0.159.0.min.js`: three.js (MIT), an offline fallback for the live server.
