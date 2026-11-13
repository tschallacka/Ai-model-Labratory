# Related work for binary docking gates

Retrieved 2026-11-12 by web search. Only abstracts and summaries were read, not full papers, so the
figures are the authors' own claims and have not been reproduced. Each entry says how it maps onto Michael's idea.

## Dynamic half (per-token docking)

| Work | Link | What it does | Overlap with docking |
|---|---|---|---|
| Deja Vu (Liu et al., ICML 2023) | https://arxiv.org/abs/2310.17157 | A small per-layer predictor guesses which MLP neurons and heads matter for this input. The authors report OPT-175B running over 2× faster than FasterTransformer. | Same goal: per-token subsets. Uses a learned predictor that runs *before* the FFN, while our docking set is an oracle computed *after* gate and up. |
| CATS (Lee et al., COLM 2024) | https://arxiv.org/abs/2404.08763 | A per-layer threshold on gated-MLP activations, calibrated on about 500 samples. Claimed within 1–2% of base at 50% sparsity with no fine-tuning, and about 15% faster generation with a custom kernel. | This is the 0/1 gate per unit, retrofitted with no training. The difference: its threshold applies to the activation, so gate still has to run in full. |
| LTE (Zheng et al., NeurIPS 2024) | https://arxiv.org/abs/2402.06126 | Groups neurons into experts and trains a router with an efficiency loss. Claimed 25% lower latency for LLaMA2-7B at 50% sparsity. | The block-level docking, but it needs training. |
| MoEfication (Zhang et al., ACL Findings 2022) | https://arxiv.org/abs/2110.01786 | Builds a co-activation graph, splits it into experts, and adds a router. Claimed 10–30% of FFN parameters used while keeping over 95% of performance (ReLU models). | Our co-docking clustering into blocks of 32 is the same construction. |
| LLM in a Flash (Alizadeh et al., Apple 2023) | https://arxiv.org/abs/2312.11514 | A sparsity predictor plus row-column bundling, so only active up/down rows are loaded from flash. | Its bundling is the memory-layout version of keeping a unit's gate/up row and down column together. |
| Hash Layers (Roller et al., NeurIPS 2021) | https://arxiv.org/abs/2106.04426 | Fixed hash routing: token id → expert, with no learned router. It matches Switch and BASE. | The closest to "the token's own code decides where it docks", but it is trained from scratch, not retrofitted. |

## Static half ("focused competency")

| Work | Link | What it does | Overlap |
|---|---|---|---|
| GRIFFIN (Dong, Chen, Chi, 2024) | https://arxiv.org/abs/2404.01365 | Picks the top FFN neurons from the prompt ("flocking") and keeps them for the whole generation. Claimed little loss at 50% FFN, and 1.25–1.29× faster. | A per-prompt static cut: halfway between our dynamic and static halves. |
| D-Pruner (Zhang et al., NAACL Findings 2024) | https://arxiv.org/abs/2405.06275 | Domain-specific pruning that keeps both general and domain weight importance. LLaMA2 at 50% is claimed comparable to dense on healthcare and legal. | This is "you don't ask a coder to write a poem", but unstructured (single weights), not whole units or blocks. |
| Voita et al., "Dead, N-gram, Positional" (ACL Findings 2024) | https://arxiv.org/abs/2309.04827 | Finds OPT FFN neurons that never fire on about 20M diverse tokens: over 70% in some early layers of 66B. More dead neurons with scale. | Direct evidence that "dead areas" exist, though in ReLU OPT. SwiGLU (SmolLM2) has no exact zero, so "dead" has to mean "never docks". |

## What looks new in Michael's framing (to be argued from our own data, not claimed)
- One mechanism for both halves: the same 0/1 docking record drives both the per-token skip and the per-domain cut.
- A domain-by-position (context bucket) matrix of how many units are needed, measured on long chat transcripts.
- Clustered blocks of 32, measured against contiguous blocks for the docking cost on the same tokens.
