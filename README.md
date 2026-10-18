---
license: mit
tags:
- deepseek
- llama.cpp
- inference
- interpretability
- notes
pretty_name: DeepSeek-V4.1-Flash fleet notes
---
# DeepSeek-V4.1-Flash on a heterogeneous llama.cpp fleet — notes and patches

Working notes from serving DeepSeek-V4.1-Flash-1017 at its native Q8_0/MXFP4 precision across four machines
(Blackwell + Ampere + three V100s + a 4090, llama.cpp RPC), 2026-10-17/18. Shared for reference, not as polished PRs.

- **[FINDINGS-kv-sharing.md](FINDINGS-kv-sharing.md)** — how V4.1 shares its compressed attention cache across layer
  groups (owners 2, 8, 14, 20), the index-key and top-k sharing maps, how we found it, and what it means for multi-GPU
  serving and for interpretability work.
- **[patches/](patches/)** — against ggml-org/llama.cpp PR #28696 head 2076412:
  `dsv41-kv-replicas.patch` (per-device replicas of reused KV layers: 3.9 -> 9.5 tok/s on our fleet),
  `dsv41-rpc-graph-cache.patch` (per-uid RPC graph cache), `dsv41-rpc-cache-verify.patch` (rpc tensor-cache integrity),
  `dsv41-mxfp4-e8m0-vector-scale.patch` (in-register E8M0 scale conversion in the x86 MXFP4 kernels, +30-39 % on AVX2).
  The full branch, including cherry-picks from cliecy/llama.cpp with attribution, is [`dsv41-fleet` on our llama.cpp fork](https://github.com/Atlas3DSS/llama.cpp/tree/dsv41-fleet).
- **[quality/](quality/)** — the 30-prompt set (polymer chemistry, lab technique, coding) and the collection harness we use
  to compare quants blind.
- **[BRANCH-NOTES.md](BRANCH-NOTES.md)** — what the branch carries and who to credit.

Headline numbers on this fleet, same model file throughout: 1.3 tok/s (naive layer split) -> 3.9 (device boundaries
aligned to the KV-sharing groups) -> 9.5 (per-device replicas) -> 9.3-10 (kernel + decode fixes). Everything outside VRAM
(15 expert layers in RAM) is what remains; six V100s arriving this week should remove it.

Mirror: https://huggingface.co/datasets/Atlas3D/deepseek-v41-flash-fleet-notes

Credits: Atlas3DSS (owner, direction, hardware); Claude (Anthropic) did the measurement, engine work and writing in this
shop; Astra (OpenAI Codex) reviewed every patch; cliecy, nicholasshirley, smalinin, Mermiges and the PR #28696 authors
for the pieces we built on. MIT, like llama.cpp.

## Added 2026-10-18 (evening)

- [FINDINGS-depth-loop.md](FINDINGS-depth-loop.md): running layers 30-35 a second time at inference, with controls; free in
  decode time on our layout, outputs diverge, quality scoring in progress. Patch: `patches/dsv41-depth-loop.patch`; raw
  sample answers: `quality/depth-loop-sample/`.
- [FINDINGS-serving-economics.md](FINDINGS-serving-economics.md): thinking vs best-of-N wall time at ~9 tok/s, why parallel
  decode does not scale with CPU-resident experts, per-request reasoning caps, a one-token logprob judge on a side GPU, and
  gated thinking with a seeded think block (`quality/gated_think.py`, `quality/rejudge.py`).
