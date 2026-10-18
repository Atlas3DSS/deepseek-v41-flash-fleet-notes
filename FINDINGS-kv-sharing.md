# DeepSeek-V4.1-Flash: cross-layer compressed-KV sharing (structure, evidence, consequences)

Date: 2026-10-17/18. Found while serving DeepSeek-V4.1-Flash-1017 across four machines with llama.cpp RPC (PR #28696
tree). Written up separately from the throughput work because the structure matters to anyone studying this model's
internals, not only to people serving it on heterogeneous hardware.

## The structure (facts read from the checkpoint and the reference graph)

DeepSeek V4.1 does not give every layer its own attention cache. Layers are organised in **groups that share one
compressed key/value stream**:

| Group | Owner (runs the compressor, writes the stream) | Readers (attend over the owner's stream) | Compress ratio |
|---|---|---|---|
| A | layer 2 | 3, 4, 5, 6, 7 | 2 tokens per row |
| B | layer 8 | 9, 10, 11, 12, 13 | 2 tokens per row |
| C | layer 14 | 15, 16, 17, 18, 19 | 2 tokens per row |
| D | layer 20 | 21 through 39 (nineteen layers) | 1 token per row |

Layers 0 and 1 have compress ratio 0: raw attention only, no compressed stream. The checkpoint's
`attention.compress_ratios` array is `[0, 0, 2 x18, 1 x20, 0, 0, 0]` (43 entries: 40 layers plus 3 trailing zeros).

How the owners are identified: only layers 2, 8, 14 and 20 carry `layers.N.attn.compressor.wkv.weight`,
`.compressor.wgate.weight` and `.compressor.norm.weight` in the safetensors release. The reference llama.cpp graph
(`src/models/deepseek41.cpp`, the `dsv41_kv_source` derivation) makes each layer read the stream of the most recent
layer that has a compressor, and refuses to load if a reader's ratio differs from its source's.

Two related sharing maps ride on the same owners:

- **Indexer keys** (`layers.N.attn.indexer.wk.weight`): owned by 2, 8, 14, 20 as well. The lightning indexer's key
  cache is shared per group like the KV stream.
- **Indexer queries / top-k selection** (`layers.N.attn.indexer.wq_b.weight`): owned by 2, 8, 14, 20, 24, 28, 32, 36.
  So group D is subdivided for the sparse top-k selection: 20-23, 24-27, 28-31, 32-35, 36-39 each reuse the top-k
  choice of their own sub-owner, while all of them attend over layer 20's keys and values.

Other structural notes from the same session: engram (n-gram hashed embedding) lookups occur at layers 1 and 14
(`engram_embd-1`, `engram_embd-14` in the scheduler trace; two tables of ~97 GiB each in the Q8 GGUF), and the
residual stream uses the hyper-connection form (`hc_*` F32 tensors, `hc_attn_pre-N` inputs per layer).

## How we found it (so the evidence can be checked)

1. Every multi-device layout we tried moved ~10 MB and ~8,000 packets per generated token, regardless of where the
   experts lived (`logs/net-per-token-run*.txt`).
2. Client-side RPC counters (`RPC_STATS` patch, `logs/rpc-stats-run6f.txt`) showed the graph cache hitting 100 % and the
   bytes dominated by per-layer tensors named `comp_k-N`, 0.5-0.8 MB each and growing with the context.
3. A scheduler trace (`GGML_SCHED_DEBUG=1`, `logs/sched-trace-run6g.txt`) showed `comp_k-N` arriving as *input copies* to
   the splits that run layer N, i.e. the tensor lives on another device. `comp_k` is a view of the group owner's
   compressed cache (`deepseek41.cpp:775-786`), which llama.cpp allocates on the owner's device.
4. The tensor-split rule in `llama-model.cpp` counts the output head as an extra layer, so our boundaries were one layer
   late and two owners (2 and 20) sat on devices other than most of their readers.

## Consequences

**For serving on more than one device.** Device boundaries must coincide with group boundaries, or the owner's whole
compressed cache crosses the interconnect once per reader layer per token, and that cost scales with context length.
Group D alone is twenty layers (~138 GiB of weights at this quantisation), so on today's fleet it necessarily spans two
GPUs; we keep the owner and most readers on one card and the remaining readers on a card in the same host, so the
copies stay on PCIe. The engine-side fix that removes the constraint is a per-device replica of a shared cache that is
updated with the new rows each step instead of re-copied whole. Implemented 2026-10-18 (`patches/dsv41-kv-replicas.patch`,
four review passes): with layer 20's owner on one V100 and readers 26-39 on another host's GPU, the per-token `comp_k`
copies disappear entirely and generation went from 3.9 to 9.5 tokens/s on the same fleet (1.6 MB/token on the wire).

**For mechanistic work on the model.** The KV cache a layer attends over is *not* produced by that layer:

- Layers 3-7, 9-13, 15-19 and 21-39 have no compressor of their own. Their queries attend over keys and values that
  layer 2, 8, 14 or 20 computed from *its* residual stream. Attention patterns in a reader layer are therefore a joint
  property of the reader's queries and the owner's compression, and ablating or patching the owner's compressor
  changes nineteen layers at once in group D.
- The compressed stream is pooled 2:1 in groups A-C (one row stands for two token positions) and 1:1 in group D.
  Position-level attribution inside groups A-C has half the resolution of the raw stream.
- The top-k sparse selection is reused in blocks of four layers inside group D, so "which tokens does layer 30 look at"
  is decided by layer 28's indexer, with keys from layer 20.
- Layers 0, 1 (raw attention) and the engram-injection points (1, 14) are the natural places where fresh positional
  information enters; group boundaries (2, 8, 14, 20) are where the compressed view of the past is rebuilt.
- The compressor at an owner layer reads that layer's attention input (`state_kv = wkv · cur` in the reference graph),
  i.e. the residual stream as produced by every earlier layer, not the owner's own experts. So perturbations (ablation,
  quantisation error) in layers 0-20 reach the stream that layers 21-39 attend over, while perturbations in 21-39 only
  change the residual. Whether that makes early layers measurably more precision-sensitive is a hypothesis for a
  per-layer sweep, not something measured here.

Anyone reproducing this needs only the safetensors file list (owner layers = the ones with `attn.compressor.*`) and the
`compress_ratios` metadata; no weights have to be loaded.

## Files

`PLAN.md` (23:20 section), `logs/sched-trace-run6g.txt`, `logs/rpc-stats-run6f.txt`, `logs/run7-kvgroups.sh` (the
aligned layout), `patches/` (RPC graph cache, cache verification, RPC_STATS counters).
