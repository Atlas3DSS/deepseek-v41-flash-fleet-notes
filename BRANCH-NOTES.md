# DeepSeek-V4.1-Flash on a heterogeneous llama.cpp RPC fleet (Atlas3DSS working branch)

This branch is ggml-org/llama.cpp PR #28696 (head 2076412, DeepSeek V4.1 support) plus the changes we needed to serve
DeepSeek-V4.1-Flash-1017 across four hosts and seven devices (Blackwell + Ampere on one box, three V100s + Xeon RAM on
another, a 4090 in WSL) at the native Q8_0/MXFP4 precision. Work log and measurements: 2026-10-17/18.

## What is in here, with credits
- `src/llama-kv-cache*`, `src/models/deepseek41.cpp`: **per-device replicas of reused (shared) KV cache layers.**
  V4.1 shares the compressed KV stream per layer group (owners 2, 8, 14, 20; readers 3-7, 9-13, 15-19, 21-39); with
  `-sm layer` across devices every remote reader pulled the whole compressed cache per token. Replicas keep only the new
  rows moving. 3.9 -> 9.5 tok/s on our fleet. Implementation: Claude (Anthropic) working in Atlas3DSS's shop; four
  static review passes by Astra (OpenAI Codex, gpt-6-astra) fixed four defects before serving.
- `ggml/src/ggml-rpc/ggml-rpc.cpp`: per-uid RPC graph cache (ported from upstream master's approach; FREE_BUFFER
  invalidation hole closed), tensor-cache integrity (hash verification on read, cleanup on failed write),
  env-gated diagnostics `RPC_STATS` / `RPC_SERVER_STATS`.
- `ggml/src/ggml-cpu/arch/x86/repack.cpp`: in-register E8M0 -> fp32 scale conversion in the MXFP4 gemv/gemm kernels
  (+30-39 % on Zen3 AVX2 for MUL_MAT_ID mxfp4; bit-identical).
- Cherry-picked with attribution (`-x`): cliecy/llama.cpp `dsv41-1017-longctx` (fused lightning indexer, top-k row
  gather at decode, two-level candidate mask, fused-op boundary fix, imatrix fix); the top-k carry follows
  nicholasshirley's analysis in the PR thread (also smalinin); MUL_MAT_ID MMQ padding per Mermiges' review note.
- `DSV41-FLEET-NOTES.md` (this file). The structure note (owners, readers, index sources, consequences for serving and
  for interpretability) is published separately by Atlas3DSS.

## Layout rule that matters more than any patch
Device boundaries must align with the KV-sharing groups unless replicas are on. `-ts` counts the output head as an
extra layer (`il / 41`), so weights sum to 41 with half-integer boundaries, e.g. `-ts 1.5,6,6,6,0,12,9.5`.

Not an upstream PR by intent: these are working patches shared for reference. Use, fork and improve freely.
