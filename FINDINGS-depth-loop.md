# Depth loop at inference time on DeepSeek V4.1-Flash (2026-10-18, preliminary)

**What we did.** Added an environment-gated switch to our llama.cpp branch (`patches/dsv41-depth-loop.patch`,
`DSV41_LOOP=start-end:alpha`): after layer `end` has run, layers `start..end` run a second time from the current residual
stream, and the result is blended back as `stream = p1 + alpha * (p2 - p1)`. Nothing is trained; the weights are re-applied as
they are. Slices containing an engram layer are refused. The hyper-connection mixes are the second pass's.

**Why this slice.** Layers 30-35 sit inside KV-sharing group D (owner 20, readers 21-39; see `FINDINGS-kv-sharing.md`), so a
second pass does not rewrite a compressed cache that later layers already consumed, and on our layout they all live on one
card (the PRO 6000) with their experts in VRAM, so the second pass moves nothing over the network.

**Controls.** Three prompts (a photoinitiator chemistry question, a resin-handling SOP, a Jacobs working-curve fit in Python),
thinking off, temperature 0, top_p 1, seed 42, 600-token cap, one request at a time, same binary and device layout for every
leg. Legs: alpha 0.5, loop off (control), alpha 1.0, alpha 0.67 (pending at the time of writing). Raw answers are in
`quality/depth-loop-sample/`.

**Results so far.**

| leg | p01 similarity to control | p11 | c01 | decode tok/s (alone) |
|---|---|---|---|---|
| control (loop off) | 1.00 (reproduces the pre-change baseline token for token) | 1.00 | 1.00 | 5.3-5.7 |
| 30-35, alpha 0.5 | 0.18 | 0.06 | 0.32 | 5.7 |
| 30-35, alpha 1.0 | 0.18 | 0.08 | 0.24 | 5.8-6.2 |

Similarity is `difflib` ratio on the answer text. The looped answers diverge within the first sentence (characters 54, 2
and 46) and the two alphas also differ from each other, so the blend weight is a continuous dial. Decode speed is unchanged
within noise: six extra layers on the fastest, non-bottleneck card are a few milliseconds on a ~175 ms token (the one-stream
server on this layout; see `FINDINGS-serving-economics.md` for why that number is not 110 ms). Prefill, being compute-bound,
would pay the full ~15 % for six of forty layers.

**What we do not know yet.** Whether the looped answers are better. One exhibit: on the photoinitiator question the control
says doubling the initiator *deepens* cure and then hedges, the looped answer commits to the opposite, shallower cure by
Beer-Lambert, which is the textbook direction for an initiator-limited inner-filter regime. Perplexity and a 30-prompt judged
pass per setting are queued. Treat everything here as one reader's three-prompt look.

**If it holds, the scheduler rule.** Loop windows belong on the fastest card that is not the token's bottleneck, with their
experts in that card's VRAM; a window must stay inside one KV-sharing group and one device; which windows help has to be
measured by sliding a window through the slice. A per-token gate (loop only when the model is unsure; the MTP heads are the
cheap uncertainty reader once exported) turns the fixed setting into adaptive depth.
