# Thinking, best-of-N, caps and a one-pass judge on a ~9 tok/s fleet (2026-10-18)

Measured on the Q8_0 DeepSeek V4.1-Flash-1017 served across four hosts at ~9 tok/s single stream (run 8 layout).

- **Thinking is not 4x the tokens here.** With thinking off the model runs to the cap: median 1,050 tokens on our 30
  prompts. With thinking on, median 1,450 tokens including the trace. Per prompt, thinking-on took a median 2.8 min; a
  best-of-3 with the model judging itself took 4.6 min; a projected best-of-2 plus judge about 5.4 min.
- **Parallel decode does not scale on a layout with CPU-resident experts.** Three streams drop from 9 tok/s to 2.5 each
  (7.4 aggregate), two streams to ~3.7 each: every extra token in the step touches a different set of experts in RAM and over
  RPC. Best-of-N only pays once the experts are all in VRAM. Separately, a three-slot server with a 49k context cost
  single-stream decode 9.3 -> 5.7 tok/s on this layout; keep one slot for single-stream work.
- **Caps work per request.** `max_tokens` bounds the total; `reasoning_budget_tokens` closes the think block with a wrap-up
  message at the budget, so the answer is written rather than truncated. Verified: a 150-token budget produced ~150 tokens of
  trace and a complete answer. We use 225 tokens as a "25 seconds of thinking" cap at 9 tok/s.
- **A one-pass judge is nearly free.** Qwen3.8-27B (Q4_K_M) on a separate 24 GB card, asked which of N candidates is best
  with `max_tokens: 1` and `logprobs`, returns a pick with a confidence (the digit mass) in about a second over an ssh tunnel.
  On six prompts it agreed with DeepSeek's self-pick five times. The confidence lets a best-of-2 roll a third candidate only
  when the pick is close (`quality/run_prompts.py --best-of 2 --roll-third-below 0.7 --judge-url ...`).
  `quality/rejudge.py` re-picks stored candidates without regenerating.
- **Gated thinking** (`quality/gated_think.py`): instant answer, then the 27B answers one token to "does this draft need
  more thinking?", and if yes the request is re-run with thinking on and the instant answer injected inside the think block
  as "My first thought was: ...". The server reuses the prompt-prefix KV, so the restart costs the re-read of the draft
  (seconds), not a cold start. It saves time when the gate says no and buys a self-refined answer when it says yes; results
  are being collected.
