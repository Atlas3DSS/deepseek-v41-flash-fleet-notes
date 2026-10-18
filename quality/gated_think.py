#!/usr/bin/env python3
"""Gated thinking: instant answer -> 27B gate ("does this need more thinking?", one forward pass) -> if yes, rerun with
thinking seeded by the instant answer as a draft inside the think block (prompt-prefix KV reused by the server).
usage: gated_think.py --label Q8-gated --url http://127.0.0.1:8190 --judge-url http://127.0.0.1:8191
         [--ids p01,p11] [--gate-threshold 0.5] [--reasoning-budget N] [--max-tokens 1500] [--instant-max 1200]"""
import argparse, json, math, os, re, time, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True); ap.add_argument("--url", required=True); ap.add_argument("--judge-url", required=True)
ap.add_argument("--ids", default=None); ap.add_argument("--gate-threshold", type=float, default=0.5)
ap.add_argument("--instant-max", type=int, default=1200); ap.add_argument("--max-tokens", type=int, default=1500)
ap.add_argument("--reasoning-budget", type=int, default=None, help="cap on the seeded reflection (tokens); None = uncapped")
ap.add_argument("--temperature", type=float, default=0.6); ap.add_argument("--top-p", type=float, default=0.95); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
D = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(D, f"answers-{a.label}.jsonl")
done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
prompts = [json.loads(l) for l in open(os.path.join(D, "prompts.jsonl")) if l.strip()]
if a.ids:
    want = set(a.ids.split(",")); prompts = [p for p in prompts if p["id"] in want]

def post(url, path, body, timeout=3600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def yes_prob(question, draft):
    q = ("Question:\n\n" + question + "\n\nDraft answer:\n\n" + draft +
         "\n\nFor an expert reader, is this draft likely to contain a substantive error, a wrong conclusion, or a material omission "
         "that careful step-by-step reasoning would fix? Answer yes or no.")
    b = {"messages": [{"role": "user", "content": q}], "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 10,
         "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    j = post(a.judge_url, "/v1/chat/completions", b, 600)
    top = ((j["choices"][0].get("logprobs") or {}).get("content") or [{}])[0].get("top_logprobs") or []
    py = pn = 0.0
    for e in top:
        t = (e.get("token") or "").strip().lower()
        if t in ("yes", "y"): py += math.exp(e["logprob"])
        elif t in ("no", "n"): pn += math.exp(e["logprob"])
    return (py / (py + pn)) if (py + pn) > 0 else 0.5, (j["choices"][0]["message"].get("content") or "").strip()

for p in prompts:
    if p["id"] in done:
        continue
    t0 = time.time()
    msgs = [{"role": "user", "content": p["prompt"]}]
    try:
        # 1) instant answer (thinking off)
        inst = post(a.url, "/v1/chat/completions", {"messages": msgs, "max_tokens": a.instant_max, "temperature": a.temperature,
                                                     "top_p": a.top_p, "seed": a.seed, "chat_template_kwargs": {"thinking": False, "enable_thinking": False}})
        draft = inst["choices"][0]["message"].get("content") or ""
        t_inst = time.time() - t0
        # 2) gate
        tg = time.time(); p_yes, gate_raw = yes_prob(p["prompt"], draft); t_gate = time.time() - tg
        rec = {"id": p["id"], "group": p["group"], "label": a.label, "instant": draft, "instant_timings": inst.get("timings", {}),
               "instant_s": round(t_inst, 1), "gate_p_yes": round(p_yes, 3), "gate_raw": gate_raw, "gate_s": round(t_gate, 2),
               "escalated": p_yes >= a.gate_threshold}
        if not rec["escalated"]:
            rec.update({"content": draft, "reasoning_content": "", "timings": inst.get("timings", {}), "finish_reason": inst["choices"][0].get("finish_reason")})
        else:
            # 3) seeded thinking: templated prompt (thinking on) + the draft inside the think block, raw completion
            tpl = post(a.url, "/apply-template", {"messages": msgs, "chat_template_kwargs": {"thinking": True, "enable_thinking": True}})
            prompt = tpl["prompt"]
            open_tag = "<think>"
            if not prompt.rstrip().endswith(open_tag):
                prompt = prompt + open_tag
            prompt += ("\nMy first thought was:\n\n" + draft +
                       "\n\nLet me check that carefully before I answer: go through each claim and number, fix anything wrong or missing, and then give the final answer.\n")
            body = {"prompt": prompt, "n_predict": a.max_tokens, "temperature": a.temperature, "top_p": a.top_p, "seed": a.seed, "cache_prompt": True}
            if a.reasoning_budget is not None:
                body.update({"reasoning_budget_tokens": a.reasoning_budget, "reasoning_budget_start_tag": "<think>", "reasoning_budget_end_tags": ["</think>"]})
            tt = time.time(); c = post(a.url, "/completion", body); t_think = time.time() - tt
            text = c.get("content") or ""
            if "</think>" in text:
                reasoning, content = text.split("</think>", 1)
            else:
                reasoning, content = text, ""
            rec.update({"content": content.strip(), "reasoning_content": reasoning.strip(), "timings": c.get("timings", {}),
                        "finish_reason": "length" if c.get("truncated") else "stop", "think_s": round(t_think, 1),
                        "prompt_cached_tokens": c.get("tokens_cached"), "seeded_prompt_tokens": c.get("tokens_evaluated")})
        rec["wall_s"] = round(time.time() - t0, 1)
    except Exception as e:
        print(f"{p['id']}: ERROR {e}", flush=True); time.sleep(10); continue
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    tm = rec.get("timings", {})
    print(f"{p['id']}: instant {len(draft)} chars in {rec['instant_s']}s | gate p_yes={rec['gate_p_yes']} ({rec['gate_s']}s) -> "
          f"{'ESCALATED: ' + str(len(rec['reasoning_content'])) + ' chars reasoning, ' + str(len(rec['content'])) + ' chars answer, ' + str(rec.get('think_s')) + 's (cached ' + str(rec.get('prompt_cached_tokens')) + ' tok, evaluated ' + str(rec.get('seeded_prompt_tokens')) + ')' if rec['escalated'] else 'kept instant'} | wall {rec['wall_s']}s", flush=True)
