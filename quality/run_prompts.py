#!/usr/bin/env python3
"""Collect answers from an OpenAI-compatible llama-server for the dsv41 quality test.

usage: run_prompts.py --label Q8 --url http://127.0.0.1:8190 [--prompts prompts.jsonl] [--max-tokens 1500]
Writes quality/answers-<label>.jsonl (one line per prompt: id, group, content, reasoning_content, timings, finish_reason)
and resumes if the file already has an answer for a prompt id. temperature 0, one request at a time.
"""
import argparse, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--url", default="http://127.0.0.1:8190")
ap.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.jsonl"))
ap.add_argument("--max-tokens", type=int, default=1500)
ap.add_argument("--out", default=None)
ap.add_argument("--no-think", action="store_true", help="disable the chat template thinking mode (chat_template_kwargs)")
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--top-p", type=float, default=1.0)
ap.add_argument("--judge-url", default=None, help="separate OpenAI-compatible server for the best-of pick (default: the answering server); the pick is one forward pass reading the logprobs of the digits")
ap.add_argument("--roll-third-below", type=float, default=None, help="with --best-of 2: if the judge's top pick has less than this share of the digit mass, generate one more candidate and pick among three (the third is always accepted; no further rolls)")
ap.add_argument("--reasoning-budget", type=int, default=None, help="cap the thinking trace at N tokens (server closes the think tag with a wrap-up message); --max-tokens stays the total cap")
ap.add_argument("--ids", default=None, help="comma-separated prompt ids to run (default all)")
ap.add_argument("--seed", type=int, default=None, help="sampling seed sent with every request")
ap.add_argument("--best-of", type=int, default=1, help="generate N candidates in parallel, then let the model pick the best (thinking off, temperature 0)")
a = ap.parse_args()
out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"answers-{a.label}.jsonl")

done = set()
if os.path.exists(out):
    for line in open(out):
        try:
            done.add(json.loads(line)["id"])
        except Exception:
            pass

prompts = [json.loads(l) for l in open(a.prompts) if l.strip()]
todo = [p for p in prompts if p["id"] not in done and (not a.ids or p["id"] in set(a.ids.split(",")))]
print(f"{a.label}: {len(done)} done, {len(todo)} to go", flush=True)

with open(out, "a") as f:
    for p in todo:
        body = {"messages": [{"role": "user", "content": p["prompt"]}], "max_tokens": a.max_tokens, "temperature": a.temperature, "top_p": a.top_p}
        if a.no_think:
            body["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
        if a.reasoning_budget is not None:
            body["reasoning_budget_tokens"] = a.reasoning_budget
        if a.seed is not None:
            body["seed"] = a.seed
        def call(b, url=None):
            req = urllib.request.Request((url or a.url) + "/v1/chat/completions", data=json.dumps(b).encode(),
                                         headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=3600))
        t0 = time.time()
        try:
            if a.best_of > 1:
                import math as _m, re as _re
                def judge(texts):
                    # System-1 pick: one forward pass, read the digit logprobs of the first token (no decoding)
                    n = len(texts)
                    judge_q = ("You are grading answers for an expert reader. Question:\n\n" + p["prompt"] +
                               "\n\n" + "\n\n".join(f"=== Candidate {i+1} ===\n{t}" for i, t in enumerate(texts)) +
                               f"\n\nWhich candidate is the most accurate, complete and practically useful? Reply with the candidate number only (1-{n}).")
                    jb = {"messages": [{"role": "user", "content": judge_q}], "max_tokens": 1, "temperature": 0,
                          "logprobs": True, "top_logprobs": 10,
                          "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
                    tj = time.time(); j = call(jb, a.judge_url); js = round(time.time() - tj, 2)
                    jtxt = (j["choices"][0]["message"].get("content") or "").strip()
                    top = ((j["choices"][0].get("logprobs") or {}).get("content") or [{}])[0].get("top_logprobs") or []
                    digit_p = {}
                    for e in top:
                        tok = (e.get("token") or "").strip()
                        if tok in [str(i + 1) for i in range(n)]:
                            digit_p[tok] = digit_p.get(tok, 0.0) + _m.exp(e["logprob"])
                    if digit_p:
                        pk = int(max(digit_p, key=digit_p.get)) - 1
                    else:
                        m = _re.search(r"[1-9]", jtxt); pk = int(m.group(0)) - 1 if m else 0
                    pk = min(max(pk, 0), n - 1)
                    tot = sum(digit_p.values()) or 1.0
                    return pk, {k: round(v / tot, 3) for k, v in digit_p.items()}, jtxt, js
                with ThreadPoolExecutor(max_workers=a.best_of) as ex:
                    cands = list(ex.map(call, [dict(body) for _ in range(a.best_of)]))
                texts = [c["choices"][0]["message"].get("content") or "" for c in cands]
                pick, judge_conf, jtxt, judge_s = judge(texts)
                rolled = False
                if a.roll_third_below is not None and a.best_of == 2 and judge_conf.get(str(pick + 1), 0.0) < a.roll_third_below:
                    cands.append(call(dict(body))); texts.append(cands[-1]["choices"][0]["message"].get("content") or "")
                    pick, judge_conf, jtxt, js2 = judge(texts); judge_s = round(judge_s + js2, 2); rolled = True
                d = cands[pick]
                gen_tok = sum(c.get("usage", {}).get("completion_tokens", 0) for c in cands) + j.get("usage", {}).get("completion_tokens", 0)
            else:
                d = call(body)
        except Exception as e:
            print(f"{p['id']}: ERROR {e}", flush=True)
            time.sleep(10)
            continue
        msg = d["choices"][0]["message"]
        rec = {"id": p["id"], "group": p["group"], "label": a.label, "content": msg.get("content") or "",
               "reasoning_content": msg.get("reasoning_content") or msg.get("reasoning") or "",
               "finish_reason": d["choices"][0].get("finish_reason"), "timings": d.get("timings", {}),
               "wall_s": round(time.time() - t0, 1), "usage": d.get("usage", {})}
        if a.best_of > 1:
            rec.update({"best_of": a.best_of, "rolled_third": rolled, "n_candidates": len(texts), "candidates": texts, "judge_pick": pick + 1, "judge_raw": jtxt,
                        "judge_conf": judge_conf, "judge_s": judge_s, "judge_url": a.judge_url or a.url,
                        "total_generated_tokens": gen_tok})
        f.write(json.dumps(rec) + "\n"); f.flush()
        t = rec["timings"]
        print(f"{p['id']}: {len(rec['content'])} chars content, {len(rec['reasoning_content'])} chars reasoning, "
              f"{t.get('predicted_n')} tok at {t.get('predicted_per_second', 0):.1f} tok/s, finish={rec['finish_reason']}", flush=True)
print("done", flush=True)
