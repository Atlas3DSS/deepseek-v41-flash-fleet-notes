#!/usr/bin/env python3
"""Collect answers from an OpenAI-compatible llama-server for the dsv41 quality test.

usage: run_prompts.py --label Q8 --url http://127.0.0.1:8190 [--prompts prompts.jsonl] [--max-tokens 1500]
Writes quality/answers-<label>.jsonl (one line per prompt: id, group, content, reasoning_content, timings, finish_reason)
and resumes if the file already has an answer for a prompt id. temperature 0, one request at a time.
"""
import argparse, json, os, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--url", default="http://127.0.0.1:8190")
ap.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.jsonl"))
ap.add_argument("--max-tokens", type=int, default=1500)
ap.add_argument("--out", default=None)
ap.add_argument("--no-think", action="store_true", help="disable the chat template thinking mode (chat_template_kwargs)")
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--top-p", type=float, default=1.0)
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
todo = [p for p in prompts if p["id"] not in done]
print(f"{a.label}: {len(done)} done, {len(todo)} to go", flush=True)

with open(out, "a") as f:
    for p in todo:
        body = {"messages": [{"role": "user", "content": p["prompt"]}], "max_tokens": a.max_tokens, "temperature": a.temperature, "top_p": a.top_p}
        if a.no_think:
            body["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
        req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            d = json.load(urllib.request.urlopen(req, timeout=3600))
        except Exception as e:
            print(f"{p['id']}: ERROR {e}", flush=True)
            time.sleep(10)
            continue
        msg = d["choices"][0]["message"]
        rec = {"id": p["id"], "group": p["group"], "label": a.label, "content": msg.get("content") or "",
               "reasoning_content": msg.get("reasoning_content") or msg.get("reasoning") or "",
               "finish_reason": d["choices"][0].get("finish_reason"), "timings": d.get("timings", {}),
               "wall_s": round(time.time() - t0, 1), "usage": d.get("usage", {})}
        f.write(json.dumps(rec) + "\n"); f.flush()
        t = rec["timings"]
        print(f"{p['id']}: {len(rec['content'])} chars content, {len(rec['reasoning_content'])} chars reasoning, "
              f"{t.get('predicted_n')} tok at {t.get('predicted_per_second', 0):.1f} tok/s, finish={rec['finish_reason']}", flush=True)
print("done", flush=True)
