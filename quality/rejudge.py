#!/usr/bin/env python3
"""Re-pick best-of-N answers with an external judge, without regenerating: reads the stored candidates from a best-of
answers file and writes a new answers file with the new pick. System-1 pick: one forward pass, digit logprobs.
usage: rejudge.py --in answers-Q8-bo3.jsonl --out answers-Q8-bo3-qwen.jsonl --judge-url http://127.0.0.1:8191 [--label L]"""
import argparse, json, math, os, time, urllib.request
ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--judge-url", required=True); ap.add_argument("--label", default=None)
a = ap.parse_args()
D = os.path.dirname(os.path.abspath(__file__))
prompts = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(D, "prompts.jsonl")) if l.strip()}
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["id"] for l in open(a.out) if l.strip()}
rows = [json.loads(l) for l in open(a.inp) if l.strip()]
for r in rows:
    if r["id"] in done or not r.get("candidates"):
        continue
    texts = r["candidates"]; n = len(texts)
    q = ("You are grading answers for an expert reader. Question:\n\n" + prompts[r["id"]]["prompt"] + "\n\n" +
         "\n\n".join(f"=== Candidate {i+1} ===\n{t}" for i, t in enumerate(texts)) +
         f"\n\nWhich candidate is the most accurate, complete and practically useful? Reply with the candidate number only (1-{n}).")
    b = {"messages": [{"role": "user", "content": q}], "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 10,
         "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(a.judge_url + "/v1/chat/completions", data=json.dumps(b).encode(),
                                                                    headers={"Content-Type": "application/json"}), timeout=600))
    except Exception as e:
        print(f"{r['id']}: ERROR {e}", flush=True); time.sleep(5); continue
    c = d["choices"][0]
    top = ((c.get("logprobs") or {}).get("content") or [{}])[0].get("top_logprobs") or []
    p = {}
    for e in top:
        tok = (e.get("token") or "").strip()
        if tok in [str(i + 1) for i in range(n)]:
            p[tok] = p.get(tok, 0.0) + math.exp(e["logprob"])
    pick = int(max(p, key=p.get)) - 1 if p else 0
    tot = sum(p.values()) or 1.0
    conf = {k: round(v / tot, 3) for k, v in p.items()}
    out = dict(r); out.update({"label": a.label or (r.get("label", "") + "-rejudged"), "content": texts[pick], "judge_pick": pick + 1,
                               "judge_conf": conf, "judge_raw": (c["message"].get("content") or "").strip(), "judge_url": a.judge_url,
                               "judge_s": round(time.time() - t0, 2), "self_judge_pick": r.get("judge_pick"),
                               "judge_prompt_tps": d.get("timings", {}).get("prompt_per_second")})
    with open(a.out, "a") as f:
        f.write(json.dumps(out) + "\n")
    agree = "agrees" if out["judge_pick"] == r.get("judge_pick") else f"differs from self-pick {r.get('judge_pick')}"
    print(f"{r['id']}: qwen picks {pick+1} conf {conf} in {out['judge_s']}s ({agree})", flush=True)
