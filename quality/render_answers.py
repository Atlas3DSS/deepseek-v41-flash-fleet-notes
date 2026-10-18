#!/usr/bin/env python3
"""Render the quality-pass answers as one HTML page: thinking-on answers with their traces, the no-think answer beside
each for comparison. usage: render_answers.py OUT.html [--think answers-Q8-think.jsonl] [--plain answers-Q8.jsonl]"""
import json, html, os, sys, re, datetime
D = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1]
think_f = sys.argv[sys.argv.index("--think") + 1] if "--think" in sys.argv else os.path.join(D, "answers-Q8-think.jsonl")
plain_f = sys.argv[sys.argv.index("--plain") + 1] if "--plain" in sys.argv else os.path.join(D, "answers-Q8.jsonl")
label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else "Q8"

def load(f):
    r = {}
    if os.path.exists(f):
        for line in open(f):
            try:
                d = json.loads(line); r[d["id"]] = d
            except Exception:
                pass
    return r

prompts = [json.loads(l) for l in open(os.path.join(D, "prompts.jsonl")) if l.strip()]
think, plain = load(think_f), load(plain_f)
bo3 = load(os.path.join(D, f"answers-{label}-bo3.jsonl"))
qj = load(os.path.join(D, f"answers-{label}-bo3-qwen.jsonl"))
import glob
# loop-experiment passes: answers-Q8-loop-<cfg>.jsonl (best-of-2 + roll, Mike judge) and answers-Q8-loop-<cfg>-think25.jsonl
loop_runs = []
for f in sorted(glob.glob(os.path.join(D, f"answers-{label}-loop-*.jsonl"))):
    tag = os.path.basename(f)[len(f"answers-{label}-loop-"):-len(".jsonl")]
    kind = "thinking, 25 s cap" if tag.endswith("-think25") else ("thinking, uncapped" if tag.endswith("-think") else "best-of-2 (+1 roll), Mike judge")
    cfg = tag[:-len("-think25")] if tag.endswith("-think25") else (tag[:-len("-think")] if tag.endswith("-think") else tag)
    cfg = {"off": "layers as trained"}.get(cfg, "layers " + cfg.replace("_", "-", 1).replace("_", ", alpha ") + " looped")
    loop_runs.append((f"{cfg} · {kind}", load(f)))
for name, f in (("thinking capped at 25 s (225-token budget)", f"answers-{label}-think25.jsonl"), ("thinking uncapped (15,000 budget, seed 42)", f"answers-{label}-think-full.jsonl")):
    if os.path.exists(os.path.join(D, f)):
        loop_runs.append((name, load(os.path.join(D, f))))
gated = load(os.path.join(D, f"answers-{label}-gated.jsonl"))  # same candidates re-picked by the Qwen judge on Mike
groups = {"polymer": "Polymer chemistry", "labtech": "Lab technique", "coding": "Coding"}

def md(t):
    t = html.escape(t)
    t = re.sub(r"```(\w+)?\n(.*?)```", lambda m: f"<pre><code>{m.group(2)}</code></pre>", t, flags=re.S)
    t = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?m)^#{1,4} (.+)$", r"<h4>\1</h4>", t)
    t = re.sub(r"(?m)^\s*[-*] (.+)$", r"<li>\1</li>", t)
    t = re.sub(r"(<li>.*?</li>\n?)+", lambda m: f"<ul>{m.group(0)}</ul>", t, flags=re.S)
    return "".join(f"<p>{p}</p>" if not p.startswith(("<pre", "<ul", "<h4")) else p for p in re.split(r"\n{2,}", t) if p.strip())

# loop sample: three prompts, greedy, seed 42, 600-token cap, thinking off; identical settings and binary across legs
sample_legs = [("layers 30-35 twice, alpha 0.5", "answers-Q8-loopsample-30_35_0.5.jsonl"), ("control: loop off, same binary", "answers-Q8-loopsample-off.jsonl"), ("layers 30-35 twice, alpha 0.67", "answers-Q8-loopsample-30_35_0.67.jsonl"), ("layers 30-35 twice, alpha 1.0", "answers-Q8-loopsample-30_35_1.0.jsonl")]
sample = [(n, load(os.path.join(D, f))) for n, f in sample_legs]
def word_diff_html(text, ref):
    """text with words that differ from ref highlighted; shared words plain. Word-level, whitespace kept."""
    import difflib, re as _re
    tw = _re.findall(r"\s+|[^\s]+", text); rw = _re.findall(r"\s+|[^\s]+", ref)
    tn = [w for w in tw if not w.isspace()]; rn = [w for w in rw if not w.isspace()]
    sm = difflib.SequenceMatcher(None, tn, rn, autojunk=False)
    flag = [False] * len(tn)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            for k in range(i1, i2): flag[k] = True
    out = []; k = 0
    for w in tw:
        if w.isspace(): out.append(w); continue
        out.append(f'<mark>{html.escape(w)}</mark>' if flag[k] else f'<span class="same">{html.escape(w)}</span>'); k += 1
    return '<div class="diff">' + "".join(out) + "</div>"

sample_html = ""
if any(r for _, r in sample):
    import difflib
    ctrl = dict(sample)[sample_legs[1][0]]
    blocks = []
    for pid in ("p01", "p11", "c01"):
        pr = next(x for x in prompts if x["id"] == pid)
        cols = ""
        for name, runs in sample:
            r = runs.get(pid)
            if not r:
                cols += f'<div class="col pending"><h4>{html.escape(name)}</h4><p class="meta">pending</p></div>'; continue
            note = ""
            c = ctrl.get(pid)
            if c and r is not c:
                ratio = difflib.SequenceMatcher(None, r["content"], c["content"]).ratio()
                note = " · identical to control" if r["content"] == c["content"] else f" · {ratio:.0%} similar to control"
            shown = md(r["content"]) if (not c or r is c) else word_diff_html(r["content"], c["content"])
            cols += f'<div class="col"><h4>{html.escape(name)}</h4><p class="meta">{r["timings"].get("predicted_n")} tokens at {r["timings"].get("predicted_per_second",0):.1f} tok/s{note}</p><div class="answer">{shown}</div></div>'
        q = plain.get(pid)
        saved = f'<details><summary>Saved greedy baseline from the earlier binary ({q["timings"].get("predicted_n")} tokens)</summary><div class="answer plain">{md(q["content"])}</div></details>' if q else ""
        blocks.append(f'<article class="card done"><header><span class="id">{pid}</span><span class="grp">{groups[pr["group"]]}</span><span class="st">loop sample</span></header><p class="prompt">{html.escape(pr["prompt"])}</p><div class="cols">{cols}</div>{saved}</article>')
    sample_html = '<h2>Depth-loop sample</h2><p class="lede">Same three prompts, thinking off, temperature 0, top_p 1, seed 42, 600-token cap, one request at a time on the same binary and device layout. The only change between columns is whether layers 30-35 run a second time and how strongly the second pass is blended in. In the looped columns, <mark>highlighted words</mark> are not in the control answer; plain words are shared with it.</p>' + "".join(blocks)

cards = []
for p in prompts:
    t = think.get(p["id"]); q = plain.get(p["id"])
    status = "done" if (t or bo3.get(p["id"]) or gated.get(p["id"]) or any(p["id"] in r for _, r in loop_runs)) else "pending"
    body = ""
    if t:
        tm = t["timings"]; fin = t.get("finish_reason")
        body += f'<div class="meta">{tm.get("predicted_n")} tokens at {tm.get("predicted_per_second", 0):.1f} tok/s · reasoning {len(t["reasoning_content"]):,} chars · answer {len(t["content"]):,} chars · {"ran into the budget" if fin == "length" else "finished"} · wall {t.get("wall_s", 0)/60:.1f} min</div>'
        body += f'<h3>Answer (thinking on)</h3><div class="answer">{md(t["content"]) if t["content"] else "<em>no final answer, trace only</em>"}</div>'
        body += f'<details><summary>Reasoning trace</summary><pre class="trace">{html.escape(t["reasoning_content"])}</pre></details>'
    b = bo3.get(p["id"])
    if b:
        j = qj.get(p["id"]); shown = b
        qnote = ""
        if j:
            conf = ", ".join(f"{k}: {v:.0%}" for k, v in sorted(j.get("judge_conf", {}).items()))
            if j["judge_pick"] == b.get("judge_pick"):
                qnote = f' · Qwen judge agrees ({conf}, {j.get("judge_s")} s)'
            else:
                qnote = f' · <strong>Qwen judge picked candidate {j["judge_pick"]} instead</strong> ({conf}, {j.get("judge_s")} s); shown below is Qwen\'s pick, the self-pick is folded'
                shown = j
        body += f'<h3>Best-of-3, thinking off (self-pick candidate {b.get("judge_pick")}{qnote}; {b.get("total_generated_tokens")} tokens generated in total, {b.get("wall_s",0)/60:.1f} min)</h3><div class="answer bo3">{md(shown["content"])}</div>'
        others = [(i+1, c) for i, c in enumerate(b.get("candidates", [])) if i+1 != shown.get("judge_pick")]
        body += '<details><summary>The other two candidates</summary>' + ''.join(f'<h4>Candidate {i}</h4><div class="answer plain">{md(c)}</div>' for i, c in others) + '</details>'
    g = gated.get(p["id"])
    if g:
        gate = f'gate p(yes) {g.get("gate_p_yes"):.0%} in {g.get("gate_s")} s'
        if g.get("escalated"):
            body += f'<h3>Gated thinking: escalated</h3><div class="meta">instant {g.get("instant_s")} s · {gate} · seeded rerun {g.get("think_s")} s (prefix cached {g.get("prompt_cached_tokens")} tokens, re-read {g.get("seeded_prompt_tokens")}) · reasoning {len(g.get("reasoning_content") or ""):,} chars · wall {g.get("wall_s",0)/60:.1f} min</div><div class="answer gated">{md(g["content"]) if g["content"] else "<em>no final answer</em>"}</div>'
            body += f'<details><summary>Seeded reasoning trace</summary><pre class="trace">{html.escape(g.get("reasoning_content") or "")}</pre></details>'
            body += f'<details><summary>The instant draft it started from</summary><div class="answer plain">{md(g["instant"])}</div></details>'
        else:
            body += f'<h3>Gated thinking: instant answer kept</h3><div class="meta">instant {g.get("instant_s")} s · {gate} · wall {g.get("wall_s",0)/60:.1f} min</div><div class="answer gated">{md(g["content"])}</div>'
    for name, runs in loop_runs:
        r = runs.get(p["id"])
        if not r:
            continue
        tm = r.get("timings", {}); extra = ""
        if r.get("best_of"):
            conf = ", ".join(f"{k}: {v:.0%}" for k, v in sorted((r.get("judge_conf") or {}).items()))
            extra = f' · pick {r.get("judge_pick")} of {r.get("n_candidates")}{" after a roll" if r.get("rolled_third") else ""} ({conf}) · {r.get("total_generated_tokens")} tokens total'
        else:
            extra = f' · reasoning {len(r.get("reasoning_content") or ""):,} chars · {tm.get("predicted_n")} tokens'
        body += f'<h3>{html.escape(name)}</h3><div class="meta">{r.get("wall_s",0)/60:.1f} min{extra}</div><div class="answer loop">{md(r["content"]) if r["content"] else "<em>no final answer</em>"}</div>'
        if r.get("reasoning_content"):
            body += f'<details><summary>Reasoning trace (capped)</summary><pre class="trace">{html.escape(r["reasoning_content"])}</pre></details>'
    if q:
        body += f'<details><summary>Same prompt with thinking off, single greedy answer ({q["timings"].get("predicted_n")} tokens)</summary><div class="answer plain">{md(q["content"])}</div></details>'
    cards.append(f'<article class="card {status}" data-group="{p["group"]}"><header><span class="id">{p["id"]}</span><span class="grp">{groups[p["group"]]}</span><span class="st">{status}</span></header><p class="prompt">{html.escape(p["prompt"])}</p>{body}</article>')

n_done = len(think); n_len = sum(1 for t in think.values() if t.get("finish_reason") == "length")
now = datetime.datetime.now().strftime("%H:%M")
page = f'''<title>V4.1 Quality Answers</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
:root{{--bg:#f6f4ef;--fg:#1f1d19;--muted:#6b665c;--card:#fffdf8;--line:#d9d3c7;--accent:#8a4b1f;--pre:#f1ede4;--ok:#2d6a4f;--mark:#ffd98a;--same:#6f6a60;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#17160f;--fg:#ebe6da;--muted:#a39d90;--card:#201e16;--line:#3a372d;--accent:#e0a068;--pre:#1a1912;--ok:#7fc8a0;--mark:#7a5a1a;--same:#a39d90;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#17160f;--fg:#ebe6da;--muted:#a39d90;--card:#201e16;--line:#3a372d;--accent:#e0a068;--pre:#1a1912;--ok:#7fc8a0;--mark:#7a5a1a;--same:#a39d90;color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font:16px/1.55 Georgia,"Iowan Old Style",serif;margin:0;padding-block:24px 56px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}}
main{{max-width:920px;margin:0 auto}} h1{{font-size:1.6rem;margin:0 0 .25rem;text-wrap:balance}} .lede{{color:var(--muted);margin:0 0 1rem;max-width:70ch}}
.filters{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}} .filters button{{font:inherit;font-size:.9rem;padding:4px 12px;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:999px;cursor:pointer}} .filters button[aria-pressed="true"]{{border-color:var(--accent);color:var(--accent)}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin-bottom:12px;min-width:0}} .card.pending{{opacity:.55}}
header{{display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}} .id{{font-family:var(--mono);font-size:.85rem;color:var(--accent)}} .grp{{font-size:.75rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}} .st{{margin-left:auto;font-size:.8rem;color:var(--ok)}} .pending .st{{color:var(--muted)}}
.prompt{{margin:.4rem 0 .6rem;color:var(--muted);max-width:72ch}} .meta{{font-size:.8rem;color:var(--muted);margin:.2rem 0 .6rem}} h3{{font-size:1rem;margin:.6rem 0 .3rem}} h4{{font-size:.95rem;margin:.6rem 0 .2rem}}
.answer p{{margin:.45rem 0;max-width:72ch}} .answer ul{{padding-left:1.2rem;max-width:72ch}} .answer pre{{background:var(--pre);padding:10px 12px;border-radius:6px;overflow-x:auto;font:13px/1.5 var(--mono)}} code{{font:.9em var(--mono);background:var(--pre);padding:1px 4px;border-radius:4px}}
details{{margin-top:8px}} summary{{cursor:pointer;color:var(--accent);font-size:.95rem}} .trace{{margin:8px 0 0;padding:12px;background:var(--pre);border-radius:6px;white-space:pre-wrap;overflow-wrap:anywhere;font:12.5px/1.5 var(--mono);max-height:60vh;overflow:auto}}
h2{{font-size:1.25rem;margin:1.4rem 0 .4rem}} .cols{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin-top:.5rem}} .col{{min-width:0;border-top:2px solid var(--accent);padding-top:6px}} .col.pending{{opacity:.5}} .col h4{{margin:.2rem 0}} .diff{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.95rem;line-height:1.6}} .diff mark{{background:var(--mark);color:inherit;border-radius:3px;padding:0 2px}} .diff .same{{color:var(--same)}}
.plain{{border-left:3px solid var(--line);padding-left:12px;margin-top:8px}} .bo3{{border-left:3px solid var(--ok);padding-left:12px}} .loop{{border-left:3px solid var(--accent);padding-left:12px}} .gated{{border-left:3px solid #4a6fa5;padding-left:12px}}
</style>
<main>
<h1>DeepSeek V4.1 ({label}) quality answers</h1>
<p class="lede">Three ways to answer the same 30 prompts: thinking on (temperature 0.6, top_p 0.95, 15,000-token budget; {n_done} of 30 in, {n_len} ran into the budget), best-of-3 with thinking off (three candidates at 0.6 decoded in parallel, the model picks one; {len(bo3)} of 30 in), and a single greedy thinking-off answer. Where a Qwen3.8-27B judge on the 4090 has re-picked (one forward pass, digit logprobs; {len(qj)} so far), the card says whether it agreed. Loop-experiment passes (best-of-2 with one gated roll judged by Qwen, and thinking capped at 25 s) appear per prompt as they land. Gated thinking: an instant answer, a one-pass Qwen gate asking whether it needs more thought ({sum(1 for g in gated.values() if g.get('escalated'))} of {len(gated)} escalated so far), and if so a rerun whose think block starts from the instant draft. Updated {now}; the page is republished as answers land.</p>
{sample_html}
<h2>Quality passes</h2>
<div class="filters" role="group" aria-label="filter"><button aria-pressed="true" data-f="all">All</button><button aria-pressed="false" data-f="polymer">Polymer</button><button aria-pressed="false" data-f="labtech">Lab technique</button><button aria-pressed="false" data-f="coding">Coding</button><button aria-pressed="false" data-f="done">Done only</button></div>
{''.join(cards)}
</main>
<script>
document.querySelectorAll('.filters button').forEach(b=>b.addEventListener('click',()=>{{document.querySelectorAll('.filters button').forEach(x=>x.setAttribute('aria-pressed','false'));b.setAttribute('aria-pressed','true');const f=b.dataset.f;document.querySelectorAll('.card').forEach(c=>{{c.hidden=!(f==='all'||c.dataset.group===f||(f==='done'&&c.classList.contains('done')))}})}}));
</script>'''
open(out, "w").write(page)
print(f"{out}: {n_done}/30 thinking answers, {len(plain)} plain answers, {len(page)/1e3:.0f} KB")
