"""Executive Intelligence Dashboard — NVIDIA-themed.

Reads the precomputed results store (data/results/analysis.json) and renders the
required executive sections with an NVIDIA-grade visual identity plus a live
interactive CEO chatbox. The dashboard only READS precomputed results for sections
00-07; section 08 calls the LLM live via Ollama for interactive Q&A.

Run from the project root:
    streamlit run app.py
"""
from __future__ import annotations

import html as _html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config

st.set_page_config(page_title=f"{config.COMPANY_NAME} — Strategic Intelligence",
                   layout="wide", initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------- styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

:root{
  --bg:#0B0C0E; --surface:#15171B; --surface2:#1C1F24; --border:#2A2D34;
  --text:#F4F6F8; --muted:#9BA1AB; --nv:#76B900; --nv-dim:#5C8F00;
  --hi:#E0533D; --mid:#E6A817; --lo:#76B900;
}
.stApp{ background:var(--bg); color:var(--text); font-family:'Inter',sans-serif; }
[data-testid="stHeader"], #MainMenu, footer{ display:none; }
.block-container{ max-width:1120px; padding-top:2.4rem; padding-bottom:5rem; }
a{ color:var(--text); text-decoration:none; border-bottom:1px solid transparent; }
a:hover{ color:var(--nv); border-bottom-color:var(--nv); }

/* hero */
.hero{ border-top:3px solid var(--nv); padding-top:1.4rem; margin-bottom:2.6rem; }
.eyebrow{ font-family:'Space Grotesk'; letter-spacing:.28em; font-size:.72rem;
  color:var(--nv); font-weight:500; text-transform:uppercase; margin-bottom:.6rem; }
.hero-title{ font-family:'Space Grotesk'; font-weight:700; font-size:3.1rem;
  line-height:1.02; letter-spacing:-.02em; margin:0; }
.hero-title .grn{ color:var(--nv); }
.hero-sub{ color:var(--muted); font-size:.92rem; margin-top:.7rem; }

/* section headers */
.sec{ display:flex; align-items:baseline; gap:.8rem; margin:2.8rem 0 .3rem; }
.sec-num{ font-family:'Space Grotesk'; color:var(--nv); font-weight:700; font-size:1rem; }
.sec-title{ font-family:'Space Grotesk'; font-weight:700; font-size:1.5rem; letter-spacing:-.01em; }
.sec-sub{ color:var(--muted); font-size:.85rem; margin-bottom:1rem; }

/* metric row */
.metric-row{ display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:.6rem 0 1rem; }
.metric{ background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:1.1rem 1.2rem; }
.metric-label{ color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; }
.metric-value{ font-family:'Space Grotesk'; font-weight:700; font-size:1.7rem; margin-top:.3rem; }
.metric-value.grn{ color:var(--nv); }
.chips{ margin:.2rem 0 .4rem; }
.chip{ display:inline-block; background:var(--surface2); border:1px solid var(--border);
  color:var(--muted); border-radius:999px; padding:.22rem .7rem; font-size:.76rem; margin-right:.5rem; }
.chip b{ color:var(--text); }

/* legend */
.legend{ background:var(--surface); border:1px solid var(--border); border-left:3px solid var(--nv);
  border-radius:12px; padding:.9rem 1.1rem; font-size:.82rem; color:var(--muted); margin:.4rem 0 1.4rem; }
.legend b{ color:var(--text); }
.legend .sw{ display:inline-block; width:10px; height:10px; border-radius:3px; margin:0 .25rem 0 .6rem; vertical-align:middle; }

/* news list */
.news{ border-bottom:1px solid var(--border); padding:.6rem 0; font-size:.92rem; }
.news .date{ color:var(--nv); font-family:'Space Grotesk'; font-size:.78rem; margin-right:.7rem; }

/* finding cards */
.card{ background:var(--surface); border:1px solid var(--border); border-radius:14px;
  padding:1.1rem 1.25rem; margin-bottom:.85rem; transition:border-color .15s; }
.card:hover{ border-color:var(--nv-dim); }
.card-top{ display:flex; justify-content:space-between; align-items:center; gap:1rem; }
.card-title{ font-family:'Space Grotesk'; font-weight:500; font-size:1.06rem; }
.card-summary{ color:#D7DBE0; font-size:.92rem; margin:.55rem 0 .7rem; line-height:1.5; }
.card-foot{ display:flex; gap:1.1rem; align-items:center; font-size:.76rem; color:var(--muted); margin-bottom:.5rem; }
.ent{ font-family:'Space Grotesk'; }
.pill{ font-size:.7rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em;
  color:var(--pc); border:1px solid var(--pc); border-radius:999px; padding:.16rem .6rem; white-space:nowrap; }
.meter{ height:6px; background:var(--surface2); border-radius:999px; overflow:hidden; }
.meter-fill{ height:100%; border-radius:999px; }
.meter-label{ display:flex; justify-content:space-between; font-size:.74rem; color:var(--muted);
  margin-top:.35rem; font-family:'Space Grotesk'; }
.evidence{ margin-top:.7rem; padding-top:.7rem; border-top:1px solid var(--border); font-size:.84rem; line-height:1.9; }
.ev{ margin-right:.35rem; }
.srctag{ font-family:'Space Grotesk'; font-size:.68rem; color:var(--nv); background:rgba(118,185,0,.09);
  border:1px solid rgba(118,185,0,.25); border-radius:6px; padding:.05rem .4rem; margin-right:.8rem; }

/* recommendations */
.rec{ background:var(--surface); border:1px solid var(--border); border-left:3px solid var(--nv);
  border-radius:12px; padding:1.1rem 1.25rem; margin-bottom:.85rem; }
.rec-top{ display:flex; justify-content:space-between; align-items:center; gap:1rem; }
.rec-title{ font-family:'Space Grotesk'; font-weight:700; font-size:1.12rem; }
.rec-meta{ color:var(--muted); font-size:.76rem; margin:.3rem 0 .6rem; }
.rec-body{ color:#D7DBE0; font-size:.92rem; line-height:1.5; }
.rec-impact{ color:var(--nv); font-size:.82rem; margin-top:.6rem; }

/* briefing */
.brief{ background:var(--surface); border:1px solid var(--border); border-radius:14px;
  padding:1.2rem 1.3rem; margin-bottom:.85rem; }
.brief-h{ font-family:'Space Grotesk'; color:var(--nv); font-weight:700; font-size:.8rem;
  text-transform:uppercase; letter-spacing:.14em; margin-bottom:.5rem; }
.brief-b{ color:#D7DBE0; font-size:.95rem; line-height:1.6; }

/* agent reasoning */
.goal-box{ background:var(--surface); border:1px solid var(--border); border-left:3px solid var(--nv);
  border-radius:12px; padding:1rem 1.2rem; margin-bottom:1rem; }
.goal-box .lbl{ font-family:'Space Grotesk'; color:var(--nv); font-weight:700; font-size:.72rem;
  text-transform:uppercase; letter-spacing:.16em; }
.goal-box .txt{ color:#D7DBE0; font-size:.95rem; line-height:1.5; margin-top:.35rem; }
.agent-meta{ display:flex; gap:.5rem; flex-wrap:wrap; margin-bottom:1rem; }
.agent-meta .tag{ background:var(--surface2); border:1px solid var(--border); border-radius:999px;
  padding:.28rem .7rem; color:var(--muted); font-size:.74rem; }
.agent-meta .tag b{ color:var(--text); }
.plan-grid{ display:grid; grid-template-columns:repeat(2,1fr); gap:.7rem; margin-bottom:1.1rem; }
.plan-card{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:.85rem 1rem; }
.plan-card .ph{ display:flex; justify-content:space-between; align-items:center; gap:.5rem; }
.plan-card .pa{ font-family:'Space Grotesk'; font-weight:700; font-size:.96rem; }
.plan-card .pt{ font-size:.66rem; text-transform:uppercase; letter-spacing:.1em; padding:.16rem .5rem;
  border-radius:999px; border:1px solid var(--border); color:var(--muted); }
.plan-card .pf{ color:#D7DBE0; font-size:.84rem; margin:.35rem 0 .4rem; }
.plan-card .pq{ color:var(--muted); font-size:.76rem; }
.plan-card .pr{ color:var(--nv); font-size:.74rem; margin-top:.4rem; font-style:italic; }
.trace{ border:1px solid var(--border); border-radius:12px; overflow:hidden; }
.trace-row{ display:flex; gap:.8rem; align-items:flex-start; padding:.6rem .9rem;
  border-bottom:1px solid var(--border); background:var(--surface); }
.trace-row:last-child{ border-bottom:none; }
.stage{ font-family:'Space Grotesk'; font-weight:700; font-size:.62rem; text-transform:uppercase;
  letter-spacing:.1em; color:var(--nv); min-width:74px; padding-top:.12rem; }
.trace-d{ color:var(--text); font-size:.84rem; }
.trace-r{ color:var(--muted); font-size:.78rem; margin-top:.1rem; }
.val{ font-size:.66rem; text-transform:uppercase; letter-spacing:.08em; padding:.18rem .55rem;
  border-radius:999px; font-weight:600; }
.val.validated{ color:#0B0C0E; background:var(--nv); }
.val.weak{ color:#0B0C0E; background:var(--mid); }
.rec-evidence{ color:var(--muted); font-size:.78rem; margin-top:.55rem; }
.rec-evidence a{ color:var(--muted); }
.rec-evidence a:hover{ color:var(--nv); }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------- data
RESULTS = Path(config.RESULTS_DIR) / "analysis.json"
if not RESULTS.exists():
    st.error("No analysis found. Run `python scripts/run_agent.py` first.")
    st.stop()
R = json.loads(RESULTS.read_text())


def esc(s):
    return _html.escape(str(s if s is not None else ""))


def conf_color(c):
    return "#76B900" if c >= 0.7 else ("#E6A817" if c >= 0.4 else "#E0533D")


def impact_color(v):
    return {"High": "#E0533D", "Medium": "#E6A817", "Low": "#76B900"}.get(v, "#9BA1AB")


def section(num, title, sub=""):
    st.markdown(f'<div class="sec"><span class="sec-num">{num:02d}</span>'
                f'<span class="sec-title">{esc(title)}</span></div>'
                + (f'<div class="sec-sub">{esc(sub)}</div>' if sub else ""),
                unsafe_allow_html=True)


def finding_card(f) -> str:
    c = f.get("confidence", 0) or 0
    ent = f.get("entailment")
    color = conf_color(c)
    ip = impact_color(f.get("impact", ""))
    ev = ""
    for s in f.get("sources", []):
        title = esc((s.get("title") or s.get("source") or "source")[:72])
        url = esc(s.get("url") or "")
        src = esc(s.get("source") or "")
        link = f'<a class="ev" href="{url}" target="_blank">{title}</a>' if url else f'<span class="ev">{title}</span>'
        ev += f'{link}<span class="srctag">{src}</span>'
    ent_html = f'<span class="ent">entailment {ent:.2f}</span>' if ent is not None else ""
    return f"""
    <div class="card">
      <div class="card-top">
        <div class="card-title">{esc(f.get('title',''))}</div>
        <div class="pill" style="--pc:{ip}">{esc(f.get('impact',''))}</div>
      </div>
      <div class="card-summary">{esc(f.get('summary',''))}</div>
      <div class="card-foot"><span>Analyst · {esc(f.get('analyst',''))}</span>{ent_html}</div>
      <div class="meter"><div class="meter-fill" style="width:{int(round(c*100))}%;background:{color}"></div></div>
      <div class="meter-label"><span>Confidence</span><span style="color:{color}">{c:.2f}</span></div>
      <div class="evidence">{ev}</div>
    </div>"""


# ----------------------------------------------------------------------------- hero
ts = R.get("generated_at", "")[:16].replace("T", " ")
st.markdown(f"""
<div class="hero">
  <div class="eyebrow">AI CEO · Live Strategic Intelligence</div>
  <h1 class="hero-title">{esc(R.get('company',''))} <span class="grn">Strategic Intelligence</span></h1>
  <div class="hero-sub">{esc(R.get('industry',''))} &nbsp;·&nbsp; {R.get('num_documents',0)} documents &nbsp;·&nbsp; {len(R.get('sources',{}))} live sources &nbsp;·&nbsp; generated {esc(ts)} UTC</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------- 0 · Agent Reasoning
# Surfaces the agent's explicit Goal -> Plan -> Decide -> Validate loop so the
# autonomous behaviour is visible, not buried in the terminal.
if R.get("goal") or R.get("plan") or R.get("decisions"):
    section(0, "Agent Reasoning", "How the agent planned, decided and validated this briefing")

    if R.get("goal"):
        st.markdown(f'<div class="goal-box"><div class="lbl">Goal</div>'
                    f'<div class="txt">{esc(R["goal"])}</div></div>', unsafe_allow_html=True)

    mem_tag = "memory of previous run used" if R.get("memory_used") else "first run — no prior memory"
    st.markdown(
        '<div class="agent-meta">'
        f'<span class="tag">planning passes <b>{R.get("iterations",1)}</b></span>'
        f'<span class="tag">analyst tasks <b>{len(R.get("plan",[]))}</b></span>'
        f'<span class="tag">decisions logged <b>{len(R.get("decisions",[]))}</b></span>'
        f'<span class="tag">{esc(mem_tag)}</span>'
        '</div>', unsafe_allow_html=True)

    plan = R.get("plan", [])
    if plan:
        st.markdown('<div class="sec-sub">Plan — the analysts and lines of inquiry the agent chose</div>',
                    unsafe_allow_html=True)
        cards = ""
        for p in plan:
            src = p.get("source_type") or "all sources"
            queries = "  ·  ".join(esc(q) for q in (p.get("queries") or []))
            rat = f'<div class="pr">{esc(p.get("rationale",""))}</div>' if p.get("rationale") else ""
            cards += (f'<div class="plan-card"><div class="ph">'
                      f'<span class="pa">{esc(p.get("analyst",""))}</span>'
                      f'<span class="pt">{esc(p.get("finding_type",""))} · {esc(src)}</span></div>'
                      f'<div class="pf">{esc(p.get("focus",""))}</div>'
                      f'<div class="pq">{queries}</div>{rat}</div>')
        st.markdown(f'<div class="plan-grid">{cards}</div>', unsafe_allow_html=True)

    decisions = R.get("decisions", [])
    if decisions:
        st.markdown('<div class="sec-sub">Decision trace — every autonomous decision, in order</div>',
                    unsafe_allow_html=True)
        rows = ""
        for d in decisions:
            rows += (f'<div class="trace-row"><div class="stage">{esc(d.get("stage",""))}</div>'
                     f'<div><div class="trace-d">{esc(d.get("decision",""))}</div>'
                     f'<div class="trace-r">{esc(d.get("reason",""))}</div></div></div>')
        st.markdown(f'<div class="trace">{rows}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------- 1 · Company Overview
section(1, "Company Overview")
srcs = R.get("sources", {})
st.markdown(
    '<div class="metric-row">'
    + f'<div class="metric"><div class="metric-label">Company</div><div class="metric-value grn">{esc(R.get("company",""))}</div></div>'
    + f'<div class="metric"><div class="metric-label">Industry</div><div class="metric-value">{esc(R.get("industry","—"))}</div></div>'
    + f'<div class="metric"><div class="metric-label">Documents</div><div class="metric-value">{R.get("num_documents",0)}</div></div>'
    + f'<div class="metric"><div class="metric-label">Live sources</div><div class="metric-value">{len(srcs)}</div></div>'
    + '</div>'
    + '<div class="chips">' + "".join(f'<span class="chip">{esc(k)} <b>{v}</b></span>' for k, v in srcs.items()) + '</div>',
    unsafe_allow_html=True)

# --------------------------------------------------------------- 2 · Market Intelligence
section(2, "Market Intelligence", "Most recent developments in the company's environment")
news_html = ""
for n in R.get("recent_news", [])[:10]:
    date = esc((n.get("published_at") or "")[:10])
    title = esc(n.get("title", ""))
    url = esc(n.get("url") or "")
    body = f'<a href="{url}" target="_blank">{title}</a>' if url else title
    news_html += f'<div class="news"><span class="date">{date}</span>{body}</div>'
st.markdown(news_html, unsafe_allow_html=True)

# score legend (shown once, before the scored sections)
st.markdown(
    '<div class="legend">'
    '<b>How to read each card.</b> &nbsp; <b>Confidence</b> = 0.6·entailment + 0.4·corroboration (distinct documents). '
    '<span class="sw" style="background:#76B900"></span><b>≥0.70 strong</b>'
    '<span class="sw" style="background:#E6A817"></span><b>0.40–0.70 moderate</b>'
    '<span class="sw" style="background:#E0533D"></span><b>&lt;0.40 thin</b>. &nbsp; '
    '<b>Entailment</b> (BART-MNLI, 0–1) = does the cited evidence literally support the claim.'
    '</div>', unsafe_allow_html=True)

# --------------------------------------------------------------- 3 · Opportunity Monitor
section(3, "Opportunity Monitor")
opps = sorted(R.get("opportunities", []), key=lambda f: f.get("confidence", 0), reverse=True)
st.markdown("".join(finding_card(f) for f in opps) or '<div class="legend">No opportunities found.</div>',
            unsafe_allow_html=True)

# --------------------------------------------------------------- 4 · Risk Monitor
section(4, "Risk Monitor")
risks = sorted(R.get("risks", []), key=lambda f: f.get("confidence", 0), reverse=True)
st.markdown("".join(finding_card(f) for f in risks) or '<div class="legend">No risks found.</div>',
            unsafe_allow_html=True)

# --------------------------------------------------------------- 5 · Sentiment Analysis
section(5, "Sentiment Analysis", "VADER compound polarity across all collected documents (-1 to +1)")
sent = R.get("sentiment", {})
st.markdown(f'<div class="metric"><div class="metric-label">Overall sentiment</div>'
            f'<div class="metric-value grn">{sent.get("overall",0.0):+.3f}</div></div>',
            unsafe_allow_html=True)

NV = "#76B900"
col_a, col_b = st.columns(2)
try:
    import plotly.graph_objects as go

    def _layout(fig, h=320):
        fig.update_layout(height=h, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#9BA1AB", family="Inter"))
        fig.update_xaxes(gridcolor="#23262C", zerolinecolor="#23262C")
        fig.update_yaxes(gridcolor="#23262C", zerolinecolor="#3A3D44")
        return fig

    with col_a:
        st.markdown('<div class="sec-sub">By source type</div>', unsafe_allow_html=True)
        bt = sent.get("by_source_type", {})
        if bt:
            fig = go.Figure(go.Bar(x=list(bt.keys()), y=[v["mean"] for v in bt.values()],
                                   marker_color=NV, marker_line_width=0))
            st.plotly_chart(_layout(fig), use_container_width=True)
    with col_b:
        st.markdown('<div class="sec-sub">Sentiment over time</div>', unsafe_allow_html=True)
        tl = sent.get("timeline", [])
        if tl:
            fig = go.Figure(go.Scatter(x=[t["date"] for t in tl], y=[t["mean"] for t in tl],
                                       mode="lines", line=dict(color=NV, width=2)))
            st.plotly_chart(_layout(fig), use_container_width=True)
except ImportError:                                   # graceful fallback if plotly missing
    with col_a:
        bt = sent.get("by_source_type", {})
        if bt:
            st.bar_chart(pd.DataFrame([{"source": k, "sentiment": v["mean"]} for k, v in bt.items()]).set_index("source"))
    with col_b:
        tl = sent.get("timeline", [])
        if tl:
            df = pd.DataFrame(tl); df["date"] = pd.to_datetime(df["date"], errors="coerce")
            st.line_chart(df.dropna(subset=["date"]).set_index("date")[["mean"]])

# --------------------------------------------------------------- 6 · Recommendations
section(6, "Strategic Recommendations")
order = {"High": 0, "Medium": 1, "Low": 2}
recs_html = ""
for r in sorted(R.get("recommendations", []), key=lambda r: order.get(r.get("priority"), 3)):
    pr = r.get("priority", "")
    impacts = "  ·  ".join(esc(x) for x in (r.get("expected_impact") or []))
    val = r.get("validation") or {}
    verdict = val.get("verdict", "")
    val_html = ""
    if verdict:
        val_html = (f'<span class="val {esc(verdict)}">{esc(verdict)}'
                    f' · {val.get("support_score",0):.2f}</span>')
    ev = val.get("evidence") or []
    ev_html = ""
    if ev:
        links = "  ·  ".join(
            f'<a href="{esc(s.get("url","#"))}" target="_blank">{esc(s.get("title","source"))}</a>'
            for s in ev[:5])
        ev_html = f'<div class="rec-evidence">Validated against · {links}</div>'
    recs_html += f"""
    <div class="rec">
      <div class="rec-top"><div class="rec-title">{esc(r.get('recommendation',''))}</div>
        <div style="display:flex;gap:.5rem;align-items:center">{val_html}
        <div class="pill" style="--pc:{impact_color(pr)}">{esc(pr)} priority</div></div></div>
      <div class="rec-meta">Risk level · {esc(r.get('risk_level','—'))}</div>
      <div class="rec-body">{esc(r.get('rationale',''))}</div>
      {f'<div class="rec-impact">Expected impact · {impacts}</div>' if impacts else ''}
      {ev_html}
    </div>"""
st.markdown(recs_html or '<div class="legend">No recommendations generated.</div>', unsafe_allow_html=True)

# --------------------------------------------------------------- 7 · CEO Briefing
section(7, "CEO Briefing")
b = R.get("briefing", {})
st.markdown(
    f'<div class="brief"><div class="brief-h">What happened</div><div class="brief-b">{esc(b.get("what_happened","—"))}</div></div>'
    f'<div class="brief"><div class="brief-h">Why it matters</div><div class="brief-b">{esc(b.get("why_it_matters","—"))}</div></div>'
    f'<div class="brief"><div class="brief-h">What management should do next</div><div class="brief-b">{esc(b.get("what_to_do_next","—"))}</div></div>',
    unsafe_allow_html=True)

# --------------------------------------------------------------- 8 · Ask the CEO Agent
section(8, "Ask the CEO Agent")
st.markdown('<div class="sec-sub">Ask a strategic question — the agent retrieves live evidence and answers as NVIDIA\'s AI CEO advisor</div>',
            unsafe_allow_html=True)

# chat history in session state
if "ceo_messages" not in st.session_state:
    st.session_state.ceo_messages = []

# display past messages
for msg in st.session_state.ceo_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# chat input
if user_q := st.chat_input("Ask the CEO agent a strategic question about NVIDIA..."):
    st.session_state.ceo_messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and reasoning..."):
            try:
                from src.retrieval import HybridRetriever
                from src.llm import chat_text

                @st.cache_resource
                def _load_retriever():
                    return HybridRetriever()

                retriever = _load_retriever()
                chunks = retriever.query(user_q, k=8)

                # build evidence context from retrieved chunks
                evidence_lines = []
                for i, c in enumerate(chunks, 1):
                    src_title = c.get("title", c.get("source", "unknown"))
                    evidence_lines.append(f"[{i}] ({src_title}): {c['text'][:400]}")
                evidence_block = "\n".join(evidence_lines)

                # include a compact summary of the latest analysis for context
                top_findings = sorted(R.get("opportunities", []) + R.get("risks", []) + R.get("trends", []),
                                      key=lambda f: f.get("confidence", 0), reverse=True)[:10]
                analysis_summary = "\n".join(
                    f"- [{f.get('type')}] {f.get('title')} (confidence {f.get('confidence', 0):.2f})"
                    for f in top_findings)

                recs_summary = "\n".join(
                    f"- {r.get('recommendation')}" for r in R.get("recommendations", []))

                system = (
                    f"You are the AI CEO strategic advisor for {config.COMPANY_NAME}. "
                    "Answer the user's question using ONLY the retrieved evidence and analysis provided below. "
                    "Cite evidence by number [1], [2] etc. If the evidence doesn't cover the question, say so. "
                    "Be concise, strategic, and actionable — you are advising the CEO.\n\n"
                    f"CURRENT ANALYSIS SUMMARY:\nTop findings:\n{analysis_summary}\n\n"
                    f"Current recommendations:\n{recs_summary}"
                )

                prompt = f"RETRIEVED EVIDENCE:\n{evidence_block}\n\nQUESTION: {user_q}"

                answer = chat_text(prompt, system=system)
                st.markdown(answer)
                st.session_state.ceo_messages.append({"role": "assistant", "content": answer})

                # show sources used
                with st.expander("📎 Evidence sources used"):
                    for i, c in enumerate(chunks[:5], 1):
                        title = c.get("title", "source")
                        url = c.get("url", "")
                        src = c.get("source", "")
                        if url:
                            st.markdown(f"[{i}] [{esc(title)}]({url})  `{esc(src)}`")
                        else:
                            st.markdown(f"[{i}] {esc(title)}  `{esc(src)}`")
            except Exception as e:
                err_msg = f"⚠️ Could not reach the LLM. Make sure Ollama is running with `{config.LLM_MODEL}` loaded.\n\n`{e}`"
                st.warning(err_msg)
                st.session_state.ceo_messages.append({"role": "assistant", "content": err_msg})
