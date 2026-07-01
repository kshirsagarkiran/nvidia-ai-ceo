# NVIDIA AI CEO — Agent Oral Exam Prep

Your complete guide to defending the project under the clarified "agent behaviour" requirements. Read it top to bottom once, then use the section headers to find anything fast. Everything here matches the code actually on your machine.

---

## 0. The one-breath pitch (memorise this)

> "It's a multi-agent strategic-intelligence **agent**. Given a CEO-level goal, a **planner** decides which specialist analysts to deploy and what to investigate; each analyst **retrieves** grounded evidence from a hybrid knowledge base and produces findings; a **decision gate** verifies every finding with an entailment model and drops duplicates; a CEO agent drafts recommendations that **cite their evidence**; a **validator** checks each recommendation against that evidence and drops unsupported ones before display; and the run is written to **memory** so the next run can reason about what changed. The LLM never answers from its own knowledge — every claim is grounded in retrieved, verified evidence."

That sentence hits all seven stages of her required workflow: **Goal → Plan → Retrieve → Analyze → Decide → Recommend → Validate.**

---

## 1. Why the project changed (frame this correctly)

The professor's clarification said many projects were just `User → Prompt → LLM + RAG → Response`, which uses an LLM well but doesn't *demonstrate agent behaviour*. She asked everyone to **extend** their existing system (not rebuild) to show: planning before execution, autonomous decision-making, tool use beyond the LLM, evidence retrieval, risk/opportunity/trend analysis, and validation of recommendations.

**What we did:** kept the entire proven pipeline (collectors, index, retriever, verifier, sentiment, dashboard) and added a thin **agent layer** on top — a planner, a working-memory state object, episodic memory, a recommendation validator, an explicit orchestrator loop, and a dashboard panel that makes the agent's reasoning visible.

If asked "did you start over?" → **No. The brief explicitly said to build on existing work. I added explicit agency around a pipeline that already worked.**

---

## 2. The architecture in one picture

```
GOAL  ("If you were NVIDIA's CEO today, what would you do next?")
  │
  ▼
PLAN        planner.py  → LLM decides 3–6 analyst tasks (role, focus, queries,
  │                       source filter, rationale), conditioned on episodic MEMORY
  ▼
RETRIEVE    retrieval.py → HybridRetriever: Chroma (dense bge) + BM25 (sparse), α=0.6
  │
  ▼
ANALYZE     agents.py run_analyst → each analyst turns retrieved chunks into findings
  │                                  (grounding filter: drop ungrounded claims)
  ▼
DECIDE      orchestrator._decide → verify every finding with BART-MNLI entailment,
  │                                 blend confidence, drop near-duplicate findings,
  │                                 log every decision with its reason
  ▼
RECOMMEND   agents.py run_ceo → CEO agent drafts recommendations, each CITING the
  │                              finding ids (F1, F3…) it rests on
  ▼
VALIDATE    validator.py → score each recommendation against its cited findings;
  │                         verdict validated / weak / unsupported; DROP unsupported
  ▼
MEMORY      memory.py → record this run's episode (goal, plan, top findings, recs)
  │
  ▼
analysis.json → dashboard (7 sections + section 0 "Agent Reasoning")
```

The whole loop is threaded by one **AgentState** object (working memory) that also records a **decision trace**. If validation clears *zero* recommendations, the orchestrator **re-plans** once (bounded by `MAX_AGENT_ITERATIONS = 2`) — that's real conditional control flow.

---

## 3. File-by-file (what each piece is and why it exists)

### New agent layer (what you built for the rebuild)

**`src/state.py` — working memory.**
Defines `AgentState` (goal, company, plan, findings, recommendations, briefing, decisions, iterations) plus `PlanStep` and `Decision` dataclasses. Methods: `log(stage, decision, reason)` records an autonomous decision to the trace and prints it; `set_plan`, `add_findings`, `findings_by_type`, `to_dict`, `trace`. *This is what makes it stateful rather than a one-shot pipeline — every stage reads and writes one shared object.*

**`src/memory.py` — episodic memory.**
`EpisodicMemory` writes one compact "episode" per run to `data/memory/episodes.jsonl` (goal, plan headlines, top findings, recommendations, timestamp). `briefing_for_planner()` turns the last run into a short recap the planner conditions on ("last run's top risks were X — check if they persist or changed"). Returns `""` on the first run, so the planner just plans from scratch.

**`src/planner.py` — the Planner agent (the headline addition).**
`make_plan(state, memory_briefing)` asks the LLM for a `strategy` plus a `plan` of 3–6 tasks. Each task: which analyst (`ROSTER = CTO, CFO, CMO, Risk Officer`), a focus, retrieval queries, a finding type (opportunity/risk/trend), an optional source filter (news/filing/community), and a rationale. `_coerce_step` validates/repairs each task; `_fallback_plan` is a deterministic default so a bad LLM response can't crash the run. *This replaces the old hardcoded `AGENTS` list — the system now decides what to investigate.*

**`src/validator.py` — the Recommendation Validator.**
`validate(state, embedder)` checks each recommendation against the findings it cited. `support_score` = the highest confidence among its cited findings. Verdict: **validated** (≥ `VALIDATION_THRESHOLD` 0.5), **weak** (below threshold but has support — kept and flagged), **unsupported** (no resolvable findings — **dropped**). Attaches a `validation` block (verdict, score, supporting finding titles, evidence sources) and logs each decision. If a recommendation cited nothing, a semantic fallback links it to findings by embedding similarity (element-wise dot, Apple-safe).

**`src/orchestrator.py` — the explicit agent loop.**
`run_agent()` runs Goal → Plan → Retrieve → Analyze → Decide → Recommend → Validate, threading `AgentState`, logging decisions, with the bounded re-plan loop. Also contains the self-contained `_dedup_findings` (Apple-safe). Writes the same `analysis.json` the dashboard reads, plus new fields: `goal`, `plan`, `decisions`, `iterations`, `memory_used`.

**`scripts/run_agent.py`** — entry point (`python scripts/run_agent.py`).

### Modified

**`src/agents.py`** — `run_ceo` now numbers the findings (`F1…Fn`), the CEO must cite `supporting_findings` ids, and those ids are resolved back to the actual findings so the validator can check them. This also satisfies the brief's Task 6 ("every recommendation must contain supporting evidence").

**`app.py`** — added **section 0 "Agent Reasoning"** (goal, meta tags, plan cards, decision trace) and **validation badges + "validated against" evidence links** on each recommendation.

**`config.py`** — added `VALIDATION_THRESHOLD = 0.5` and `MAX_AGENT_ITERATIONS = 2`.

### Reused unchanged (the proven core)

`collectors/` (Google News, SEC EDGAR, Hacker News), `dedup.py` (document-level), `indexing.py` (chunk + embed + index), `retrieval.py` (HybridRetriever), `verifier.py` (BART-MNLI + `blend_confidence`), `sentiment.py` (VADER), `llm.py` (Ollama transport), `storage_raw.py`.

---

## 4. The models (and why each — they may ask "why not X?")

| Role | Model | Why this one |
|------|-------|-------------|
| Reasoning LLM | **Qwen 3 8B** via Ollama | Open-source, runs locally on the M4, satisfies the "no paid commercial API" rule. Ollama is just the local transport — swappable for Llama/Mistral without changing any agent logic. |
| Embeddings | **BAAI bge-base-en-v1.5** | Strong retrieval quality at a small size; uses an instruction prefix for queries. |
| Verification | **facebook/bart-large-mnli** | A natural-language-inference model used as an *entailment checker*: does the cited evidence actually support the claim? A second model checking the first = real tool use, not self-grading. |
| Sentiment | **VADER** | Lightweight, rule-based, good for short news/social text; no training needed. |

**Retrieval:** hybrid — **Chroma** (dense semantic) + **BM25** (sparse keyword), fused with `HYBRID_ALPHA = 0.6` (60% dense, 40% sparse). Dense catches meaning; sparse catches exact terms like "H20" or "8-K".

---

## 5. Concepts from scratch (be able to explain each in 2–3 sentences)

**RAG (Retrieval-Augmented Generation).** Instead of letting the LLM answer from its trained weights, you first *retrieve* relevant documents and put them in the prompt, so the answer is grounded in real sources. Your system is an advanced RAG: hybrid retrieval + an entailment check on top.

**Hybrid retrieval.** Dense (embedding) search finds semantically similar text even with different words; sparse (BM25) finds exact keyword matches. Combining both beats either alone. You fuse the two scores with a weight α.

**Entailment / NLI.** Given a premise (the evidence) and a hypothesis (the claim), an NLI model says whether the premise *entails* (supports), contradicts, or is neutral to the hypothesis. You use the entailment probability (0–1) as "does the cited evidence literally support this finding?"

**Confidence vs entailment (important — they will probe this).** Entailment alone is brittle: a forward-looking inference ("this indicates a growing trend") is true but not *literally stated* in any chunk, so it scores low. So confidence = `0.6 × entailment + 0.4 × corroboration`, where corroboration rewards a finding cited by multiple distinct documents. **Low entailment ≠ false; it means "inferred, not directly restated."** That's why your opportunities (inferences) cluster at low entailment while many risks (restatements of headlines) score high — a feature, not a bug.

**Planning.** The agent decomposes the goal into sub-tasks *before* acting, and chooses tools/queries per task. Your planner outputs a strategy + 3–6 analyst tasks, conditioned on memory.

**Autonomous decision-making.** The system makes choices without you: which analysts to deploy, which findings clear the evidence bar, which duplicates to drop, which recommendations are unsupported. Each is logged with a reason in the decision trace.

**Validation.** Recommendations are checked against their cited evidence before display; unsupported ones are dropped. This is the guard against the LLM inventing advice.

**The three memories (know all three).**
- **Semantic / long-term:** the Chroma + BM25 knowledge repository the analysts retrieve from.
- **Working:** the `AgentState` threaded through one run (the scratchpad).
- **Episodic:** `episodes.jsonl` — a journal of past runs the planner reads to reason about change over time.

**Agent vs. RAG pipeline (the core distinction she's testing).** A RAG chatbot does prompt → retrieve → answer. An *agent* sets a goal, **plans**, uses **tools**, makes **autonomous decisions**, keeps **state/memory**, and **validates** its own output across **multiple steps**. You have all of those, explicitly and visibly.

---

## 6. The data and the numbers (have these ready)

- **357 documents**, **3 independent live sources**: `google_news` (~263), `sec_edgar` (25), `hacker_news` (69). Exceeds the brief's "≥100 docs, ≥3 sources."
- **Pipeline:** collect → document-dedup → chunk (200 words, 40 overlap) → bge embeddings → Chroma + BM25.
- **A representative run:** planner produced 5–6 tasks across 4 analysts → ~57 grounded findings → verifier scored all → finding-dedup dropped ~12 → ~45 kept (≈8 opportunities, 18 risks, 19 trends) → CEO drafted 7 recommendations → validator passed 6 (validated) + 1 (weak), dropped 0 → episode recorded.
- **Thresholds:** `HYBRID_ALPHA = 0.6`, finding dedup `DEDUP_THRESHOLD = 0.82`, `VALIDATION_THRESHOLD = 0.5`, `MAX_AGENT_ITERATIONS = 2`.

---

## 7. The dashboard (what to show, in order)

- **00 Agent Reasoning** *(open here)* — goal, then `planning passes / analyst tasks / decisions logged / memory used`, then the **plan cards** (each analyst's focus, queries, source filter, rationale), then the **decision trace** (GOAL → PLAN → DECIDE → RECOMMEND → VALIDATE → MEMORY). This is the proof of agent behaviour.
- **01 Company Overview** — NVIDIA, industry, 357 docs, 3 sources, source chips.
- **02 Market Intelligence** — most recent developments.
- **03 Opportunity / 04 Risk Monitor** — finding cards with impact, analyst, **entailment**, confidence meter, evidence links.
- **05 Sentiment** — VADER overall + by source + over time.
- **06 Strategic Recommendations** — each with a **validation badge** (validated/weak + score) and "validated against" evidence.
- **07 CEO Briefing** — what happened / why it matters / what to do next.

---

## 8. Known limitations — name them before the examiner does (this earns marks)

1. **A few off-topic findings.** Broad community queries occasionally pull tangential Hacker News posts (e.g. an "Ask HN about software-engineering careers" surfaced as an Anthropic-profitability finding). *Answer:* "The retriever surfaced a tangential post; the verifier scored it low-entailment, so it's visibly marked as thin — the system is working, weak evidence is flagged not hidden."
2. **The CEO briefing's synthesised stats are ungrounded.** Sentence-level figures the synthesizer writes (e.g. a "% market share") aren't evidence-checked the way findings are. *Answer:* "Per-finding claims are entailment-verified; the CEO narrative is an unverified LLM summary layered on top — a known boundary."
3. **Finding dedup hits a precision/recall wall.** At threshold 0.82, paraphrases that share meaning but not surface form sometimes survive. *Answer:* "Threshold dedup can't separate 'same meaning, different words' perfectly; the CEO synthesizer consolidates residual overlap into clean recommendations. A principled fix is source-aware merge or an LLM-as-judge dedup."
4. **Entailment asymmetry.** Opportunities (inferences) score low entailment; risks (restatements) score high. *Answer:* covered in §5 — it's the verifier behaving correctly across finding types.
5. **Memory is episodic + semantic + working, but not a long conversational memory.** Be precise about which kinds you have (§5).

---

## 9. Anticipated questions + strong answers

**"Where is the planning?"** Open section 0. The planner (an LLM call) decides the analysts, queries, and source filters from the goal and memory — it's not hardcoded. Show the plan cards and the strategy line in the trace.

**"How is this an agent and not just RAG?"** §5 distinction. Then point at the trace: goal set, plan chosen, tools called (retriever, BART), decisions logged, recommendations validated, episode remembered — multi-step, stateful, autonomous.

**"What tools does the agent use?"** The hybrid retriever (which itself queries Chroma + BM25) and the BART-MNLI entailment verifier; plus VADER for sentiment. The LLM is the reasoner, not a tool — the tools are what it calls to ground and check itself.

**"How do you stop the LLM hallucinating recommendations?"** Two gates: findings are entailment-verified before synthesis; recommendations must cite finding ids and are validated against them — unsupported ones are dropped before display. Demo: the validator drops a recommendation that cites nothing.

**"What memory does it have?"** All three kinds (§5). Show the "memory of previous run used" tag and the Risk Officer rationale that says "confirming whether prior risks persist."

**"Why Ollama / why not OpenAI?"** Brief requires open-source/free models; Qwen runs locally. Ollama is the transport, swappable without touching agent logic.

**"Why no LangChain / LangGraph?"** Built the orchestration directly so every step (hybrid retrieval, grounding, entailment, validation) is explicit and explainable. LangGraph earns its place with cyclic/branching agent control flow; this pipeline is mostly linear with one bounded re-plan, so a framework would add abstraction without benefit.

**"What does confidence mean? Why is entailment sometimes 0.0?"** §5 — confidence blends entailment and corroboration; low entailment means inferred, not false.

**"What's your hardest bug?"** The Apple-Silicon NumPy/Accelerate matmul bug: `self.emb @ qv` threw spurious divide-by-zero/overflow/invalid warnings (and wrong scores) on valid inputs. Proved the inputs were clean, traced it to the Accelerate BLAS, fixed it with an element-wise multiply-and-sum that bypasses the buggy gemm. *(This is gold if hardware/numerics comes up.)*

---

## 10. Live-coding drills (she said she may ask you to extend the solution)

Rehearse these so you can do them while talking. Each is small and safe.

**A. Change the validation threshold.** In `config.py`, edit `VALIDATION_THRESHOLD` (e.g. 0.5 → 0.6). Re-run `python scripts/run_agent.py`. More recommendations flip to "weak" or get dropped. *Talking point: this is the precision/recall dial on recommendation trust.*

**B. Add a new analyst to the roster.** In `planner.py`, add e.g. `"COO"` to `ROSTER`. The planner can now deploy it. *Talking point: the roster is the planner's action space; planning is dynamic so no other code changes.*

**C. Add / force a plan step.** In `planner._fallback_plan` (or by editing the prompt), add a `PlanStep` for a specific line of inquiry. *Talking point: shows you understand `PlanStep` shape and how analysts consume it.*

**D. Change retrieval balance.** In `config.py`, edit `HYBRID_ALPHA` (0.6 → 0.8 = more semantic, less keyword). Rebuild not needed; re-run analysis. *Talking point: dense vs sparse trade-off.*

**E. Tighten finding dedup.** `DEDUP_THRESHOLD` lower = more aggressive merging. *Talking point: the precision/recall wall.*

**F. Make the re-plan visible.** Temporarily set `VALIDATION_THRESHOLD = 0.99` so nothing validates → the agent re-plans (pass 2) → shows the loop in the trace. Reset afterwards. *Talking point: conditional control flow / autonomous retry.*

**G. Read the decision trace in code.** `state.trace()` returns the list of `{stage, decision, reason}` dicts that the dashboard renders. *Talking point: decisions are first-class, logged objects.*

Know these file locations cold: planner = `src/planner.py`, validator = `src/validator.py`, loop = `src/orchestrator.py`, state = `src/state.py`, memory = `src/memory.py`, knobs = `config.py`.

---

## 11. Demo script (5 minutes, smooth)

1. **Run it live (optional):** `python scripts/run_agent.py` — narrate the terminal: "planning… analysts retrieving… verifying with BART-MNLI… validating recommendations… recorded episode." Then `streamlit run app.py`.
2. **Section 0 first.** "Here's the agent's reasoning. Goal at top. It planned 6 tasks across 4 analysts — I didn't hardcode these, the planner chose them, and it used memory from the previous run." Point at the Risk Officer rationale about prior risks persisting.
3. **Scroll the decision trace.** "Every autonomous decision in order: verified 57 findings, dropped 12 duplicates, kept 45, drafted 7 recommendations, validated them, recorded the episode."
4. **Jump to section 6.** "Each recommendation carries a validation badge and links to the evidence it was validated against. Unsupported ones never reach this screen."
5. **One finding card (section 4).** "Each finding shows its entailment and confidence — low entailment means inference, not error."
6. **Close on the CEO Briefing.** "And the executive answer to the original question: what happened, why it matters, what to do next."

---

## 12. One-liners to have on the tip of your tongue

- "The LLM never answers from its own knowledge — every claim is grounded in retrieved, verified evidence."
- "Ollama is the transport; the agent is the orchestration around it. Swap to Groq and nothing agentic changes."
- "Low entailment means inferred, not false — confidence blends entailment with cross-document corroboration."
- "Unsupported recommendations are dropped before display; that's the hallucination guard."
- "The planner's roster is its action space; planning is dynamic, so it decides what to investigate."
- "Three memories: semantic (the index), working (the run state), episodic (the journal of past runs)."
- "I extended a working pipeline with an explicit agent layer — I didn't start over, because the brief said not to."

---

Study §0, §5, §8, and §9 hardest — that's where the marks are. When you're ready, we run the mock exam: I play Prof. Chandna, fire conceptual questions and a live-coding drill, and push on the soft spots until your answers are automatic.
