# EAAP — Enterprise Agent Application Platform (Core Implementation)

A working core implementation of the **EAAP** product specification
(enterprise agent development & runtime platform): agents are defined as
versioned node graphs, executed by a persistent state-machine engine with
per-node checkpoints, human-in-the-loop approvals, a model gateway with
fallback and rate limiting, a validated tool registry, hybrid knowledge
retrieval, and an evaluation center — all behind a multi-tenant REST API.

> **Scope, honestly.** The full PRD describes a 27-week, multi-service
> enterprise build (visual canvas, Firecracker sandbox, MCP fleet, K8s
> operations). This repository implements the **architectural spine** of
> that design as a single runnable FastAPI service, and the
> [FR coverage table](#fr-coverage) below marks every requirement as
> Implemented / Partial / Roadmap. Nothing here is a mockup: the engine,
> persistence, retrieval and evals all run for real, offline, against a
> deterministic mock LLM provider (or a real OpenAI-compatible endpoint
> when configured).

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m app.seed          # demo tenant, KB, 2 agents, eval set
.venv/bin/uvicorn app.main:app --port 8077
```

Demo API key: `eaap-demo-key` (tenant is derived from the key — never from
the request body).

```bash
# Run the seeded contract Q&A agent (retrieval-grounded)
curl -X POST localhost:8077/v1/agents/<agent_id>/runs \
  -H "Authorization: Bearer eaap-demo-key" -H 'Content-Type: application/json' \
  -d '{"input": {"question": "How many days do we have to pay an invoice?"}}'

# Ask something the KB cannot answer -> the run pauses for legal review
#   GET  /v1/approvals                       -> pending approval
#   POST /v1/approvals/<id>/decision          {"decision": "approved"}
#   GET  /v1/runs/<run_id>/trace              -> full node/event trace
#   GET  /v1/runs/<run_id>/events             -> SSE stream (Last-Event-ID ok)
```

Tests and live smoke test:

```bash
.venv/bin/python -m pytest tests/ -q   # 27 tests
bash smoke_test.sh                     # real uvicorn + curl, writes smoke_output.txt
```

## Architecture

Control plane vs. runtime plane are separate modules (a control-plane
failure never touches a running execution, per the PRD's design principles):

```
app/
  main.py       REST API (§7.2 core set) + SSE (§7.3) + HTML trace viewer
  agents.py     Agent lifecycle: draft -> validate -> publish (immutable versions)
  graph.py      DSL validation: cycles (controlled loops only), orphans, config
  engine.py     Runtime: persistent state machine, checkpoints, HITL, parallel
  providers.py  Model gateway: provider abstraction, fallback chain, token
                bucket rate limit, token counting + cost estimation
  tools.py      Tool registry: JSON-Schema arg validation, redacted audit
  knowledge.py  KB ingestion, chunking, hybrid retrieval (vector + BM25, RRF)
  evals.py      Datasets, rule evaluators, batch re-runs, version comparison
  guard.py      Prompt-injection pattern guard (extra, cf. FR-1006)
  db.py         SQLite persistence (PRD §6.3 tables) + hash-chained audit log
  expr.py       Sandboxed expression language + {{template}} rendering
  seed.py       Demo tenant / KB / agents / eval dataset
```

**Graph DSL** — exactly the PRD §7.2 shape. Node types: `start`, `end`,
`llm`, `knowledge_retrieval`, `tool`, `condition`, `code`, `human_review`,
`variable_assign`, `http_request`, `parallel` (fan-out branches), `loop`
(controlled iteration with `max_iterations`). Edges carry `when`
conditions — literal (`"true"`/`"false"`/decision strings) or expressions
over the execution context (`nodes.llm.confidence < 0.6`, `vars.counter`,
`input.question`), evaluated by a whitelisted AST interpreter — no `eval`.

**Crash recovery.** After every node, the engine writes a checkpoint (pc +
full context + loop counters) and a node-execution row with idempotency key
`run_id:node_id:attempt`. Kill the worker mid-run and `POST
/v1/runs/{id}/resume` continues from the last completed node without
re-executing it (covered by a dedicated crash-simulation test).

**HITL.** A `human_review` node creates an approval record and parks the
run in `waiting_approval`. Approve / reject / modify via the approvals API
resumes from the checkpoint and routes the branch by decision.

**Knowledge.** Documents are chunked with overlap and embedded with a
deterministic hashed bag-of-words embedder (offline; swappable). Retrieval
fuses vector cosine and BM25 rankings with Reciprocal Rank Fusion, returns
citations (`[DocName#chunk]`), supports a relevance floor, and is hard
scoped by tenant.

**Evals.** Datasets of cases with rule evaluators (`contains`,
`not_contains`, `regex`, `json_schema`) are re-run in batch against any
published agent version; reports include pass rate, average latency and
token cost, and two eval runs can be diffed (metric deltas + per-case pass
flips) — the data you need for a regression gate.

**Extras (innovation).** A built-in HTML trace viewer (`GET /ui/runs/{id}`)
and a prompt-injection guard that flags suspicious tool arguments / LLM
output in the audit log and on node outputs.

## Demo agents (seeded)

1. **contract-qa** — the PRD §7.2 example: `start → knowledge_retrieval →
   llm → condition(confidence < 0.6) → human_review(legal) → end`, with a
   three-document contracts KB. KB-covered questions answer with citations;
   off-topic questions drop to low confidence and pause for legal review.
2. **report-generator** — `variable_assign → parallel(text_stats +
   current_time) → llm → end`, showing fan-out tool calls and templated,
   structured run output.

## API (core set)

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/agents` | Create agent |
| PUT | `/v1/agents/{id}/draft` | Save draft graph (DSL) |
| POST | `/v1/agents/{id}/validate` | Validate draft graph |
| POST | `/v1/agents/{id}/publish` | Publish immutable version (validates first) |
| POST | `/v1/agents/{id}/runs` | Create + start run |
| GET | `/v1/runs/{run_id}` | Run state and result |
| POST | `/v1/runs/{run_id}/cancel` | Cancel a run |
| POST | `/v1/runs/{run_id}/resume` | Resume from last checkpoint |
| GET | `/v1/runs/{run_id}/trace` | Node executions, events, checkpoints |
| GET | `/v1/runs/{run_id}/events` | SSE event stream |
| GET/POST | `/v1/approvals`, `/v1/approvals/{id}/decision` | HITL queue and decisions |
| POST | `/v1/knowledge-bases`, `/{id}/documents`, `/{id}/search` | KB management + hybrid search |
| GET/POST | `/v1/tools` | Tool registry / direct tool invocation |
| POST | `/v1/evals/datasets`, `/v1/evals/runs` | Eval datasets and batch runs |
| GET | `/v1/evals/runs/{id}`, `/v1/evals/compare?a=&b=` | Eval report and version diff |
| GET | `/v1/usage` | Tenant token/cost usage |

## FR coverage

Statuses: **Implemented** = works and is tested · **Partial** = core works,
PRD-grade hardening pending · **Roadmap** = designed, not built here.

| Area | Status | Notes |
|---|---|---|
| FR-101 visual canvas | Roadmap | DSL-first core; the graph JSON is the canvas's data model |
| FR-102 node types | Implemented | 11 of 12 types; sub-graph node is Roadmap (FR-207) |
| FR-103 graph validation | Implemented | Cycle/orphan/config checks; invalid graphs cannot publish |
| FR-104 variables & expressions | Implemented | Sandboxed expression subset, typed templates |
| FR-105 real-time debugging | Partial | Full trace/events/checkpoints; no breakpoints or step-into |
| FR-106 versioning | Implemented | Immutable versions; any version runnable by id |
| FR-107 collaboration | Roadmap | |
| FR-108 DSL import/export | Partial | Graph is JSON DSL via API; Git sync Roadmap |
| FR-109 AI graph generation | Roadmap | |
| FR-201 checkpoints & resume | Implemented | Checkpoint after every node; crash-resume tested |
| FR-202 concurrency | Implemented | Parallel branch fan-out with join |
| FR-203 timeout & retry | Partial | Per-node retry with backoff; timeouts on code/http/tool nodes |
| FR-204 idempotency | Implemented | `run:node:attempt` key on every node execution |
| FR-205 streaming | Partial | SSE run/node events with Last-Event-ID replay; token deltas Roadmap |
| FR-206 cancel & pause | Implemented | Cancel flag honored between nodes; HITL pause |
| FR-207 sub-graphs | Roadmap | |
| FR-208 scheduling fairness | Roadmap | |
| FR-209 long runs | Partial | Durable state machine; not soak-tested for 7 days |
| FR-301 providers | Partial | OpenAI-compatible + mock; native Anthropic/domestic protocols Roadmap |
| FR-302 capability tags | Roadmap | |
| FR-303 routing & fallback | Implemented | Primary → fallback chain, tested |
| FR-304 rate limiting | Implemented | Per-tenant token bucket (429-style `RATE_LIMITED`) |
| FR-305 semantic cache | Roadmap | |
| FR-306 structured output | Partial | JSON extraction helper + schema evaluator; no auto-repair loop |
| FR-307 credential handling | Partial | API keys stored hashed; provider keys via env, never logged |
| FR-308 model canary | Roadmap | |
| FR-401 tool registry | Implemented | Builtins; custom tools register in code |
| FR-402 OpenAPI import | Roadmap | |
| FR-403 MCP client | Roadmap | |
| FR-404 argument validation | Implemented | Schema-checked before execution |
| FR-405 tool auth | Partial | Tenant scoping enforced; per-tool OAuth Roadmap |
| FR-406 tool audit | Implemented | Full calls audited with redacted args |
| FR-407 circuit breaker | Implemented | Per-tool breaker: N consecutive failures pause the tool (CIRCUIT_OPEN), half-open trial after cooldown, reset via POST /v1/tools/{name}/reset; env EAAP_TOOL_BREAKER_THRESHOLD / EAAP_TOOL_BREAKER_COOLDOWN_SEC |
| FR-501–505 code sandbox | Partial | Subprocess with stripped env, timeout, temp cwd — **not** microVM isolation; do not run hostile code |
| FR-601 doc parsing | Partial | Text/Markdown ingestion; PDF/OCR Roadmap |
| FR-602 chunking | Implemented | Paragraph-aware with overlap |
| FR-603 hybrid retrieval | Implemented | Vector + BM25 fused with RRF |
| FR-604 reranking | Roadmap | |
| FR-605 citations | Implemented | `[Doc#chunk]` on every result |
| FR-606 incremental indexing | Roadmap | |
| FR-607 permission filtering | Partial | Hard tenant isolation; document-level ACL Roadmap |
| FR-701–703 multi-agent | Roadmap | |
| FR-704 loop termination | Implemented | `max_iterations` + global step cap |
| FR-801 approvals | Implemented | Approve/reject/modify drives branching |
| FR-802 notifications | Partial | Via API + SSE events; email/IM/webhook Roadmap |
| FR-803 approval timeouts | Roadmap | |
| FR-804 feedback → dataset | Roadmap | |
| FR-901 trace | Implemented | run → node → llm/tool, tokens, latency, cost |
| FR-902 datasets | Implemented | Stored, versioned field, JSONL import |
| FR-903 evaluators | Partial | Rule evaluators done; LLM-as-judge / human labels Roadmap |
| FR-904 batch re-runs | Implemented | Synchronous batch over any version |
| FR-905 version comparison | Implemented | Metric deltas + per-case flips |
| FR-906 regression gate | Partial | Compare API exists; publish is not blocked by it yet |
| FR-907 alerting | Roadmap | |
| FR-1001 multi-tenancy | Implemented | tenant_id enforced on every query; isolation tested |
| FR-1002 RBAC | Partial | Approval roles recorded; full permission matrix Roadmap |
| FR-1003 SSO | Roadmap | |
| FR-1004 key management | Partial | Hashing at rest; KMS envelope Roadmap |
| FR-1005 content safety | Partial | Sensitive-value redaction in audit; PII detection Roadmap |
| FR-1006 injection defense | Partial | Pattern guard flags tool args / LLM output (extra) |
| FR-1007 audit log | Implemented | Append-only with SHA-256 hash chain |
| FR-1101 token metering | Implemented | Per run/node/model, aggregated in `/v1/usage` |
| FR-1102 cost attribution | Partial | Tenant → run → node; agent/version rollups Roadmap |
| FR-1103 budgets & quotas | Partial | Rate limiting done; budget stop policies Roadmap |
| FR-1104 billing export | Roadmap | |
| FR-1201–1205 environments/ops | Roadmap | Single-service deployment; SQLite backup = file copy |
| FR-1301 REST API | Implemented | This service |
| FR-1302 SSE | Implemented | |
| FR-1303 SDKs | Roadmap | |
| FR-1304 webhooks | Roadmap | Events are persisted and streamable; push delivery not built |
| FR-1305 embedding | Roadmap | |
| FR-1401 templates | Partial | Two seeded agents act as starter templates |
| FR-1402/1403 template market | Roadmap | |

## Design decisions

- **Persistent state machine over in-memory DAG walking** — the only way
  HITL pauses and crash recovery compose honestly.
- **SQLite with a single locked connection** — zero-ops for a core build;
  the schema mirrors PRD §6.3 so tables map 1:1 to the production design.
- **Deterministic mock provider** — demos and tests must be reproducible
  offline; its confidence convention (grounded ⇒ 0.92, ungrounded ⇒ 0.30)
  is what drives the demo's confidence gate.
- **Hashed embeddings** — real vector math with zero dependencies; the
  embedder is one function, ready to swap for a model-backed one.
- **Tenant from the API key, always** — request bodies can never name a
  tenant (PRD §7.1), and every query is tenant-scoped.
