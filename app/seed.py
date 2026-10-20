"""Demo seed: one tenant + API key, a contracts knowledge base, two agents
(PRD §7.2 contract Q&A shape; a report generator using parallel + tools),
and an eval dataset. Run:  python -m app.seed
Idempotent: existing objects are reused.
"""
import hashlib

from . import agents as agents_mod
from . import db, evals, knowledge

DEMO_TENANT = "tenant_demo"
DEMO_KEY = "eaap-demo-key"

KB_DOCS = [
    ("MSA Payment Terms",
     "Payment terms: all invoices are due within 30 days of the invoice "
     "date. Late payments accrue a fee of 1.5 percent per month. "
     "Payment is made by bank transfer to the account stated on the invoice."),
    ("Warranty Policy",
     "Warranty: all delivered software is covered by a 12 month warranty "
     "from the acceptance date. During the warranty period, defects are "
     "fixed free of charge within 5 business days of a confirmed report."),
    ("Liability and Support",
     "Liability: total liability under this agreement is capped at the "
     "fees paid in the 12 months preceding the claim. Support hours are "
     "09:00 to 18:00 China Standard Time, Monday to Friday, excluding "
     "public holidays."),
]

CONTRACT_QA_GRAPH = {
    "nodes": [
        {"id": "start", "type": "start"},
        {"id": "kb", "type": "knowledge_retrieval",
         "config": {"kb_ids": ["contracts"], "top_k": 5, "min_score": 0.2}},
        {"id": "llm", "type": "llm",
         "config": {"model": "mock-1", "temperature": 0.2,
                    "prompt": "You are a contract assistant.\n"
                              "SOURCES:\n{{nodes.kb.context}}\n"
                              "QUESTION: {{input.question}}\n"
                              "Answer using only the sources."}},
        {"id": "check", "type": "condition",
         "config": {"expr": "nodes.llm.confidence < 0.6"}},
        {"id": "human", "type": "human_review",
         "config": {"role": "legal",
                    "message": "Low-confidence answer needs legal review: "
                               "{{nodes.llm.text}}"}},
        {"id": "end", "type": "end",
         "config": {"output": "{{nodes.llm.text}}"}},
    ],
    "edges": [
        {"from": "start", "to": "kb"},
        {"from": "kb", "to": "llm"},
        {"from": "llm", "to": "check"},
        {"from": "check", "to": "human", "when": "true"},
        {"from": "check", "to": "end", "when": "false"},
        {"from": "human", "to": "end"},
    ],
}

REPORT_GRAPH = {
    "nodes": [
        {"id": "start", "type": "start"},
        {"id": "prep", "type": "variable_assign",
         "config": {"assignments": {"title": "= 'Report: ' + input.topic"}}},
        {"id": "par", "type": "parallel",
         "config": {"branches": [["stats"], ["now"]]}},
        {"id": "stats", "type": "tool",
         "config": {"tool": "text_stats", "args": {"text": "{{input.text}}"}}},
        {"id": "now", "type": "tool",
         "config": {"tool": "current_time", "args": {}}},
        {"id": "write", "type": "llm",
         "config": {"model": "mock-1",
                    "prompt": "Write a short report titled {{vars.title}}. "
                              "Source statistics: {{nodes.stats}}. "
                              "Generated at {{nodes.now.iso}}."}},
        {"id": "end", "type": "end",
         "config": {"output": {"title": "{{vars.title}}",
                               "report": "{{nodes.write.text}}",
                               "word_count": "{{nodes.stats.words}}"}}},
    ],
    "edges": [
        {"from": "start", "to": "prep"},
        {"from": "prep", "to": "par"},
        {"from": "par", "to": "write"},
        {"from": "write", "to": "end"},
    ],
}

EVAL_CASES = [
    {"input": {"question": "How many days do we have to pay an invoice?"},
     "expected": "30 days"},
    {"input": {"question": "How long is the warranty period?"},
     "expected": "12 month"},
    {"input": {"question": "What is the weather in Tokyo today?"},
     "expected": "human should review"},
]


def seed() -> dict:
    db.init_db()
    existing = db.q1("SELECT * FROM tenants WHERE id=?", (DEMO_TENANT,))
    if existing is None:
        db.x("INSERT INTO tenants (id, name, plan, status, created_at)"
             " VALUES (?,?,?,?,?)",
             (DEMO_TENANT, "Demo Tenant", "pro", "active", db.now_ts()))
        db.x("INSERT INTO api_keys (id, tenant_id, key_hash, scopes, created_at)"
             " VALUES (?,?,?,?,?)",
             (db.new_id("key"), DEMO_TENANT,
              hashlib.sha256(DEMO_KEY.encode()).hexdigest(), '["*"]',
              db.now_ts()))

    kb = knowledge.find_kb_by_name(DEMO_TENANT, "contracts")
    if kb is None:
        kb = knowledge.create_kb(DEMO_TENANT, "contracts")
        for name, content in KB_DOCS:
            knowledge.add_document(DEMO_TENANT, kb["id"], name, content)

    out = {"tenant_id": DEMO_TENANT, "api_key": DEMO_KEY, "kb_id": kb["id"]}
    for name, graph, desc in (
            ("contract-qa", CONTRACT_QA_GRAPH,
             "Contract Q&A with retrieval, confidence gate and legal review"),
            ("report-generator", REPORT_GRAPH,
             "Report generator: parallel tool calls + LLM write-up")):
        row = db.q1("SELECT id FROM agents WHERE tenant_id=? AND name=?",
                    (DEMO_TENANT, name))
        if row:
            agent = agents_mod.get_agent(DEMO_TENANT, row["id"])
        else:
            agent = agents_mod.create_agent(DEMO_TENANT, name, desc)
            agents_mod.save_draft(DEMO_TENANT, agent["id"], graph)
            published = agents_mod.publish(DEMO_TENANT, agent["id"],
                                           "seed v1")
            agent = agents_mod.get_agent(DEMO_TENANT, agent["id"])
            agent["published_version_id"] = published["version_id"]
        out[name] = agent["id"]
        out[f"{name}_version"] = agent.get("current_version_id")

    ds = db.q1("SELECT id FROM eval_datasets WHERE tenant_id=? AND name=?",
               (DEMO_TENANT, "contract-qa-baseline"))
    if ds:
        out["eval_dataset_id"] = ds["id"]
    else:
        created = evals.create_dataset(DEMO_TENANT, "contract-qa-baseline",
                                       EVAL_CASES)
        out["eval_dataset_id"] = created["id"]
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(seed(), indent=2))
