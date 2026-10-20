#!/usr/bin/env bash
# Live smoke test against a real uvicorn server (mock LLM provider, offline).
# Usage: bash smoke_test.sh   — transcript is tee'd to smoke_output.txt
set -u
cd "$(dirname "$0")"
export EAAP_DB_PATH="$PWD/smoke.db"
PORT=8077
BASE="http://127.0.0.1:$PORT"
AUTH="Authorization: Bearer eaap-demo-key"
rm -f smoke.db*

.venv/bin/python -m uvicorn app.main:app --port $PORT >/tmp/eaap-uvicorn.log 2>&1 &
UVPID=$!
trap 'kill $UVPID 2>/dev/null' EXIT
sleep 2

exec > >(tee smoke_output.txt) 2>&1

echo "== health =="
curl -s $BASE/health; echo
echo "== seed =="
EAAP_DB_PATH=$EAAP_DB_PATH .venv/bin/python -m app.seed

QA_ID=$(curl -s $BASE/v1/agents -H "$AUTH" | .venv/bin/python -c "import json,sys; print([a['id'] for a in json.load(sys.stdin) if a['name']=='contract-qa'][0])")
echo "contract-qa agent: $QA_ID"

poll() {  # $1 = run id -> echoes final run json
  for i in $(seq 1 40); do
    R=$(curl -s $BASE/v1/runs/$1 -H "$AUTH")
    S=$(echo "$R" | .venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['status'])")
    [ "$S" != "running" ] && [ "$S" != "queued" ] && { echo "$R"; return; }
    sleep 0.25
  done
  echo "$R"
}

echo "== run 1: KB-covered question (expect: completes, grounded, cited) =="
RUN1=$(curl -s -X POST $BASE/v1/agents/$QA_ID/runs -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"input": {"question": "How many days do we have to pay an invoice?"}}' \
  | .venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['id'])")
echo "run id: $RUN1"
poll $RUN1 | .venv/bin/python -c "import json,sys; r=json.load(sys.stdin); print('status:', r['status']); print('output:', str(r['output'])[:160]); print('tokens:', r['total_tokens'], 'cost:', r['total_cost'])"

echo "== run 2: off-topic question (expect: HITL pause -> approve -> completed) =="
RUN2=$(curl -s -X POST $BASE/v1/agents/$QA_ID/runs -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"input": {"question": "What is the weather in Tokyo today?"}}' \
  | .venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['id'])")
echo "run id: $RUN2"
poll $RUN2 | .venv/bin/python -c "import json,sys; r=json.load(sys.stdin); print('status:', r['status'])"
APPR=$(curl -s "$BASE/v1/approvals?status=pending" -H "$AUTH" | .venv/bin/python -c "import json,sys; print([a['id'] for a in json.load(sys.stdin) if a['run_id']=='$RUN2'][0])")
echo "approval id: $APPR"
curl -s -X POST $BASE/v1/approvals/$APPR/decision -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"decision": "approved", "comment": "reviewed - send with disclaimer"}' \
  | .venv/bin/python -c "import json,sys; a=json.load(sys.stdin); print('decision recorded:', a['status'])"
poll $RUN2 | .venv/bin/python -c "import json,sys; r=json.load(sys.stdin); print('status after approval:', r['status']); print('output:', str(r['output'])[:110])"

echo "== trace of run 2 =="
curl -s $BASE/v1/runs/$RUN2/trace -H "$AUTH" | .venv/bin/python -c "
import json,sys
t=json.load(sys.stdin)
for e in t['node_executions']: print(' node:', e['node_id'], e['node_type'], e['status'], str(e['latency_ms'])+'ms')
print(' events:', [ev['type'] for ev in t['events']])
print(' checkpoints:', t['checkpoints'])"

echo "== SSE replay of run 2 events =="
curl -s -N --max-time 8 $BASE/v1/runs/$RUN2/events -H "$AUTH" | head -30

echo "== eval run on seeded dataset =="
DS_ID=$(EAAP_DB_PATH=$EAAP_DB_PATH .venv/bin/python -c "
from app import db; db.init_db()
print(db.q1(\"SELECT id FROM eval_datasets WHERE name='contract-qa-baseline'\")['id'])")
VER_ID=$(curl -s $BASE/v1/agents/$QA_ID -H "$AUTH" | .venv/bin/python -c "import json,sys; print(json.load(sys.stdin)['current_version_id'])")
curl -s -X POST $BASE/v1/evals/runs -H "$AUTH" -H 'Content-Type: application/json' \
  -d "{\"dataset_id\": \"$DS_ID\", \"agent_id\": \"$QA_ID\", \"version_id\": \"$VER_ID\"}" \
  | .venv/bin/python -c "import json,sys; e=json.load(sys.stdin); print('eval status:', e['status'], '| metrics:', e['metrics'])"

echo "== usage =="
curl -s $BASE/v1/usage -H "$AUTH"; echo
echo "== trace viewer (HTML) =="
curl -s $BASE/ui/runs/$RUN2 -H "$AUTH" | head -c 200; echo
echo "== smoke done =="
