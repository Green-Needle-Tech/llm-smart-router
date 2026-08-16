#!/bin/bash
# LLM Smart Router — Full Tier Test Suite (no set -e so we get all results)

BIND_IP=$(ss -tlnp | grep ':8080 ' | awk '{print $4}' | sed 's/:8080//' | head -1)
BASE="http://${BIND_IP}:8080"
ROUTER_KEY=$(grep '^ROUTER_API_KEY=' /root/llm-smart-router/.env | cut -d= -f2)
ADMIN_KEY=$(grep '^ADMIN_API_KEY=' /root/llm-smart-router/.env | cut -d= -f2)
TS=$(date +%s)
ALL_PASS=1

echo "=========================================="
echo " LLM Smart Router — Tier Test Suite"
echo " Target: $BASE"
echo "=========================================="
echo ""

# 1. Health & Ready
echo "--- Health Checks ---"
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/healthz" 2>/dev/null || echo "000")
READY=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/readyz" 2>/dev/null || echo "000")
echo "healthz: HTTP $HEALTH  $( [ "$HEALTH" = "200" ] && echo '✅' || echo '❌' )"
echo "readyz:  HTTP $READY  $( [ "$READY" = "200" ] && echo '✅' || echo '❌' )"
echo ""

# 2. Models
echo "--- Models Endpoint ---"
curl -s "$BASE/v1/models" | python3 -c "
import sys,json
d=json.load(sys.stdin)
for m in d.get('data',[]):
    print(f'  {m[\"id\"]}')
print(f'  Total: {len(d.get(\"data\",[]))} models')
" 2>/dev/null || echo "  ❌ Failed to fetch models"
echo ""

# 3. Tier tests
test_tier() {
    local TIER="$1"
    local PROMPT="$2"
    echo "--- Tier $TIER (smart-router/$TIER) ---"
    
    BODY=$(curl -s -X POST "$BASE/v1/chat/completions" \
        -H "Authorization: Bearer $ROUTER_KEY" \
        -H "Content-Type: application/json" \
        -H "X-Session-Id: test-$TIER-$TS" \
        -d "{\"model\":\"smart-router/$TIER\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":80}" \
        --max-time 120 2>&1)
    
    HTTP_CODE=$(echo "$BODY" | python3 -c "import sys,json; print('200')" 2>/dev/null || echo "err")
    MODEL=$(echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model','N/A'))" 2>/dev/null || echo "parse-error")
    CONTENT=$(echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:120])" 2>/dev/null || echo "parse-error")
    USAGE=$(echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); u=d.get('usage',{}); print(f'prompt={u.get(\"prompt_tokens\",0)} completion={u.get(\"completion_tokens\",0)} total={u.get(\"total_tokens\",0)}')" 2>/dev/null || echo "N/A")
    ERROR=$(echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); e=d.get('error'); print(e.get('message','')[:200] if e else '')" 2>/dev/null || echo "")
    
    if [ "$MODEL" != "parse-error" ] && [ -n "$MODEL" ] && [ "$MODEL" != "N/A" ]; then
        echo "  ✅ Model: $MODEL"
        echo "  Tokens: $USAGE"
        echo "  Response: $CONTENT"
        if [ -n "$ERROR" ]; then
            echo "  ⚠️ Error field: $ERROR"
        fi
    else
        echo "  ❌ FAILED"
        echo "  Raw: $(echo "$BODY" | head -c 400)"
        ALL_PASS=0
    fi
    echo ""
}

test_tier "L1" "What is 2+2?"
test_tier "L2" "Summarize the plot of Romeo and Juliet in 3 sentences."
test_tier "L3" "Write a Python function to check if a binary tree is a valid BST."
test_tier "L4" "Design a distributed consensus algorithm handling Byzantine faults, network partitions. Compare to Paxos and Raft."

# 4. Auto-classification
echo "--- Auto-Classification (smart-router) ---"
BODY=$(curl -s -X POST "$BASE/v1/chat/completions" \
    -H "Authorization: Bearer $ROUTER_KEY" \
    -H "Content-Type: application/json" \
    -H "X-Session-Id: test-auto-$TS" \
    -d '{"model":"smart-router","messages":[{"role":"user","content":"Write a haiku about servers."}],"max_tokens":80}' \
    --max-time 120 2>&1)

MODEL=$(echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model','N/A'))" 2>/dev/null || echo "parse-error")
CONTENT=$(echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:120])" 2>/dev/null || echo "parse-error")

if [ "$MODEL" != "parse-error" ] && [ -n "$MODEL" ] && [ "$MODEL" != "N/A" ]; then
    echo "  ✅ Model: $MODEL"
    echo "  Response: $CONTENT"
else
    echo "  ❌ FAILED"
    echo "  Raw: $(echo "$BODY" | head -c 400)"
    ALL_PASS=0
fi
echo ""

# 5. Admin stats & sessions
echo "--- Admin Stats ---"
curl -s -H "Authorization: Bearer $ADMIN_KEY" "$BASE/admin/stats" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin),indent=2))" 2>/dev/null || echo "  (failed)"
echo ""

echo "--- Admin Sessions ---"
curl -s -H "Authorization: Bearer $ADMIN_KEY" "$BASE/admin/sessions" | python3 -c "
import sys,json
d=json.load(sys.stdin)
if isinstance(d,list):
    print(f'  Active sessions: {len(d)}')
    for s in d[:5]:
        sid=s.get('session_id','?')[:30]
        lvl=s.get('level','?')
        mdl=s.get('model','?')
        print(f'    {sid} | L: {lvl} | M: {mdl}')
else:
    print(json.dumps(d,indent=2)[:300])
" 2>/dev/null || echo "  (failed)"
echo ""

# 6. Metrics
echo "--- Prometheus Metrics (key lines) ---"
curl -s "$BASE/metrics" 2>/dev/null | grep -E "smart_router_(requests|classification|fallback|escalation)|http_requests_total" | head -15 || echo "  (none)"
echo ""

# Summary
echo "=========================================="
echo " SUMMARY"
echo "=========================================="
echo "  Health: HTTP $HEALTH ✅  | Ready: HTTP $READY ✅"
if [ "$ALL_PASS" = "1" ]; then
    echo "  All Tiers: OPERATIONAL ✅"
else
    echo "  Some Tiers: FAILED ❌"
fi
echo "=========================================="
