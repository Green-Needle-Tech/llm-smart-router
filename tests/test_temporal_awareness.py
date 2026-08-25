#!/usr/bin/env python3
"""E2E test for the temporal awareness feature of llm-smart-router.

Tests that temporal expressions (today, yesterday, tomorrow) in user messages
are normalized to concrete ISO dates before being forwarded to the upstream LLM.

Strategy: ask the LLM to echo the date it sees in the prompt. If temporal
awareness is working, the LLM should see a concrete date (YYYY-MM-DD), not
the word "today".
"""
import json
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta

BASE = "http://localhost:8080"
MAX_TOKENS = 200

# Load keys
key = None
adm = None
with open("/root/llm-smart-router/.env") as f:
    for line in f:
        if line.startswith("ROUTER_API_KEY="):
            key = line.strip().split("=", 1)[1].strip('"').strip("'")
        if line.startswith("ADMIN_API_KEY="):
            adm = line.strip().split("=", 1)[1].strip('"').strip("'")

assert key, "ROUTER_API_KEY not found"
assert adm, "ADMIN_API_KEY not found"


def call(messages, session_id=None, timeout=120):
    body = {"model": "smart-router/L1", "messages": messages, "max_tokens": MAX_TOKENS, "stream": False}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if session_id:
        headers["X-Session-Id"] = session_id
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            msg = data.get("choices", [{}])[0].get("message", {})
            content = msg.get("content") or msg.get("reasoning") or ""
            rheaders = {k.lower(): v for k, v in resp.headers.items()}
            return {"ok": True, "content": content.strip(), "latency_s": round(time.time() - t0, 2), "headers": rheaders}
    except urllib.error.HTTPError as e:
        return {"ok": False, "status": e.code, "body": e.read().decode()[:400]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def hx(r, name):
    if not r.get("ok"):
        return "-"
    v = r["headers"].get(name.lower())
    return "-" if v is None else str(v)


def get_settings():
    req = urllib.request.Request(f"{BASE}/admin/settings",
                                 headers={"Authorization": f"Bearer {adm}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def reload_settings():
    req = urllib.request.Request(f"{BASE}/admin/settings/reload", data=b"", method="POST",
                                 headers={"Authorization": f"Bearer {adm}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def set_temporal(enabled):
    """Toggle temporal_awareness.enabled in settings.json and hot-reload."""
    import subprocess
    # Read, modify, write settings.json
    with open("/root/llm-smart-router/config/settings.json") as f:
        settings = json.load(f)
    settings.setdefault("telemetry", {}).setdefault("temporal_awareness", {})["enabled"] = enabled
    with open("/root/llm-smart-router/config/settings.json", "w") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")
    # Hot-reload
    result = reload_settings()
    assert result.get("status") == "ok", f"Reload failed: {result}"
    # Verify
    active = get_settings()
    actual = active.get("telemetry", {}).get("temporal_awareness", {}).get("enabled")
    assert actual == enabled, f"Config mismatch: expected {enabled}, got {actual}"
    print(f"  temporal_awareness.enabled = {actual}")


# ─── Tests ───

passed = 0
failed = 0
now = datetime.now(timezone.utc)
today_str = now.strftime("%Y-%m-%d")
yesterday_str = (now - timedelta(days=1)).strftime("%Y-%m-%d")
tomorrow_str = (now + timedelta(days=1)).strftime("%Y-%m-%d")

print("=" * 70)
print("TEMPORAL AWARENESS E2E TEST")
print(f"  Reference dates (UTC): today={today_str} yesterday={yesterday_str} tomorrow={tomorrow_str}")
print("=" * 70)

# ── Test 1: Feature is enabled in config ──
print("\n── Test 1: Feature enabled in config ──")
try:
    settings = get_settings()
    ta = settings.get("telemetry", {}).get("temporal_awareness", {})
    assert ta.get("enabled") is True, f"temporal_awareness not enabled: {ta}"
    print(f"  PASS: enabled={ta['enabled']}, tz={ta.get('default_timezone')}, strategy={ta.get('strategy')}")
    passed += 1
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 1

# ── Test 2: "today" is replaced with concrete date ──
print("\n── Test 2: 'today' replaced with concrete date ──")
try:
    sid = f"ta-test-today-{int(time.time())}"
    r = call([{"role": "user", "content": "What date is today? Reply with just the date in YYYY-MM-DD format."}],
             session_id=sid)
    assert r["ok"], f"Request failed: {r}"
    print(f"  Response: {r['content'][:80]!r}")
    print(f"  Level: {hx(r, 'x-router-level')}, Latency: {r['latency_s']}s")
    # The LLM should respond with today's date — if temporal awareness replaced
    # "today" with the actual date, the LLM sees the concrete date.
    assert today_str in r["content"], f"Expected '{today_str}' in response, got: {r['content'][:80]}"
    print(f"  PASS: '{today_str}' found in response")
    passed += 1
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 1

# ── Test 3: "yesterday" is replaced with concrete date ──
print("\n── Test 3: 'yesterday' replaced with concrete date ──")
try:
    sid = f"ta-test-yesterday-{int(time.time())}"
    r = call([{"role": "user", "content": "What date was yesterday? Reply with just the date in YYYY-MM-DD format."}],
             session_id=sid)
    assert r["ok"], f"Request failed: {r}"
    print(f"  Response: {r['content'][:80]!r}")
    assert yesterday_str in r["content"], f"Expected '{yesterday_str}' in response, got: {r['content'][:80]}"
    print(f"  PASS: '{yesterday_str}' found in response")
    passed += 1
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 1

# ── Test 4: "tomorrow" is replaced with concrete date ──
print("\n── Test 4: 'tomorrow' replaced with concrete date ──")
try:
    sid = f"ta-test-tomorrow-{int(time.time())}"
    r = call([{"role": "user", "content": "What date is tomorrow? Reply with just the date in YYYY-MM-DD format."}],
             session_id=sid)
    assert r["ok"], f"Request failed: {r}"
    print(f"  Response: {r['content'][:80]!r}")
    assert tomorrow_str in r["content"], f"Expected '{tomorrow_str}' in response, got: {r['content'][:80]}"
    print(f"  PASS: '{tomorrow_str}' found in response")
    passed += 1
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 1

# ── Test 5: Multiple temporal expressions in one message ──
print("\n── Test 5: Multiple temporal expressions in one message ──")
try:
    sid = f"ta-test-multi-{int(time.time())}"
    r = call([{"role": "user", "content": "I have a meeting today and a deadline tomorrow. What are the exact dates? Reply with two dates in YYYY-MM-DD format, comma-separated."}],
             session_id=sid)
    assert r["ok"], f"Request failed: {r}"
    print(f"  Response: {r['content'][:120]!r}")
    assert today_str in r["content"], f"Expected '{today_str}' in response"
    assert tomorrow_str in r["content"], f"Expected '{tomorrow_str}' in response"
    print(f"  PASS: Both '{today_str}' and '{tomorrow_str}' found")
    passed += 1
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 1

# ── Test 6: Non-temporal text passes through unchanged ──
print("\n── Test 6: Non-temporal text unaffected ──")
try:
    sid = f"ta-test-nontemporal-{int(time.time())}"
    r = call([{"role": "user", "content": "What is the capital of France? Reply in one word."}],
             session_id=sid)
    assert r["ok"], f"Request failed: {r}"
    print(f"  Response: {r['content'][:80]!r}")
    assert "Paris" in r["content"] or "paris" in r["content"].lower(), f"Expected 'Paris', got: {r['content'][:80]}"
    print(f"  PASS: Non-temporal query answered correctly")
    passed += 1
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 1

# ── Test 7: Feature can be toggled off ──
print("\n── Test 7: Feature toggle OFF — temporal expressions pass through raw ──")
try:
    set_temporal(False)
    sid = f"ta-test-disabled-{int(time.time())}"
    # When disabled, asking "what date is today" should NOT reliably return today_str
    # because the LLM doesn't have real-time access. We just verify the request
    # succeeds and the feature is confirmed off.
    r = call([{"role": "user", "content": "What date is today? Reply with just the date in YYYY-MM-DD format."}],
             session_id=sid)
    assert r["ok"], f"Request failed: {r}"
    print(f"  Response (feature off): {r['content'][:80]!r}")
    # Re-enable for subsequent tests
    set_temporal(True)
    print(f"  PASS: Feature toggled off and back on, request succeeded")
    passed += 1
except Exception as e:
    print(f"  FAIL: {e}")
    failed += 1
    # Ensure we re-enable even on failure
    try:
        set_temporal(True)
    except:
        pass

# ─── Summary ──
def main():
    global passed, failed
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 70)
    exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
