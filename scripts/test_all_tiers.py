#!/usr/bin/env python3
"""Test all tiers of the LLM smart router at localhost:8080 (v2 — reasoning-safe)."""
import json
import time
import urllib.request
import urllib.error

BASE = "http://localhost:8080"
MAX_TOKENS = 1000  # reasoning models need headroom

key = None
adm = None
for line in open("/root/llm-smart-router/.env"):
    if line.startswith("ROUTER_API_KEY="):
        key = line.strip().split("=", 1)[1].strip('"').strip("'")
    if line.startswith("ADMIN_API_KEY="):
        adm = line.strip().split("=", 1)[1].strip('"').strip("'")

def call(model, messages, session_id=None, extra_headers=None, timeout=180, raw=False):
    body = {"model": model, "messages": messages, "max_tokens": MAX_TOKENS, "stream": False}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if session_id:
        headers["X-Session-Id"] = session_id
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            latency = time.time() - t0
            rheaders = {k.lower(): v for k, v in resp.headers.items()}
            data = json.loads(resp.read().decode())
            if raw:
                return {"ok": True, "latency_s": round(latency, 2), "raw": data, "headers": rheaders}
            msg = data.get("choices", [{}])[0].get("message", {})
            content = msg.get("content")
            if not content:
                content = msg.get("reasoning") or "(no content; reasoning only)"
            return {
                "ok": True,
                "latency_s": round(latency, 2),
                "model_reported": data.get("model"),
                "content": content[:160],
                "finish": data.get("choices", [{}])[0].get("finish_reason"),
                "headers": rheaders,
            }
    except urllib.error.HTTPError as e:
        return {"ok": False, "latency_s": round(time.time() - t0, 2), "status": e.code, "body": e.read().decode()[:400]}
    except Exception as e:
        return {"ok": False, "latency_s": round(time.time() - t0, 2), "error": str(e)}

def hx(r, name):
    if not r.get("ok"):
        return "-"
    v = r["headers"].get(name.lower())
    return "-" if v is None else str(v)[:40]

def post_json(path, payload, key_):
    req = urllib.request.Request(f"{BASE}{path}", data=json.dumps(payload).encode(),
                                 headers={"Authorization": f"Bearer {key_}", "Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return round(time.time() - t0, 2), resp.read().decode()[:400]
    except urllib.error.HTTPError as e:
        return round(time.time() - t0, 2), f"HTTP {e.code}: {e.read().decode()[:300]}"

def get_json(path, key_):
    req = urllib.request.Request(f"{BASE}{path}", headers={"Authorization": f"Bearer {key_}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return round(time.time() - t0, 2), resp.read().decode()[:500]
    except urllib.error.HTTPError as e:
        return round(time.time() - t0, 2), f"HTTP {e.code}: {e.read().decode()[:300]}"

# 1) Forced tier endpoints L1-L5
print("=" * 78)
print("1) FORCED TIER ENDPOINTS  smart-router/L1 .. L5")
print("=" * 78)
simple = [{"role": "user", "content": "Say hello in exactly 5 words."}]
for tier in ["L1", "L2", "L3", "L4", "L5"]:
    sid = f"tier-test-{tier}-{int(time.time())}"
    r = call(f"smart-router/{tier}", simple, session_id=sid)
    if r.get("ok"):
        print(f"  {tier}: model={r['model_reported']:<38} level={hx(r,'X-Router-Level'):<4} "
              f"src={hx(r,'X-Router-Classification-Source'):<10} class_ms={hx(r,'X-Router-Classification-Ms'):<5} "
              f"total_ms={hx(r,'X-Router-Total-Ms'):<7} lat={r['latency_s']}s finish={r['finish']}")
        print(f"       resp: {r['content'][:110]!r}")
    else:
        print(f"  {tier}: FAILED {r}")

# 2) Auto mode classification quality
print()
print("=" * 78)
print("2) AUTO MODE (smart-router) — classifier picks the tier")
print("=" * 78)
cases = [
    ("trivial ", [{"role": "user", "content": "What is 2+2? Answer with just the number."}]),
    ("easy    ", [{"role": "user", "content": "Summarize this in one sentence: The quick brown fox jumps over the lazy dog near the river bank at dawn."}]),
    ("medium  ", [{"role": "user", "content": "Write a Python function that finds all prime numbers up to N using a sieve, with edge cases and a docstring. Include tests."}]),
    ("hard    ", [{"role": "user", "content": "Design a distributed system for a global ride-hailing platform: discuss partitioning, consistency, geo-replication, and failure handling with tradeoffs."}]),
    ("extreme ", [{"role": "user", "content": "Propose a novel research hypothesis in computational biology and design a multi-agent pipeline to investigate it, including experimental validation strategy."}]),
]
for label, msgs in cases:
    sid = f"auto-test-{label.strip()}-{int(time.time())}"
    r = call("smart-router", msgs, session_id=sid)
    if r.get("ok"):
        print(f"  {label}: classified={hx(r,'X-Router-Level'):<4} model={r['model_reported']:<38} "
              f"src={hx(r,'X-Router-Classification-Source'):<10} conf={hx(r,'X-Router-Classifier-Model'):<34} "
              f"class_ms={hx(r,'X-Router-Classification-Ms'):<5} cost={hx(r,'X-Router-Estimated-Cost-Usd'):<10} lat={r['latency_s']}s")
    else:
        print(f"  {label}: FAILED {r}")

# 3) Session pinning
print()
print("=" * 78)
print("3) SESSION PINNING — turn 2 must reuse pin, 0 classifier ms")
print("=" * 78)
sid = f"pin-test-{int(time.time())}"
r1 = call("smart-router", [{"role": "user", "content": "Write a short Python function to compute fibonacci with memoization."}], session_id=sid)
r2 = call("smart-router", [{"role": "user", "content": "Write a short Python function to compute fibonacci with memoization."},
                           {"role": "assistant", "content": "Here is one."},
                           {"role": "user", "content": "Now also handle negative inputs gracefully."}], session_id=sid)
for tag, r in [("turn1", r1), ("turn2", r2)]:
    if r.get("ok"):
        print(f"  {tag}: level={hx(r,'X-Router-Level'):<4} model={r['model_reported']:<38} src={hx(r,'X-Router-Classification-Source'):<10} "
              f"class_ms={hx(r,'X-Router-Classification-Ms'):<5} turn={hx(r,'X-Router-Session-Turn'):<3} "
              f"fallback={hx(r,'X-Router-Fallback-Used'):<6} lat={r['latency_s']}s")
    else:
        print(f"  {tag}: FAILED {r}")

# 4) classify-only + stateless
print()
print("=" * 78)
print("4) classify-only + stateless modes")
print("=" * 78)
t, body = post_json("/v1/chat/completions", {"model": "smart-router/classify-only",
    "messages": [{"role": "user", "content": "Refactor a large legacy codebase to microservices."}]}, key)
print(f"  classify-only ({t}s): {body}")
r = call("smart-router/stateless", [{"role": "user", "content": "What is the capital of France? Answer in 3 words."}], session_id=f"stateless-{int(time.time())}")
if r.get("ok"):
    print(f"  stateless: model={r['model_reported']:<38} level={hx(r,'X-Router-Level'):<4} src={hx(r,'X-Router-Classification-Source'):<10} lat={r['latency_s']}s")
else:
    print(f"  stateless: FAILED {r}")

# 5) Debug classify + pin inspection + admin stats
print()
print("=" * 78)
print("5) Debug endpoints")
print("=" * 78)
t, body = post_json("/v1/router/classify", {"messages": [{"role": "user", "content": "Fix this concurrency bug in our payment service."}]}, key)
print(f"  /v1/router/classify ({t}s): {body}")
t, body = get_json(f"/v1/router/sessions/{sid}", key)
print(f"  /v1/router/sessions/{sid} ({t}s): {body[:380]}")
t, body = get_json("/admin/stats", adm)
print(f"  /admin/stats ({t}s): {body[:380]}")
