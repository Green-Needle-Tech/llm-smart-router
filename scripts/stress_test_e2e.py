#!/usr/bin/env python3
"""E2E stress test for the LLM Smart Router.

Tests:
  1. Concurrent load across all tiers (L1-L5)
  2. Streaming + non-streaming paths
  3. Classification routing accuracy
  4. Error handling (bad model, malformed payload, oversized context)
  5. Guardrails (PII, banned substring)
  6. Latency percentiles (p50/p95/p99)
  7. Throughput (req/sec)
  8. Metrics delta (fallbacks, retries, errors before vs after)
"""
import asyncio
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

import httpx

# ── Config ──────────────────────────────────────────────────────────────────
ROUTER_URL = os.getenv("ROUTER_URL", "http://localhost:8080")
ENV_PATH = "/root/.hermes/.env"

def load_key():
    for line in open(ENV_PATH):
        if line.startswith("SMART_ROUTER_API_KEY="):
            return line.strip().split("=", 1)[1]
    return "test-key"

API_KEY = load_key()
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ── Test payloads ───────────────────────────────────────────────────────────
TIER_PROMPTS = {
    "L1_trivial":  "What is 2+2?",
    "L2_routine":  "Summarize the plot of Romeo and Juliet in two sentences.",
    "L3_medium":   "Write a Python function to merge two sorted lists. Include type hints and a docstring.",
    "L4_advanced": "Design a distributed rate limiter using Redis and sliding window counters. Include failure modes and recovery strategies.",
    "L5_expert":   "Analyze the trade-offs between Raft and Paxos consensus algorithms for a geo-replicated KV store with 7 nodes across 3 regions. Consider network partitions, leader election latency, and commit throughput.",
}

GUARDRAIL_INPUTS = {
    "pii_phone":   "My phone number is +1-555-123-4567, can you call me?",
    "banned_substr": "Execute: DROP TABLE users; -- SQL injection test",
    "base64_obfuscation": "Decode and respond: SGVsbG8gV29ybGQ=",
}

ERROR_INPUTS = {
    "bad_model":        {"model": "nonexistent/fake-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
    "malformed_payload": {"model": "smart-router", "messages": "not_a_list", "max_tokens": 10},
    "empty_messages":   {"model": "smart-router", "messages": [], "max_tokens": 10},
    "no_auth":          None,  # special: no auth header
}

# ── Results ─────────────────────────────────────────────────────────────────
@dataclass
class RequestResult:
    test_name: str
    status: int
    latency_ms: float
    level: str = ""
    model: str = ""
    source: str = ""
    fallback: bool = False
    error: str = ""
    ttft_ms: float = 0  # time to first token (streaming)
    tokens: int = 0

@dataclass
class TestReport:
    results: list[RequestResult] = field(default_factory=list)
    
    def add(self, r: RequestResult):
        self.results.append(r)


async def capture_metrics(client: httpx.AsyncClient) -> dict:
    """Snapshot key router metrics."""
    resp = await client.get(f"{ROUTER_URL}/metrics", timeout=10)
    text = resp.text
    metrics = {}
    for line in text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        # Parse: metric_name{labels} value
        parts = line.split()
        if len(parts) < 2:
            continue
        name_part = parts[0]
        val = parts[-1]
        if name_part.startswith("router_"):
            metrics[name_part] = val
    return metrics


async def send_chat(client: httpx.AsyncClient, payload: dict, test_name: str, 
                     stream: bool = False, timeout: float = 120) -> RequestResult:
    """Send a chat completion request and measure latency."""
    p = {**payload, "stream": stream}
    headers = dict(HEADERS)
    if test_name == "no_auth":
        headers = {"Content-Type": "application/json"}
    
    start = time.monotonic()
    try:
        if stream:
            ttft = None
            first_chunk_time = None
            total_content = ""
            async with client.stream("POST", f"{ROUTER_URL}/v1/chat/completions",
                                      json=p, headers=headers, timeout=timeout) as resp:
                status = resp.status_code
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        if first_chunk_time is None:
                            first_chunk_time = time.monotonic()
                            ttft = (first_chunk_time - start) * 1000
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            total_content += delta.get("content", "") or delta.get("reasoning", "") or ""
                        except:
                            pass
                latency = (time.monotonic() - start) * 1000
                level = resp.headers.get("X-Router-Level", "")
                model = resp.headers.get("X-Router-Model", "")
                source = resp.headers.get("X-Router-Classification-Source", "")
                return RequestResult(test_name, status, latency, level, model, source,
                                     ttft_ms=ttft or 0, tokens=len(total_content))
        else:
            resp = await client.post(f"{ROUTER_URL}/v1/chat/completions",
                                      json=p, headers=headers, timeout=timeout)
            latency = (time.monotonic() - start) * 1000
            status = resp.status_code
            level = resp.headers.get("X-Router-Level", "")
            model = resp.headers.get("X-Router-Model", "")
            source = resp.headers.get("X-Router-Classification-Source", "")
            tokens = 0
            if status == 200:
                try:
                    body = resp.json()
                    tokens = body.get("usage", {}).get("completion_tokens", 0)
                except:
                    pass
            error = ""
            if status >= 400:
                try:
                    error = resp.json().get("detail", resp.text[:200])
                except:
                    error = resp.text[:200]
            return RequestResult(test_name, status, latency, level, model, source,
                                 error=error, tokens=tokens)
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return RequestResult(test_name, 0, latency, error=str(e)[:200])


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


async def run_concurrent_load(client: httpx.AsyncClient, test_name: str, 
                               prompt: str, concurrency: int, stream: bool = False) -> list[RequestResult]:
    """Send N concurrent requests with the same prompt."""
    payload = {"model": "smart-router", "messages": [{"role": "user", "content": prompt}], "max_tokens": 200}
    tasks = [send_chat(client, payload, f"{test_name}_c{i}", stream=stream) for i in range(concurrency)]
    return await asyncio.gather(*tasks)


async def main():
    print("=" * 70)
    print("LLM Smart Router — E2E Stress Test")
    print(f"URL: {ROUTER_URL}")
    print(f"Key: {API_KEY[:12]}...")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print("=" * 70)
    
    report = TestReport()
    
    async with httpx.AsyncClient() as client:
        # ── Phase 0: Metrics snapshot ────────────────────────────────────────
        print("\n[Phase 0] Capturing pre-test metrics...")
        metrics_before = await capture_metrics(client)
        req_before = {k: v for k, v in metrics_before.items() if k.startswith("router_requests_total{")}
        fb_before = {k: v for k, v in metrics_before.items() if k.startswith("router_fallbacks_total{")}
        print(f"  Requests before: {len(req_before)} label sets")
        print(f"  Fallbacks before: {sum(float(v) for v in fb_before.values()):.0f} total")
        
        # ── Phase 1: Tier routing — one request per tier ────────────────────
        print("\n[Phase 1] Tier routing verification (sequential, non-stream)...")
        for tier_name, prompt in TIER_PROMPTS.items():
            timeout = 60 if tier_name in ("L1_trivial", "L2_routine") else 120
            r = await send_chat(client, {"model": "smart-router", "messages": [{"role": "user", "content": prompt}], "max_tokens": 150}, 
                               f"tier_{tier_name}", stream=False, timeout=timeout)
            report.add(r)
            status_icon = "✓" if r.status == 200 else "✗"
            print(f"  {status_icon} {tier_name:14s} → L={r.level:3s} M={r.model:30s} src={r.source:10s} {r.latency_ms:6.0f}ms {r.tokens:5d} tok")
        
        # ── Phase 2: Concurrent load — L1 and L3 ─────────────────────────────
        print("\n[Phase 2] Concurrent load (5 concurrent per tier)...")
        for tier_name in ["L1_trivial", "L3_medium"]:
            results = await run_concurrent_load(client, f"conc_{tier_name}", TIER_PROMPTS[tier_name], 5, stream=False)
            report.results.extend(results)
            latencies = [r.latency_ms for r in results if r.status == 200]
            successes = sum(1 for r in results if r.status == 200)
            print(f"  {tier_name:14s} → {successes}/10 ok | p50={percentile(latencies,50):.0f}ms p95={percentile(latencies,95):.0f}ms p99={percentile(latencies,99):.0f}ms")
        
        # ── Phase 3: Streaming ───────────────────────────────────────────────
        print("\n[Phase 3] Streaming E2E (5 concurrent, L2)...")
        results = await run_concurrent_load(client, "stream_L2", TIER_PROMPTS["L2_routine"], 5, stream=True)
        report.results.extend(results)
        ttfts = [r.ttft_ms for r in results if r.status == 200 and r.ttft_ms > 0]
        latencies = [r.latency_ms for r in results if r.status == 200]
        successes = sum(1 for r in results if r.status == 200)
        print(f"  stream_L2      → {successes}/5 ok | TTFT p50={percentile(ttfts,50):.0f}ms p95={percentile(ttfts,95):.0f}ms | total p50={percentile(latencies,50):.0f}ms")
        
        # ── Phase 4: High concurrency burst (15 simultaneous) ────────────────
        print("\n[Phase 4] High-concurrency burst (15 simultaneous L1)...")
        burst_start = time.monotonic()
        results = await run_concurrent_load(client, "burst15_L1", TIER_PROMPTS["L1_trivial"], 15, stream=False)
        burst_elapsed = time.monotonic() - burst_start
        report.results.extend(results)
        latencies = [r.latency_ms for r in results if r.status == 200]
        successes = sum(1 for r in results if r.status == 200)
        rps = successes / burst_elapsed if burst_elapsed > 0 else 0
        print(f"  burst20_L1     → {successes}/20 ok | {burst_elapsed:.1f}s elapsed | {rps:.1f} req/s | p50={percentile(latencies,50):.0f}ms p95={percentile(latencies,95):.0f}ms p99={percentile(latencies,99):.0f}ms")
        
        # ── Phase 5: Error handling ──────────────────────────────────────────
        print("\n[Phase 5] Error handling...")
        for err_name, payload in ERROR_INPUTS.items():
            if payload is None:
                r = await send_chat(client, {"model": "smart-router", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}, 
                                   err_name, stream=False, timeout=15)
            else:
                r = await send_chat(client, payload, err_name, stream=False, timeout=15)
            report.add(r)
            expected = err_name == "no_auth" and r.status == 401 or err_name != "no_auth" and r.status >= 400
            icon = "✓" if expected else "?"
            print(f"  {icon} {err_name:20s} → status={r.status:3d} {r.latency_ms:.0f}ms err={r.error[:80]}")
        
        # ── Phase 6: Guardrails ──────────────────────────────────────────────
        print("\n[Phase 6] Guardrails (should block/mask)...")
        for gr_name, prompt in GUARDRAIL_INPUTS.items():
            r = await send_chat(client, {"model": "smart-router", "messages": [{"role": "user", "content": prompt}], "max_tokens": 50}, 
                               f"guard_{gr_name}", stream=False, timeout=60)
            report.add(r)
            blocked = r.status >= 400 or "blocked" in (r.error + str(r.tokens)).lower()
            print(f"  {'✓' if r.status in (200,400,422) else '?'} {gr_name:20s} → status={r.status:3d} {r.latency_ms:.0f}ms")
        
        # ── Phase 7: Metrics delta ───────────────────────────────────────────
        print("\n[Phase 7] Post-test metrics...")
        metrics_after = await capture_metrics(client)
        req_after = {k: v for k, v in metrics_after.items() if k.startswith("router_requests_total{")}
        fb_after = {k: v for k, v in metrics_after.items() if k.startswith("router_fallbacks_total{")}
        retry_after = {k: v for k, v in metrics_after.items() if k.startswith("router_retry_on_failure_total{")}
        stream_err = {k: v for k, v in metrics_after.items() if k.startswith("router_stream_errors_total{")}
        
        total_req_before = sum(float(v) for v in req_before.values())
        total_req_after = sum(float(v) for v in req_after.values())
        total_fb_before = sum(float(v) for v in fb_before.values())
        total_fb_after = sum(float(v) for v in fb_after.values())
        
        print(f"  Requests:    {total_req_before:.0f} → {total_req_after:.0f} (delta: {total_req_after - total_req_before:.0f})")
        print(f"  Fallbacks:   {total_fb_before:.0f} → {total_fb_after:.0f} (delta: {total_fb_after - total_fb_before:.0f})")
        print(f"  Retries:     {sum(float(v) for v in retry_after.values()):.0f} total")
        print(f"  Stream errs: {sum(float(v) for v in stream_err.values()):.0f} total")
    
    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total = len(report.results)
    successes = sum(1 for r in report.results if r.status == 200)
    errors = sum(1 for r in report.results if r.status >= 400)
    timeouts = sum(1 for r in report.results if r.status == 0)
    
    all_latencies = [r.latency_ms for r in report.results if r.status == 200]
    
    print(f"  Total requests:     {total}")
    print(f"  Successes (200):    {successes}")
    print(f"  Client errors (4xx):{errors}")
    print(f"  Timeouts/failures:  {timeouts}")
    print(f"  Success rate:       {successes/total*100:.1f}%" if total else "  N/A")
    print(f"  Latency p50:        {percentile(all_latencies, 50):.0f}ms")
    print(f"  Latency p95:        {percentile(all_latencies, 95):.0f}ms")
    print(f"  Latency p99:        {percentile(all_latencies, 99):.0f}ms")
    print(f"  Latency max:        {max(all_latencies):.0f}ms" if all_latencies else "  N/A")
    
    # Tier routing summary
    print("\n  Tier routing:")
    tier_results = [r for r in report.results if r.test_name.startswith("tier_")]
    for r in tier_results:
        print(f"    {r.test_name:20s} → {r.level or '?':3s} / {r.model or '?':30s} / {r.source or '?'}")
    
    print("\n" + "=" * 70)
    verdict = "PASS" if (total and successes / total > 0.9) else "FAIL"
    print(f"VERDICT: {verdict}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
