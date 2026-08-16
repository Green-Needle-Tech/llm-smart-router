#!/usr/bin/env python3
"""Classifier accuracy test: send prompts of known difficulty, compare to expected level."""
import json
import os
import sys
import time
import urllib.request

BASE = "http://localhost:8080"
KEY = None
for line in open("/root/llm-smart-router/.env"):
    if line.startswith("ROUTER_API_KEY="):
        KEY = line.split("=", 1)[1].strip()

# (prompt, expected_level, note)
CASES = [
    # --- L1 TRIVIAL ---
    ("Convert this list to uppercase: apple, banana, cherry", "L1", "mechanical transform"),
    ("Extract all email addresses from: contact bob@x.com or sue@y.org for info", "L1", "extraction"),
    ("Is this sentence positive or negative? 'I love this product.'", "L1", "classification"),
    ("Format this as JSON: name John, age 30, city Paris", "L1", "formatting"),
    ("Translate 'good morning' to French", "L1", "lookup transform"),

    # --- L2 EASY ---
    ("What is the capital of Australia?", "L2", "general knowledge"),
    ("Write a two-sentence product description for a stainless steel water bottle", "L2", "short generation"),
    ("Explain what DNS is in simple terms", "L2", "single-step explain"),
    ("Give me 5 name ideas for a coffee shop", "L2", "short creative"),
    ("Summarize the plot of Romeo and Juliet in 3 sentences", "L2", "short summary"),

    # --- L3 MEDIUM ---
    ("Write a Python function that merges two sorted linked lists and returns the head, handling empty inputs", "L3", "real code"),
    ("My Docker container exits immediately with code 137. Walk me through diagnosing it.", "L3", "debugging"),
    ("Compare PostgreSQL vs MongoDB for a multi-tenant SaaS analytics workload and recommend one", "L3", "tradeoff analysis"),
    ("Here's a stack trace from a Node app: TypeError: Cannot read property 'map' of undefined at line 42. Fix it.", "L3", "debugging"),
    ("Write a SQL query that finds the top 3 customers by revenue per region, with ties broken by signup date", "L3", "correctness-critical SQL"),

    # --- L4 HARD ---
    ("Design a globally distributed, multi-region event sourcing system with exactly-once delivery guarantees, handling network partitions and clock skew. Cover storage, consensus, and replay.", "L4", "system design"),
    ("Prove that any comparison-based sorting algorithm requires Omega(n log n) comparisons in the worst case, then explain how radix sort circumvents this bound.", "L4", "subtle math proof"),
    ("Refactor our 200k-line monolith into microservices. Give me a phased 18-month migration plan with rollback strategy at each phase, team topology, and how to handle the shared database.", "L4", "long-horizon planning"),
    ("We have a race condition in our distributed lock implementation that only manifests under high load across 3 datacenters. Redis-based, 5s TTL, clients renew at 2s. Analyze what could go wrong and design a fix.", "L4", "deep novel reasoning"),
    ("Should we accept a $40M acquisition offer or raise a Series C? We have 18mo runway, 140% NDR, but our largest customer (31% of ARR) is up for renewal in 4 months. Analyze.", "L4", "high-stakes ambiguous judgment"),

    # --- UNKNOWN / vague ---
    ("hi", "UNKNOWN", "greeting"),
    ("help me", "UNKNOWN", "too vague"),
    ("ok thanks", "UNKNOWN", "acknowledgement"),
]


def classify(prompt, debug=False):
    url = f"{BASE}/v1/router/classify"
    if debug:
        url += "?debug=digest"
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "router": {"bypass_cache": True},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {KEY}",
        },
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    data["_wall_ms"] = int((time.time() - t0) * 1000)
    return data


def lvl_num(l):
    return {"L1": 1, "L2": 2, "L3": 3, "L4": 4}.get(l, 0)


def main():
    results = []
    print(f"{'#':<3} {'EXP':<8} {'GOT':<8} {'CONF':<6} {'MS':<6} {'SRC':<10} NOTE")
    print("-" * 100)
    for i, (prompt, expected, note) in enumerate(CASES, 1):
        try:
            r = classify(prompt)
        except Exception as e:
            print(f"{i:<3} {expected:<8} ERROR    -      -      -          {e}")
            results.append({"expected": expected, "got": "ERROR", "note": note})
            continue
        got = r["level"]
        conf = r.get("confidence", 0)
        ms = r.get("latency_ms", 0)
        src = r.get("source", "")
        mark = "OK " if got == expected else ("HI " if lvl_num(got) > lvl_num(expected) else "LO ")
        if expected == "UNKNOWN" and got != "UNKNOWN":
            mark = "?? "
        print(f"{i:<3} {expected:<8} {got:<8} {conf:<6.2f} {ms:<6} {src:<10} {mark}{note}")
        results.append({
            "n": i, "prompt": prompt[:60], "expected": expected, "got": got,
            "conf": conf, "ms": ms, "src": src, "note": note, "mark": mark.strip(),
        })

    with open("/root/llm-smart-router/scripts/classifier_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    graded = [r for r in results if r.get("expected") != "UNKNOWN" and r.get("got") != "ERROR"]
    exact = sum(1 for r in graded if r["got"] == r["expected"])
    over = sum(1 for r in graded if lvl_num(r["got"]) > lvl_num(r["expected"]))
    under = sum(1 for r in graded if 0 < lvl_num(r["got"]) < lvl_num(r["expected"]))
    lats = [r["ms"] for r in results if r.get("ms")]
    print("\n" + "=" * 100)
    print(f"Graded (L1-L4): {len(graded)}  |  Exact: {exact} ({exact/max(1,len(graded))*100:.0f}%)  |  Over-escalated: {over}  |  Under-called: {under}")
    unk = [r for r in results if r.get("expected") == "UNKNOWN"]
    unk_ok = sum(1 for r in unk if r["got"] == "UNKNOWN")
    print(f"UNKNOWN cases: {unk_ok}/{len(unk)} correctly returned UNKNOWN")
    if lats:
        lats.sort()
        print(f"Latency: min {lats[0]}ms  median {lats[len(lats)//2]}ms  max {lats[-1]}ms  avg {sum(lats)//len(lats)}ms")


if __name__ == "__main__":
    main()
