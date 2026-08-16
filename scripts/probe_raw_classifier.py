#!/usr/bin/env python3
"""Probe the raw classifier model output for vague prompts."""
import json, urllib.request

KEY = None
OR_KEY = None
for line in open("/root/llm-smart-router/.env"):
    if line.startswith("ROUTER_API_KEY="):
        KEY = line.split("=", 1)[1].strip()
    if line.startswith("OPENROUTER_API_KEY="):
        OR_KEY = line.split("=", 1)[1].strip()

TEMPLATE = open("/tmp/classifier.txt").read()

def raw_classify(prompt):
    digest = f"<<<UNTRUSTED_INPUT_BEGIN>>>\n[conversation: 1 messages, ~{max(1,len(prompt)//4)} task tokens]\n{prompt}\n<<<UNTRUSTED_INPUT_END>>>"
    body = TEMPLATE.replace("{{PROMPT_DIGEST}}", digest)
    payload = {
        "model": "google/gemini-3.1-flash-lite",
        "messages": [{"role": "user", "content": body}],
        "temperature": 0,
        "max_tokens": 60,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"].strip()

for p in ["hi", "help me", "ok thanks", "let's start", "hello there"]:
    print(f"{p!r:<16} -> {raw_classify(p)}")
