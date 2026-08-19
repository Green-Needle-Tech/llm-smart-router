#!/usr/bin/env python3
"""Change a tier's model (and optionally fallbacks) in settings.json and hot-reload the router.

Usage:
    python3 scripts/set_tier_model.py L4 z-ai/glm-5.3
    python3 scripts/set_tier_model.py L4 z-ai/glm-5.3 --fallbacks z-ai/glm-5.2,anthropic/claude-opus-5
    python3 scripts/set_tier_model.py L4 z-ai/glm-5.3 --no-verify   # skip live probe

Zero downtime: edits the bind-mounted settings.json, then POSTs /admin/settings/reload.
Existing session pins keep their level (on_config_change: keep_level) and resolve
to the new model on the next request.
"""
import argparse
import json
import sys
import urllib.request

ROOT = "/root/llm-smart-router"
SETTINGS = f"{ROOT}/config/settings.json"
BASE = "http://localhost:8080"


def load_keys():
    key = adm = None
    for line in open(f"{ROOT}/.env"):
        if line.startswith("ROUTER_API_KEY="):
            key = line.strip().split("=", 1)[1].strip('"').strip("'")
        if line.startswith("ADMIN_API_KEY="):
            adm = line.strip().split("=", 1)[1].strip('"').strip("'")
    return key, adm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("tier", help="Tier level, e.g. L4")
    p.add_argument("model", help="OpenRouter model id, e.g. z-ai/glm-5.3")
    p.add_argument("--fallbacks", help="Comma-separated fallback list (replaces existing)")
    p.add_argument("--no-verify", action="store_true", help="Skip live probe after reload")
    args = p.parse_args()

    tier = args.tier.upper()
    if not tier.startswith("L") or tier[1:] not in "12345" or len(tier) != 2:
        sys.exit(f"Invalid tier: {args.tier} (expected L1–L5)")

    # 1. Edit settings.json
    with open(SETTINGS) as f:
        settings = json.load(f)
    if tier not in settings.get("routing", {}):
        sys.exit(f"Tier {tier} not found in settings.json")

    old = settings["routing"][tier]["model"]
    settings["routing"][tier]["model"] = args.model
    if args.fallbacks:
        settings["routing"][tier]["fallbacks"] = [m.strip() for m in args.fallbacks.split(",") if m.strip()]
    with open(SETTINGS, "w") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")
    print(f"settings.json: {tier} model {old} -> {args.model}")

    # 2. Hot-reload (no restart)
    _, adm = load_keys()
    if not adm:
        sys.exit("ADMIN_API_KEY not found in .env")
    req = urllib.request.Request(f"{BASE}/admin/settings/reload", data=b"", method="POST",
                                 headers={"Authorization": f"Bearer {adm}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        print(f"reload: HTTP {resp.status} {resp.read().decode()}")

    # 3. Confirm active config
    req = urllib.request.Request(f"{BASE}/admin/settings",
                                 headers={"Authorization": f"Bearer {adm}"})
    active = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
    t = active["routing"][tier]
    print(f"active: {tier} = {t['model']} | fallbacks: {t['fallbacks']}")
    if t["model"] != args.model:
        sys.exit("MISMATCH: active config does not match requested model")

    # 4. Live probe (optional)
    if not args.no_verify:
        key, _ = load_keys()
        body = json.dumps({"model": f"smart-router/{tier}",
                           "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
                           "max_tokens": 1000, "stream": False}).encode()
        req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {key}",
                                              "Content-Type": "application/json",
                                              "X-Session-Id": f"tier-change-{tier.lower()}"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        content = (data["choices"][0]["message"].get("content") or "")[:80]
        print(f"probe: HTTP 200 | {content!r}")
    print("done")


if __name__ == "__main__":
    main()
