#!/usr/bin/env python3
"""Interactive Onboarding & Health Check Setup CLI for LLM Smart Router.

Tests the connection to LLM Smart Router, verifies API keys, inspects available
models/tiers, and prints or saves ready-to-use configuration for Hermes Agent,
LangChain, LlamaIndex, Cursor, or OpenAI SDK.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = "http://localhost:8080/v1"


def get_default_key() -> str:
    for env_path in [Path(".env"), Path("/root/llm-smart-router/.env"), Path("../.env")]:
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("ROUTER_API_KEY=") and not line.startswith("#"):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("ROUTER_API_KEY", "")


def check_router_health(base_url: str, api_key: str) -> tuple[bool, str, list[str]]:
    """Verify router connectivity and list available models."""
    root_url = base_url.rstrip("/")
    if root_url.endswith("/v1"):
        health_url = root_url[:-3] + "/healthz"
        models_url = root_url + "/models"
    else:
        health_url = root_url + "/healthz"
        models_url = root_url + "/v1/models"

    # 1. Health check
    try:
        req = urllib.request.Request(health_url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                return False, f"Healthcheck failed with HTTP status {resp.status}", []
    except Exception as e:
        return False, f"Could not connect to {health_url}: {e}", []

    # 2. Models check
    models = []
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(models_url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("id") for m in data.get("data", []) if "id" in m]
    except Exception as e:
        return True, f"Router is UP at {health_url}, but /models check failed: {e}", []

    return True, f"Router is healthy at {health_url}", models


def main():
    parser = argparse.ArgumentParser(
        description="LLM Smart Router Agent Onboarding & Quickstart Setup"
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Router URL (default: {DEFAULT_URL})")
    parser.add_argument("--key", default=None, help="Router API key (ROUTER_API_KEY)")
    parser.add_argument(
        "--apply-hermes",
        action="store_true",
        help="Directly apply configuration to ~/.hermes/config.yaml",
    )
    args = parser.parse_args()

    api_key = args.key or get_default_key()
    url = args.url.rstrip("/")

    print("=" * 65)
    print(" 🚀 LLM Smart Router — Agent Onboarding & Connection Setup")
    print("=" * 65)

    print(f"\n[*] Checking connectivity to {url}...")
    healthy, msg, models = check_router_health(url, api_key)

    if not healthy:
        print(f"[-] ERROR: {msg}")
        print("\nTroubleshooting tips:")
        print(" 1. Ensure the router container is running: docker compose up -d")
        print(" 2. Verify port mapping (default is 8080:8000)")
        print(" 3. Check logs: docker logs smart-router")
        sys.exit(1)

    print(f"[+] SUCCESS: {msg}")
    if models:
        print(f"[+] Available models ({len(models)}): {', '.join(models[:8])}{'...' if len(models) > 8 else ''}")

    print("\n" + "-" * 65)
    print(" 📋 HERMES AGENT CONFIGURATION (~/.hermes/config.yaml)")
    print("-" * 65)
    hermes_yaml = f"""model:
  provider: custom
  default: smart-router
  base_url: {url}
  api_mode: chat_completions
  context_length: 1000000
  api_key: "{api_key if api_key else 'YOUR_ROUTER_API_KEY'}"
"""
    print(hermes_yaml)

    if args.apply_hermes:
        config_path = Path.home() / ".hermes" / "config.yaml"
        if not config_path.exists():
            print(f"[-] {config_path} does not exist. Cannot apply directly.")
        else:
            print(f"[*] To update {config_path}, use `hermes config set` commands or patch the file directly.")

    print("-" * 65)
    print(" 💻 CLI ONE-LINERS FOR HERMES AGENT")
    print("-" * 65)
    print(f'hermes config set model.provider custom')
    print(f'hermes config set model.default smart-router')
    print(f'hermes config set model.base_url "{url}"')
    print(f'hermes config set model.api_mode chat_completions')
    print(f'hermes config set model.api_key "{api_key if api_key else "YOUR_ROUTER_API_KEY"}"')
    print(f'hermes config set model.context_length 1000000')

    print("\n" + "-" * 65)
    print(" 🐍 PYTHON OPENAI / GENERIC AGENT CLIENT")
    print("-" * 65)
    print(f"""from openai import OpenAI

client = OpenAI(
    base_url="{url}",
    api_key="{api_key if api_key else 'YOUR_ROUTER_API_KEY'}",
)

# Multi-turn agents: pass X-Session-Id in extra_headers
response = client.chat.completions.create(
    model="smart-router",
    messages=[{{"role": "user", "content": "Hello!"}}],
    extra_headers={{"X-Session-Id": "agent-run-001"}}
)
print(response.choices[0].message.content)
""")

    print("=" * 65)
    print("💡 Tip: Run `python3 scripts/generate_agent_config.py --agent all` to see templates for LangChain, LlamaIndex, Cursor, and more!")
    print("=" * 65)


if __name__ == "__main__":
    main()
