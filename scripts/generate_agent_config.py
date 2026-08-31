#!/usr/bin/env python3
"""Hermes Agent and Generic AI Agent Config Generator for LLM Smart Router.

Generates drop-in configs, environment variables, or CLI commands to connect
agents (Hermes, LangChain, LlamaIndex, AutoGen, CrewAI, OpenAI SDK, Cursor/Claude Code)
to LLM Smart Router.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add repo root to sys.path if running as standalone script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.script_utils import resolve_safe_output_path

ROUTER_DEFAULT_URL = "http://localhost:8080/v1"
ROUTER_DEFAULT_MODEL = "smart-router"
ROUTER_DEFAULT_CONTEXT_WINDOW = 1_000_000


def get_router_context_window() -> int:
    """Read context_window from settings.json, fall back to default."""
    for settings_path in [
        Path("/root/llm-smart-router/config/settings.json"),
        Path("config/settings.json"),
        Path("../config/settings.json"),
    ]:
        if settings_path.exists():
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                return int(cfg.get("provider", {}).get("context_window", ROUTER_DEFAULT_CONTEXT_WINDOW))
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                pass
    return ROUTER_DEFAULT_CONTEXT_WINDOW


def get_router_key() -> str:
    # Check current directory .env, then parent
    for env_path in [Path(".env"), Path("/root/llm-smart-router/.env"), Path("../.env")]:
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("ROUTER_API_KEY=") and not line.startswith("#"):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("ROUTER_API_KEY", "your-router-api-key")


def generate_hermes_config(url: str, key: str, model: str, context_window: int) -> str:
    return f"""# Hermes Agent configuration for LLM Smart Router (~/.hermes/config.yaml)
# Merge or replace the 'model' block in your ~/.hermes/config.yaml:

model:
  provider: custom
  default: {model}
  base_url: {url}
  api_mode: chat_completions
  context_length: {context_window}
  api_key: "{key}"

# Optional: define fallback if the router is unreachable
fallback_providers:
  - provider: openrouter
    model: z-ai/glm-5.2
"""


def generate_hermes_cli_commands(url: str, key: str, model: str, context_window: int) -> str:
    return f"""# Hermes CLI configuration commands:
hermes config set model.provider custom
hermes config set model.default {model}
hermes config set model.base_url {url}
hermes config set model.api_mode chat_completions
hermes config set model.api_key "{key}"
hermes config set model.context_length {context_window}
"""


def generate_openai_python(url: str, key: str, model: str, context_window: int = 1_000_000) -> str:
    return f"""# OpenAI Python SDK Integration
from openai import OpenAI

client = OpenAI(
    base_url="{url}",
    api_key="{key}",
)

# Single-turn or Multi-turn with session pinning
response = client.chat.completions.create(
    model="{model}",
    messages=[
        {{"role": "system", "content": "You are a helpful assistant."}},
        {{"role": "user", "content": "Analyze the time complexity of merge sort."}},
    ],
    # Pass session-id to pin classification across multi-turn runs
    extra_headers={{"X-Session-Id": "agent-session-001"}},
)

print(f"Assigned Model: {{response.model}}")
print(response.choices[0].message.content)
"""


def generate_langchain_python(url: str, key: str, model: str, context_window: int = 1_000_000) -> str:
    return f"""# LangChain Integration
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="{url}",
    api_key="{key}",
    model="{model}",
    default_headers={{"X-Session-Id": "langchain-agent-001"}},
)

response = llm.invoke("Summarize the architectural differences between REST and gRPC.")
print(response.content)
"""


def generate_llamaindex_python(url: str, key: str, model: str, context_window: int = 1_000_000) -> str:
    return f"""# LlamaIndex Integration
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    api_base="{url}",
    api_key="{key}",
    model="{model}",
    is_chat_model=True,
    additional_kwargs={{"extra_headers": {{"X-Session-Id": "llamaindex-agent-001"}}}},
)

response = llm.complete("Explain vector database indexing.")
print(response.text)
"""


def generate_env_vars(url: str, key: str, model: str, context_window: int = 1_000_000) -> str:
    return f"""# Standard OpenAI-Compatible Environment Variables
OPENAI_BASE_URL={url}
OPENAI_API_BASE={url}
OPENAI_API_KEY={key}
OPENAI_MODEL_NAME={model}
"""


def generate_claude_cursor_json(url: str, key: str, model: str, context_window: int = 1_000_000) -> str:
    cfg = {
        "openai.baseUrl": url,
        "openai.apiKey": key,
        "openai.model": model,
    }
    return json.dumps(cfg, indent=2)


TEMPLATES = {
    "hermes": generate_hermes_config,
    "hermes-cli": generate_hermes_cli_commands,
    "openai": generate_openai_python,
    "langchain": generate_langchain_python,
    "llamaindex": generate_llamaindex_python,
    "env": generate_env_vars,
    "cursor": generate_claude_cursor_json,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM Smart Router integration templates for AI agents."
    )
    parser.add_argument(
        "--agent",
        "-a",
        choices=list(TEMPLATES.keys()) + ["all"],
        default="hermes",
        help="Target agent or framework (default: hermes)",
    )
    parser.add_argument(
        "--url",
        "-u",
        default=ROUTER_DEFAULT_URL,
        help=f"Router base URL (default: {ROUTER_DEFAULT_URL})",
    )
    parser.add_argument(
        "--key",
        "-k",
        default=None,
        help="Router API key (defaults to ROUTER_API_KEY from .env)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=ROUTER_DEFAULT_MODEL,
        help=f"Virtual router model name (e.g. smart-router, smart-router/L3; default: {ROUTER_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=Path,
        default=None,
        help="Write output directly to file path",
    )

    args = parser.parse_args()
    key = args.key or get_router_key()
    ctx_window = get_router_context_window()

    outputs = []
    if args.agent == "all":
        for name, fn in TEMPLATES.items():
            outputs.append(f"{'='*60}\n# Target: {name.upper()}\n{'='*60}\n" + fn(args.url, key, args.model, ctx_window))
        final_text = "\n\n".join(outputs)
    else:
        final_text = TEMPLATES[args.agent](args.url, key, args.model, ctx_window)

    if args.out:
        out_path = resolve_safe_output_path(args.out)
        out_path.write_text(final_text, encoding="utf-8")
        print(f"Successfully written configuration to {out_path}")
    else:
        print(final_text)


if __name__ == "__main__":
    main()
