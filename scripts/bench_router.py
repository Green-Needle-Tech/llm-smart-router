#!/usr/bin/env python3
"""Benchmark router: replay a trace, report cost + accuracy."""
import argparse
import asyncio
import json
from pathlib import Path

import httpx


def resolve_safe_path(path_str: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve and validate that a file path exists and is a regular file."""
    base = (base_dir or Path.cwd()).resolve()
    resolved = (base / path_str if not Path(path_str).is_absolute() else Path(path_str)).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Path is not a regular file: {resolved}")
    return resolved


async def replay_trace(trace_path: str | Path, router_url: str, router_key: str, mode: str):
    """Replay a captured trace through the router."""
    target_path = resolve_safe_path(trace_path)
    trace = [json.loads(line) for line in target_path.read_text().strip().split("\n")]
    total_cost = 0.0
    results = []

    async with httpx.AsyncClient(timeout=120) as client:
        for i, turn in enumerate(trace):
            session_id = turn.get("session_id", f"bench-{i}")
            payload = {
                "model": "smart-router" if mode == "pinned" else turn.get("model", "smart-router"),
                "messages": turn["messages"],
                "stream": False,
            }
            headers = {
                "Authorization": f"Bearer {router_key}",
                "X-Session-Id": session_id,
                "Content-Type": "application/json",
            }

            resp = await client.post(
                f"{router_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            data = resp.json()
            cost = float(resp.headers.get("X-Router-Estimated-Cost-Usd", 0))
            total_cost += cost
            results.append({
                "turn": i,
                "session": session_id,
                "level": resp.headers.get("X-Router-Level"),
                "model": resp.headers.get("X-Router-Model"),
                "source": resp.headers.get("X-Router-Classification-Source"),
                "cost": cost,
            })

    return total_cost, results


def main():
    parser = argparse.ArgumentParser(description="Benchmark the LLM Smart Router")
    parser.add_argument("--trace", required=True, help="Path to trace file (JSONL)")
    parser.add_argument("--router-url", default="http://localhost:8080")
    parser.add_argument("--router-key", default="test-key")
    parser.add_argument("--mode", choices=["pinned", "baseline"], default="pinned")
    args = parser.parse_args()

    cost, results = asyncio.run(replay_trace(
        resolve_safe_path(args.trace), args.router_url, args.router_key, args.mode
    ))

    print(f"\n{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Total turns: {len(results)}")
    print(f"Total cost: ${cost:.4f}")
    print(f"Classifier calls: {sum(1 for r in results if r['source'] != 'session')}")
    print(f"Session hits: {sum(1 for r in results if r['source'] == 'session')}")
    print(f"Level distribution:")
    for level in ["L1", "L2", "L3", "L4"]:
        count = sum(1 for r in results if r["level"] == level)
        print(f"  {level}: {count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
