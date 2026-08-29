#!/usr/bin/env python3
"""Evaluate classifier accuracy against a labeled set."""
import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

# Add repo root to sys.path if running as standalone script
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import httpx  # noqa: E402

from scripts.script_utils import resolve_safe_path, validate_safe_http_url  # noqa: E402


async def evaluate(labeled_path: str | Path, router_url: str, router_key: str):
    """Score classifier against labeled session openers."""
    target_path = resolve_safe_path(labeled_path)
    safe_router_url = validate_safe_http_url(router_url).rstrip("/")
    samples = [json.loads(line) for line in target_path.read_text().strip().split("\n")]

    results = []
    async with httpx.AsyncClient(timeout=30) as client:
        for sample in samples:
            expected = sample["expected_level"]
            payload = {
                "model": "smart-router/classify-only",
                "messages": sample["messages"],
            }
            headers = {
                "Authorization": f"Bearer {router_key}",
                "Content-Type": "application/json",
            }

            resp = await client.post(
                f"{safe_router_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            data = resp.json()
            predicted = data.get("level", "UNKNOWN")
            results.append({
                "expected": expected,
                "predicted": predicted,
                "correct": predicted == expected,
            })

    return results


def report(results):
    """Print evaluation report."""
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total > 0 else 0

    # Per-level stats
    per_level = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in results:
        exp, pred = r["expected"], r["predicted"]
        if exp == pred:
            per_level[exp]["tp"] += 1
        else:
            per_level[pred]["fp"] += 1
            per_level[exp]["fn"] += 1

    # Within-one-level accuracy
    within_one = sum(
        1 for r in results
        if abs(int(r["expected"][1]) - int(r["predicted"][1])) <= 1
    )

    # Severe under-classification
    severe_under = sum(
        1 for r in results
        if r["expected"] in ("L3", "L4") and r["predicted"] in ("L1",)
    )

    print(f"\n{'='*60}")
    print("Classifier Evaluation Report")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Exact accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"Within-one-level accuracy: {within_one}/{total} = {within_one/total:.1%}")
    print(f"Severe under-classification: {severe_under}/{total} = {severe_under/total:.1%}")
    print("\nPer-level precision/recall:")
    for level in ["L1", "L2", "L3", "L4"]:
        stats = per_level[level]
        tp = stats["tp"]
        precision = tp / (tp + stats["fp"]) if (tp + stats["fp"]) > 0 else 0
        recall = tp / (tp + stats["fn"]) if (tp + stats["fn"]) > 0 else 0
        print(f"  {level}: P={precision:.2f} R={recall:.2f} (tp={tp} fp={stats['fp']} fn={stats['fn']})")

    # Confusion matrix
    print("\nConfusion matrix (rows=expected, cols=predicted):")
    levels = ["L1", "L2", "L3", "L4"]
    header = "       " + "  ".join(f"{l:>3}" for l in levels)
    print(header)
    for exp in levels:
        row = [sum(1 for r in results if r["expected"] == exp and r["predicted"] == pred) for pred in levels]
        print(f"  {exp}  " + "  ".join(f"{v:>3}" for v in row))
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier accuracy")
    parser.add_argument("--labeled", required=True, help="Path to labeled JSONL file")
    parser.add_argument("--router-url", default="http://localhost:8080")
    parser.add_argument("--router-key", default="test-key")
    args = parser.parse_args()

    try:
        validated_url = validate_safe_http_url(args.router_url)
    except ValueError as err:
        print(f"[-] ERROR: Invalid --router-url argument: {err}")
        sys.exit(1)

    results = asyncio.run(evaluate(resolve_safe_path(args.labeled), validated_url, args.router_key))
    report(results)


if __name__ == "__main__":
    main()
