#!/usr/bin/env python3
"""Evaluate classifier accuracy against a labeled set."""
import argparse
import asyncio
import json
from collections import defaultdict
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


async def evaluate(labeled_path: str | Path, router_url: str, router_key: str):
    """Score classifier against labeled session openers."""
    target_path = resolve_safe_path(labeled_path)
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
                f"{router_url}/v1/chat/completions",
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
    print(f"Classifier Evaluation Report")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Exact accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"Within-one-level accuracy: {within_one}/{total} = {within_one/total:.1%}")
    print(f"Severe under-classification: {severe_under}/{total} = {severe_under/total:.1%}")
    print(f"\nPer-level precision/recall:")
    for level in ["L1", "L2", "L3", "L4"]:
        stats = per_level[level]
        tp = stats["tp"]
        precision = tp / (tp + stats["fp"]) if (tp + stats["fp"]) > 0 else 0
        recall = tp / (tp + stats["fn"]) if (tp + stats["fn"]) > 0 else 0
        print(f"  {level}: P={precision:.2f} R={recall:.2f} (tp={tp} fp={stats['fp']} fn={stats['fn']})")

    # Confusion matrix
    print(f"\nConfusion matrix (rows=expected, cols=predicted):")
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

    results = asyncio.run(evaluate(resolve_safe_path(args.labeled), args.router_url, args.router_key))
    report(results)


if __name__ == "__main__":
    main()
