"""Live classifier-model benchmark: accuracy, latency, cost, parse reliability.

Uses the router's real classifier prompt + payload shape against OpenRouter.
"""
import json
import time
import httpx

PROMPT_FILE = "config/prompts/classifier.txt"
ENV_FILE = ".env"
MAX_TOKENS = 500  # raised from 60: reasoning models burn budget on hidden reasoning

CASES = [
    # (name, digest, expected)
    ("greeting", "hi", "UNKNOWN"),
    ("vague", "help me", "UNKNOWN"),
    ("trivial-upper", "Convert this list of 5 names to uppercase.", "L1"),
    ("trivial-extract", "Extract the email addresses from this text and list them.", "L1"),
    ("easy-fact", "What is the capital of Australia?", "L2"),
    ("easy-draft", "Write a one-paragraph birthday message for my colleague.", "L2"),
    ("medium-code", "Fix a race condition in our async job queue where workers process the same job twice. Analyze locking across 3 files and add tests.", "L3"),
    ("medium-analysis", "Compare PostgreSQL vs MongoDB for a social feed with 10k writes/sec. Give tradeoffs and a recommendation.", "L3"),
    ("hard-arch", "Design a multi-region event-driven architecture for a payment platform processing 50k TPS with exactly-once semantics.", "L4"),
    ("hard-math", "Prove or disprove: every bounded monotone sequence converges. Then extend to nets in general topological spaces.", "L4"),
    ("extreme-orch", "Orchestrate a team of AI agents to autonomously research, prototype, and validate a novel battery chemistry, coordinating lab equipment and literature review over weeks.", "L5"),
    ("extreme-novel", "Discover a new algorithm for online bipartite matching with competitive ratio better than 1-1/e, with full proofs.", "L5"),
]

MODELS = [
    "google/gemini-3.1-flash-lite",
    "google/gemini-2.5-flash-lite",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "deepseek/deepseek-v4-flash",
    "mistralai/mistral-small-3.2-24b-instruct",
]


def load_key():
    for line in open(ENV_FILE):
        if line.startswith("OPENROUTER_API_KEY="):
            return line.strip().split("=", 1)[1].strip('"').strip("'")
    raise RuntimeError("no key")


def run_case(client, key, model, prompt_template, digest, max_tokens=MAX_TOKENS):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_template.replace("{{PROMPT_DIGEST}}", digest)}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    t0 = time.monotonic()
    try:
        r = client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=20,
        )
        dt = time.monotonic() - t0
        if r.status_code != 200:
            return {"ok": False, "status": r.status_code, "latency": dt, "err": r.text[:100]}
        d = r.json()
        content = d["choices"][0]["message"]["content"]
        usage = d.get("usage", {})
        out = {
            "ok": True,
            "latency": dt,
            "cost": usage.get("cost", 0.0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
        if content is None:
            out["level"] = "NULL-CONTENT"
            out["ok"] = False
        else:
            try:
                j = json.loads(content)
                out["level"] = j.get("level", "?")
                out["confidence"] = j.get("confidence", 0)
            except Exception:
                out["level"] = "PARSE-FAIL"
                out["ok"] = False
        return out
    except Exception as e:
        return {"ok": False, "status": f"EXC:{type(e).__name__}", "latency": time.monotonic() - t0, "err": str(e)[:80]}


def main():
    key = load_key()
    prompt_template = open(PROMPT_FILE).read()
    client = httpx.Client()

    results = {}  # model -> list of case results
    for model in MODELS:
        results[model] = []
        for name, digest, expect in CASES:
            r = run_case(client, key, model, prompt_template, digest)
            r["case"] = name
            r["expect"] = expect
            results[model].append(r)
            time.sleep(0.3)  # be gentle with rate limits

    # Summary
    print(f"\n{'model':42s} {'acc':>5s} {'±1lvl':>6s} {'avg_lat':>8s} {'p95_lat':>8s} {'cost/call':>9s} {'fails':>5s}")
    print("-" * 90)
    summary = {}
    for model, rs in results.items():
        correct = sum(1 for r in rs if r.get("level") == r["expect"])
        fails = sum(1 for r in rs if not r.get("ok"))
        lats = sorted(r["latency"] for r in rs)
        avg_lat = sum(lats) / len(lats)
        p95 = lats[int(len(lats) * 0.95) - 1] if len(lats) > 1 else lats[0]
        costs = [r.get("cost", 0) for r in rs if r.get("ok")]
        cost_call = sum(costs) / len(costs) if costs else 0

        def lvl_num(l):
            order = ["UNKNOWN", "L1", "L2", "L3", "L4", "L5"]
            return order.index(l) if l in order else None

        within1 = 0
        for r in rs:
            a, b = lvl_num(r.get("level", "?")), lvl_num(r["expect"])
            if a is not None and b is not None and abs(a - b) <= 1:
                within1 += 1
        summary[model] = dict(acc=correct, within1=within1, fails=fails, avg_lat=avg_lat, cost=cost_call)
        print(f"{model:42s} {correct:2d}/12 {within1:4d}/12 {avg_lat:7.2f}s {p95:7.2f}s {cost_call:9.6f} {fails:5d}")

    # Per-case detail
    print("\nPer-case detail (level | latency | cost):")
    for model in MODELS:
        print(f"\n  {model}")
        for r in results[model]:
            mark = "OK " if r.get("level") == r["expect"] else "XX "
            print(f"    {mark}{r['case']:16s} exp={r['expect']:7s} got={str(r.get('level')):12s} {r['latency']:5.2f}s ${r.get('cost', 0):.6f}")

    with open("/root/llm-smart-router/scripts/bench_classifier_results.json", "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)
    print("\nSaved: scripts/bench_classifier_results.json")


if __name__ == "__main__":
    main()
