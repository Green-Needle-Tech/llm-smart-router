"""Tests for scaffolding stripping."""
from app.classify.scaffolding import CommonPrefixLearner, strip_sections, split_scaffolding


def test_common_prefix_learning():
    learner = CommonPrefixLearner(max_samples=5, min_prefix_chars=50)

    # Add samples with a common prefix
    s1 = "You are IrisBot, an AI agent. Task: fix bug"
    s2 = "You are IrisBot, an AI agent. Task: write test"
    s3 = "You are IrisBot, an AI agent. Task: deploy app"

    learner.add_sample(s1)
    learner.add_sample(s2)
    learner.add_sample(s3)

    prefix = learner.prefix
    assert "You are IrisBot, an AI agent." in prefix


def test_strip_prefix():
    learner = CommonPrefixLearner(max_samples=5, min_prefix_chars=10)
    base = "This is a common prefix that is long enough for testing. "
    s1 = base + "Task one."
    s2 = base + "Task two."
    learner.add_sample(s1)
    learner.add_sample(s2)

    # The learned prefix will be "This is a common prefix...Task " (shared chars)
    test_msg = base + "Task something new."
    stripped, chars = learner.strip(test_msg)
    assert chars > 0
    assert "something new" in stripped


def test_strip_sections():
    text = """# Soul
You are a deep reasoning agent.

# Memory
User built a distributed system last week.

# Task
Fix the typo in README.md"""

    patterns = [
        r"(?is)^#+\s*memory\b.*?(?=^#+\s|\Z)",
    ]
    cleaned, removed = strip_sections(text, patterns)
    assert "distributed system" not in cleaned
    assert "Fix the typo" in cleaned


def test_split_scaffolding_task_text():
    result = split_scaffolding(
        system_message="Big system prompt",
        user_messages=["user task"],
        task_text="Explicit task text",
    )
    assert result["task_user"] == "Explicit task text"
    assert result["stripped_by"] == ["task_text"]


def test_split_scaffolding_ignore_system():
    result = split_scaffolding(
        system_message="Big system prompt",
        user_messages=["user task"],
        ignore_system=True,
    )
    assert result["task_system"] == ""
    assert result["scaffolding_chars"] > 0


def test_split_scaffolding_disabled():
    result = split_scaffolding(
        system_message="System prompt",
        user_messages=["user task"],
        strip_enabled=False,
    )
    assert result["scaffolding_chars"] == 0
