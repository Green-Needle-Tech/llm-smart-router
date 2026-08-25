
import re

# Define regex patterns for common temporal expressions
# These patterns will be used by the TemporalAwarenessEngine

TEMPORAL_EXPRESSION_PATTERNS = [
    # Specific days
    (r"\btoday\b", "today"),
    (r"\byesterday\b", "yesterday"),
    (r"\btomorrow\b", "tomorrow"),

    # Days of the week
    (r"\b(last|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", "relative_day_of_week"),
    (r"\b(this|coming)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", "this_coming_day_of_week"),

    # Relative periods
    (r"\blast\s+week\b", "last_week"),
    (r"\bthis\s+week\b", "this_week"),
    (r"\bnext\s+week\b", "next_week"),
    (r"\blast\s+month\b", "last_month"),
    (r"\bthis\s+month\b", "this_month"),
    (r"\bnext\s+month\b", "next_month"),
    (r"\blast\s+year\b", "last_year"),
    (r"\bthis\s+year\b", "this_year"),
    (r"\bnext\s+year\b", "next_year"),

    # N days/weeks/months/years ago/from now
    (r"\b(last|past)\s+(\d+)\s+(day|week|month|year)s?\b", "past_n_units"),
    (r"\b(\d+)\s+(day|week|month|year)s?\s+ago\b", "n_units_ago"),
    (r"\bin\s+(\d+)\s+(day|week|month|year)s?\b", "in_n_units"),

    # Date ranges (e.g., "from X to Y") - more complex, might need custom parsing
    # (r"from\s+(.+?)\s+to\s+(.+?)", "date_range"),

    # Time expressions (e.g., "morning", "afternoon", "evening") - for future expansion
    # (r"\b(morning|afternoon|evening|night)\b", "time_of_day"),
]

# Compile all patterns for efficiency
COMPILED_TEMPORAL_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), tag) for pattern, tag in TEMPORAL_EXPRESSION_PATTERNS
]
