"""Single source of truth for application versioning."""
from __future__ import annotations

# Application version — must match pyproject.toml [project] version.
APPLICATION_VERSION = "2.16.3"

# Configuration schema version — used in settings.json.
# This is independent of the application version; it only changes when the
# settings.json schema changes in a backwards-incompatible way.
CONFIG_SCHEMA_VERSION = 1
