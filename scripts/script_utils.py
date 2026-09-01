#!/usr/bin/env python3
"""Shared helper functions for CLI utility scripts."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

_ALLOWED_HTTP_SCHEMES = frozenset({"http", "https"})
_ALLOWED_HOST_PATTERN = re.compile(
    r"^(?:localhost|127(?:\.\d+){1,3}|\[::1\]|(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,})$"
)


def validate_safe_http_url(
    url_str: str,
    allowed_schemes: frozenset[str] = _ALLOWED_HTTP_SCHEMES,
) -> str:
    """Validate and sanitize an HTTP/HTTPS URL string before network execution.

    Protects against SSRF, scheme injection (e.g., file://, gopher://), and malformed endpoints.
    """
    cleaned = url_str.strip()
    if not cleaned:
        raise ValueError("URL string cannot be empty.")

    parsed = urlparse(cleaned)
    scheme = parsed.scheme.lower()
    if scheme not in allowed_schemes:
        raise ValueError(
            f"Invalid URL scheme '{scheme}': must be one of {sorted(allowed_schemes)}"
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"URL '{url_str}' is missing a valid hostname.")

    if not _ALLOWED_HOST_PATTERN.match(hostname):
        raise ValueError(f"URL hostname '{hostname}' contains invalid characters or format.")

    if parsed.port is not None and not (1 <= parsed.port <= 65535):
        raise ValueError(f"URL port {parsed.port} is out of valid range (1-65535).")

    return cleaned


def resolve_safe_path(path_str: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve and validate that a file path exists and is a regular file."""
    base = (base_dir or Path.cwd()).resolve()
    resolved = (base / path_str if not Path(path_str).is_absolute() else Path(path_str)).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Path is not a regular file: {resolved}")
    return resolved


def resolve_safe_output_path(path_str: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve and validate an output file path before writing to the file system."""
    base = (base_dir or Path.cwd()).resolve()
    resolved = (base / path_str if not Path(path_str).is_absolute() else Path(path_str)).resolve()
    parent = resolved.parent
    if not parent.exists():
        raise FileNotFoundError(f"Destination directory not found: {parent}")
    if not parent.is_dir():
        raise NotADirectoryError(f"Destination parent is not a directory: {parent}")
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"Destination path exists and is not a regular file: {resolved}")
    return resolved
