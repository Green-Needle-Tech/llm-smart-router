#!/usr/bin/env python3
"""Shared helper functions for CLI utility scripts."""
from __future__ import annotations

from pathlib import Path


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
