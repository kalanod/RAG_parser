"""Helpers for working with file paths and metadata."""
from __future__ import annotations

import mimetypes
from pathlib import Path


def guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    return mime or "application/octet-stream"


def safe_filename(name: str) -> str:
    return "".join(ch for ch in name if ch.isalnum() or ch in {"-", "_", "."})

