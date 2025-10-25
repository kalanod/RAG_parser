"""Formatting utilities for exporting parsed segments."""
from __future__ import annotations

import json
from typing import Iterable, Iterator

from core.segment import Segment


def to_markdown(segments: Iterable[Segment]) -> str:
    """Render *segments* into a Markdown document."""

    lines = []
    for segment in segments:
        lines.append(f"### Source: {segment.source}")
        if segment.page is not None:
            lines.append(f"- Page: {segment.page}")
        lines.append("")
        lines.append(segment.text)
        lines.append("")
    return "\n".join(lines)


def to_jsonl(segments: Iterable[Segment]) -> Iterator[str]:
    """Yield JSON lines representing *segments*."""

    for segment in segments:
        yield json.dumps(segment.to_dict(), ensure_ascii=False)

