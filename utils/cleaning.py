"""Utility helpers for text normalisation."""
from __future__ import annotations

import re
from core.segment import Segment

_RE_MULTISPACE = re.compile(r"\s+")


def normalise_whitespace(text: str) -> str:
    """Collapse consecutive whitespace characters."""

    return _RE_MULTISPACE.sub(" ", text).strip()


def clean_segment(segment: Segment) -> Segment:
    """Return a cleaned copy of *segment*."""

    cleaned_text = normalise_whitespace(segment.text)
    if cleaned_text == segment.text:
        return segment
    return Segment(
        text=cleaned_text,
        source=segment.source,
        page=segment.page,
        metadata=segment.metadata.copy(),
        segment_id=segment.segment_id,
    )

