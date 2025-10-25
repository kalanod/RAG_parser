"""Simple chunking helpers for RAG pipelines."""
from __future__ import annotations

from typing import Iterable, Iterator, List

from core.segment import Segment


def chunk_segments(segments: Iterable[Segment], *, max_tokens: int = 500) -> Iterator[Segment]:
    """Combine segments into larger chunks capped by *max_tokens* words."""

    buffer: List[str] = []
    metadata: List[Segment] = []

    for segment in segments:
        words = segment.text.split()
        if sum(len(s.text.split()) for s in metadata) + len(words) > max_tokens and buffer:
            yield _flush_buffer(buffer, metadata)
            buffer = []
            metadata = []

        buffer.append(segment.text)
        metadata.append(segment)

    if buffer:
        yield _flush_buffer(buffer, metadata)


def _flush_buffer(buffer: List[str], metadata: List[Segment]) -> Segment:
    text = "\n\n".join(buffer)
    merged_metadata = metadata[0].metadata.copy() if metadata else {}
    merged_metadata.update({"merged_segments": [seg.segment_id for seg in metadata]})
    return Segment.from_text(
        text=text,
        source=metadata[0].source if metadata else "",
        metadata=merged_metadata,
    )

