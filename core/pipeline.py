"""High-level document parsing orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from core.registry import resolve_parser
from core.segment import Segment
from utils.chunking import chunk_segments
from utils.cleaning import clean_segment
from utils.logger import get_logger


LOGGER = get_logger(__name__)


def parse_document(path: Path, *, chunk: bool = True) -> List[Segment]:
    """Parse *path* into :class:`Segment` objects."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    LOGGER.info("Parsing document: %s", path)
    parser = resolve_parser(path)
    segments = parser.parse(path)
    LOGGER.debug("Produced %s raw segments", len(segments))

    cleaned = [clean_segment(segment) for segment in segments]
    LOGGER.debug("Cleaned segments: %s", len(cleaned))

    if chunk:
        chunks = list(chunk_segments(cleaned))
        LOGGER.debug("Chunked into %s segments", len(chunks))
        return chunks

    return cleaned


def parse_many(paths: Iterable[Path], *, chunk: bool = True) -> List[Segment]:
    """Parse multiple *paths* and return a combined list of segments."""

    result: List[Segment] = []
    for path in paths:
        result.extend(parse_document(Path(path), chunk=chunk))
    return result

