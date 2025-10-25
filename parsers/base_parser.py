"""Base interfaces for document parsers used in the RAG pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, List

from core.segment import Segment


class DocumentParser(ABC):
    """Abstract document parser that produces :class:`Segment` objects."""

    supported_extensions: Iterable[str] = ()

    @abstractmethod
    def parse(self, path: Path, **kwargs: object) -> List[Segment]:
        """Parse *path* into a list of :class:`Segment` objects."""
        raise NotImplementedError


class SimpleParser(DocumentParser):
    """Convenience parser that returns an empty result.

    Subclasses can override :meth:`process_text` to transform extracted text
    into :class:`Segment` objects. This class is intentionally conservative so
    that optional dependencies can be added later without breaking imports.
    """

    supported_extensions: Iterable[str] = ()

    def parse(self, path: Path, **kwargs: object) -> List[Segment]:
        text = self.extract_text(path, **kwargs)
        return list(self.process_text(text, path=path))

    def extract_text(self, path: Path, **kwargs: object) -> str:
        """Extract raw text from *path*.

        The default implementation simply returns an empty string.
        """

        return ""

    def process_text(self, text: str, **kwargs: object) -> Iterable[Segment]:
        """Convert *text* into an iterable of :class:`Segment` objects."""

        if text:
            yield Segment.from_text(text=text, source=str(kwargs.get("path", "")))

