"""DOCX document parser."""
from __future__ import annotations

from pathlib import Path
from typing import List

from core.segment import Segment
from parsers.base_parser import SimpleParser
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class DOCXParser(SimpleParser):
    supported_extensions = (".docx",)

    def parse(self, path: Path, **kwargs: object) -> List[Segment]:
        try:
            return super().parse(path, **kwargs)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("Failed to parse DOCX %s: %s", path, exc)
            return [
                Segment.from_text(
                    text="",
                    source=str(path),
                    metadata={"error": str(exc)},
                )
            ]

    def extract_text(self, path: Path, **kwargs: object) -> str:
        text = self._extract_with_docx(path)
        if text:
            return text
        text = self._extract_with_docx2python(path)
        if text:
            return text
        LOGGER.warning(
            "No DOCX extraction backend available for %s. Returning empty result.",
            path,
        )
        return ""

    def _extract_with_docx(self, path: Path) -> str:
        try:
            from docx import Document  # type: ignore
        except Exception:
            return ""

        document = Document(str(path))  # pragma: no cover
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    def _extract_with_docx2python(self, path: Path) -> str:
        try:
            from docx2python import docx2python  # type: ignore
        except Exception:
            return ""

        with docx2python(str(path)) as doc:  # pragma: no cover
            return "\n".join(doc.text.splitlines())

