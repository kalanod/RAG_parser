"""PDF parser built on top of optional third-party libraries."""
from __future__ import annotations

from pathlib import Path
from typing import List

from core.segment import Segment
from parsers.base_parser import SimpleParser
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class PDFParser(SimpleParser):
    supported_extensions = (".pdf",)

    def parse(self, path: Path, **kwargs: object) -> List[Segment]:
        try:
            return super().parse(path, **kwargs)
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOGGER.exception("Failed to parse PDF %s: %s", path, exc)
            return [
                Segment.from_text(
                    text="",
                    source=str(path),
                    metadata={"error": str(exc)},
                )
            ]

    def extract_text(self, path: Path, **kwargs: object) -> str:
        text = self._extract_with_pymupdf(path)
        if text:
            return text
        text = self._extract_with_pdfplumber(path)
        if text:
            return text
        LOGGER.warning(
            "No PDF extraction backend available for %s. Returning empty result.",
            path,
        )
        return ""

    def _extract_with_pymupdf(self, path: Path) -> str:
        try:
            import fitz  # type: ignore
        except Exception:
            return ""

        text_parts: List[str] = []
        with fitz.open(path) as doc:  # pragma: no cover - requires dependency
            for page in doc:
                text_parts.append(page.get_text())
        return "\n".join(text_parts)

    def _extract_with_pdfplumber(self, path: Path) -> str:
        try:
            import pdfplumber  # type: ignore
        except Exception:
            return ""

        text_parts: List[str] = []
        with pdfplumber.open(path) as pdf:  # pragma: no cover - requires dependency
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)

