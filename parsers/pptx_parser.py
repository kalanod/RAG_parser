"""PPTX slide deck parser."""
from __future__ import annotations

from pathlib import Path
from typing import List

from core.segment import Segment
from parsers.base_parser import DocumentParser
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class PPTXParser(DocumentParser):
    supported_extensions = (".pptx",)

    def parse(self, path: Path, **kwargs: object) -> List[Segment]:
        text = self._extract_with_python_pptx(path)
        if not text:
            LOGGER.warning(
                "No PPTX extraction backend available for %s. Returning empty result.",
                path,
            )
            return []
        return [Segment.from_text(text=text, source=str(path))]

    def _extract_with_python_pptx(self, path: Path) -> str:
        try:
            from pptx import Presentation  # type: ignore
        except Exception:
            return ""

        try:
            presentation = Presentation(path)  # pragma: no cover
        except Exception:
            return ""
        lines = []
        for index, slide in enumerate(presentation.slides, start=1):
            lines.append(f"# Slide {index}")
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    lines.append(shape.text)
        return "\n".join(lines)

