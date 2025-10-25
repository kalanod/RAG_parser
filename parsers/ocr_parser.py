"""OCR based parser for image documents."""
from __future__ import annotations

from pathlib import Path
from typing import List

from core.segment import Segment
from parsers.base_parser import DocumentParser
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class ImageOCRParser(DocumentParser):
    supported_extensions = (".png", ".jpg", ".jpeg")

    def parse(self, path: Path, **kwargs: object) -> List[Segment]:
        text = self._extract_with_paddle(path)
        if not text:
            text = self._extract_with_pytesseract(path)
        if not text:
            LOGGER.warning(
                "No OCR backend available for %s. Returning empty result.", path
            )
            return []
        return [Segment.from_text(text=text, source=str(path))]

    def _extract_with_paddle(self, path: Path) -> str:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception:
            return ""

        ocr = PaddleOCR(use_angle_cls=True, lang="en")  # pragma: no cover
        result = ocr.ocr(str(path), cls=True)
        lines = [line[1][0] for page in result for line in page]
        return "\n".join(lines)

    def _extract_with_pytesseract(self, path: Path) -> str:
        try:
            from PIL import Image  # type: ignore
            import pytesseract  # type: ignore
        except Exception:
            return ""

        image = Image.open(path)  # pragma: no cover
        return pytesseract.image_to_string(image)

