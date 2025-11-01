"""PDF parser built on top of optional third-party libraries."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List, Optional

from core.segment import Segment
from parsers.base_parser import SimpleParser
from utils.logger import get_logger

LOGGER = get_logger(__name__)


class PDFParser(SimpleParser):
    supported_extensions = (".pdf",)

    def __init__(self) -> None:
        self._paddle_ocr: Optional[object] = None

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
                page_text = self._extract_page_with_images(page, doc)
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)

    def _extract_page_with_images(self, page: object, doc: object) -> str:
        try:
            raw_dict = page.get_text("rawdict")
        except Exception:  # pragma: no cover - defensive fallback
            return page.get_text()

        blocks = raw_dict.get("blocks", []) if isinstance(raw_dict, dict) else []
        page_parts: List[str] = []
        for block in blocks:
            block_type = block.get("type")
            if block_type == 0:
                text = self._extract_text_block(block)
                if text:
                    page_parts.append(text)
            elif block_type == 1:
                text = self._extract_image_block(block, doc)
                if text:
                    page_parts.append(text)
        return "\n".join(page_parts)

    def _extract_text_block(self, block: dict) -> str:
        lines = block.get("lines", [])
        parts: List[str] = []
        for line in lines:
            spans = line.get("spans", [])
            for span in spans:
                text = span.get("text")
                if text:
                    parts.append(text)
        return "".join(parts).strip()

    def _extract_image_block(self, block: dict, doc: object) -> str:
        xref = block.get("image")
        if not xref:
            return ""
        try:
            image_info = doc.extract_image(xref)
        except Exception:  # pragma: no cover - defensive fallback
            return ""
        image_bytes: Optional[bytes] = image_info.get("image") if image_info else None
        if not image_bytes:
            return ""
        return self._perform_ocr(image_bytes)

    def _perform_ocr(self, image_bytes: bytes) -> str:
        text = self._ocr_with_paddle(image_bytes)
        if text:
            return text.strip()
        text = self._ocr_with_pytesseract(image_bytes)
        return text.strip() if text else ""

    def _ocr_with_paddle(self, image_bytes: bytes) -> str:
        try:
            from PIL import Image  # type: ignore
            import numpy as np  # type: ignore
            from paddleocr import PaddleOCR  # type: ignore
        except Exception:
            return ""

        if self._paddle_ocr is None:
            try:
                self._paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")  # pragma: no cover
            except Exception:
                return ""

        try:
            image = Image.open(BytesIO(image_bytes))
        except Exception:
            return ""
        try:
            array = np.array(image.convert("RGB"))
        except Exception:
            return ""

        try:
            result = self._paddle_ocr.ocr(array, cls=True)
        except Exception:  # pragma: no cover - OCR backend failure
            return ""

        lines = [line[1][0] for page in result for line in page]
        return "\n".join(lines)

    def _ocr_with_pytesseract(self, image_bytes: bytes) -> str:
        try:
            from PIL import Image  # type: ignore
            import pytesseract  # type: ignore
        except Exception:
            return ""

        try:
            image = Image.open(BytesIO(image_bytes))
        except Exception:
            return ""
        try:
            return pytesseract.image_to_string(image)
        except Exception:  # pragma: no cover - OCR backend failure
            return ""

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

