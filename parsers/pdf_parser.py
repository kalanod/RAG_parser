"""PDF parser built on top of optional third-party libraries."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, Iterable, Iterator, List, Tuple

from core.segment import Segment
from parsers.base_parser import SimpleParser
from parsers.ocr_parser import ImageOCRParser
from utils.logger import get_logger

LOGGER = get_logger(__name__)


_RE_MULTISPACE_LINE = re.compile(r"[ \t]{2,}")
_RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class _Block:
    """Intermediate representation of a semantic text block."""

    text: str
    type: str


class PDFParser(SimpleParser):
    supported_extensions = (".pdf",)

    #: Minimal amount of characters a block should have before we try to merge it
    _min_block_size = 200
    #: Hard upper bound for a block size. Larger blocks are split by sentences.
    _max_block_size = 1200

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

    def process_text(self, text: str, **kwargs: object) -> Iterable[Segment]:
        """Convert raw text into cleaned, structured :class:`Segment` objects."""

        source = str(kwargs.get("path", ""))
        yield from self._build_segments(text, source=source)

    def _build_segments(
        self,
        text: str,
        *,
        source: str,
        metadata_factory: Callable[["_Block"], Dict[str, object]] | None = None,
        page_factory: Callable[["_Block"], int | None] | None = None,
    ) -> Iterable[Segment]:
        cleaned = self._clean_text(text)
        if not cleaned:
            return

        blocks = list(self._split_into_blocks(cleaned))
        blocks = self._merge_blocks(blocks)
        blocks = self._enforce_max_size(blocks)

        for block in blocks:
            if not block.text:
                continue
            metadata: Dict[str, object] = {"type": block.type}
            if metadata_factory is not None:
                extra = metadata_factory(block)
                if extra:
                    metadata.update(extra)
            page = page_factory(block) if page_factory is not None else None
            yield Segment.from_text(
                text=block.text,
                source=source,
                metadata=metadata,
                page=page,
            )

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

    # -- Text processing helpers -------------------------------------------------

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""

        normalised = text.replace("\r\n", "\n").replace("\r", "\n")
        # Remove hyphenation at line breaks (e.g. ``инфор-\nмация``)
        normalised = re.sub(r"(?<=\w)-\n(?=\w)", "", normalised)

        lines = [line.rstrip() for line in normalised.split("\n")]
        counter = Counter()
        for line in lines:
            normalised_line = re.sub(r"\s+", " ", line).strip()
            if normalised_line:
                counter[normalised_line] += 1

        cleaned_lines: List[str] = []
        for line in lines:
            normalised_line = re.sub(r"\s+", " ", line).strip()
            if not normalised_line:
                if cleaned_lines and cleaned_lines[-1] == "":
                    continue
                cleaned_lines.append("")
                continue

            # Drop lines that look like headers/footers repeated across pages.
            if counter[normalised_line] >= 3 and len(normalised_line) <= 80:
                continue

            cleaned_lines.append(line.strip())

        cleaned_text = "\n".join(cleaned_lines).strip()
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        return cleaned_text

    def _split_into_blocks(self, text: str) -> Iterator[_Block]:
        if not text:
            return

        for raw_block in re.split(r"\n{2,}", text):
            block = raw_block.strip()
            if not block:
                continue

            block_type = self._classify_block(block)
            normalised_text = self._normalise_block(block, block_type)
            yield _Block(text=normalised_text, type=block_type)

    def _classify_block(self, block: str) -> str:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            return "paragraph"

        if self._is_table_block(lines):
            return "table"

        if self._is_list_block(lines):
            return "list"

        if len(lines) == 1 and self._looks_like_heading(lines[0]):
            return "heading"

        return "paragraph"

    def _is_list_block(self, lines: List[str]) -> bool:
        list_markers = 0
        for line in lines:
            if re.match(r"^(?:[-*•]\s+|\d+[\).]\s+)", line):
                list_markers += 1
        return list_markers >= max(1, len(lines) // 2)

    def _is_table_block(self, lines: List[str]) -> bool:
        if any("|" in line or "\t" in line for line in lines):
            return True
        multi_space_lines = sum(1 for line in lines if _RE_MULTISPACE_LINE.search(line))
        return multi_space_lines >= max(2, len(lines) // 2)

    def _looks_like_heading(self, line: str) -> bool:
        text = line.strip()
        if len(text) > 120:
            return False
        if text.endswith(('.', '!', '?')):
            return False
        words = text.split()
        if len(words) <= 2:
            return True
        if len(words) <= 12 and any(ch.isupper() for ch in text if ch.isalpha()):
            if text.istitle():
                return True
            alpha_chars = [ch for ch in text if ch.isalpha()]
            if alpha_chars:
                uppercase_ratio = sum(1 for ch in alpha_chars if ch.isupper()) / len(alpha_chars)
                if uppercase_ratio >= 0.4:
                    return True
            if words[0][0].isupper():
                return True
        return False

    def _normalise_block(self, block: str, block_type: str) -> str:
        if block_type == "table":
            lines = [re.sub(r"\s+$", "", line) for line in block.splitlines()]
            return "\n".join(lines).strip()

        if block_type == "list":
            cleaned_lines = [re.sub(r"\s+", " ", line).strip() for line in block.splitlines() if line.strip()]
            return "\n".join(cleaned_lines)

        # Paragraphs and headings
        cleaned_lines = [re.sub(r"\s+", " ", line).strip() for line in block.splitlines() if line.strip()]
        if block_type == "paragraph":
            return " ".join(cleaned_lines)
        return " ".join(cleaned_lines)

    def _merge_blocks(self, blocks: List[_Block]) -> List[_Block]:
        if not blocks:
            return []

        merged: List[_Block] = []
        buffer: _Block | None = None

        for block in blocks:
            if block.type == "heading":
                if buffer is not None:
                    merged.append(buffer)
                    buffer = None
                merged.append(block)
                continue

            if buffer is None:
                buffer = block
                continue

            if buffer.type != block.type:
                merged.append(buffer)
                buffer = block
                continue

            combined_length = len(buffer.text) + 2 + len(block.text)
            if (len(buffer.text) < self._min_block_size or len(block.text) < self._min_block_size) and combined_length <= self._max_block_size:
                buffer = _Block(text=f"{buffer.text}\n\n{block.text}", type=buffer.type)
                continue

            if combined_length <= self._max_block_size:
                buffer = _Block(text=f"{buffer.text}\n\n{block.text}", type=buffer.type)
                continue

            merged.append(buffer)
            buffer = block

        if buffer is not None:
            merged.append(buffer)
        return merged

    def _enforce_max_size(self, blocks: List[_Block]) -> List[_Block]:
        limited: List[_Block] = []
        for block in blocks:
            if len(block.text) <= self._max_block_size or block.type != "paragraph":
                limited.append(block)
                continue

            sentences = _RE_SENTENCE_SPLIT.split(block.text)
            current: List[str] = []
            current_len = 0
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                if current_len + len(sentence) + (1 if current else 0) <= self._max_block_size:
                    current.append(sentence)
                    current_len += len(sentence) + (1 if current_len else 0)
                else:
                    limited.append(_Block(text=" ".join(current).strip(), type=block.type))
                    current = [sentence]
                    current_len = len(sentence)
            if current:
                limited.append(_Block(text=" ".join(current).strip(), type=block.type))

        return limited


class HybridPDFParser(PDFParser):
    """PDF parser that falls back to OCR when no text layer is detected."""

    #: Minimal amount of non-whitespace characters per document to consider it text-based
    _min_document_chars = 300
    #: Minimal amount of non-whitespace characters per page to consider it text-based
    _min_page_chars = 80

    def parse(self, path: Path, **kwargs: object) -> List[Segment]:
        text_pages, backend = self._extract_text_pages(path)
        if text_pages and self._looks_like_text_document(text_pages):
            LOGGER.debug(
                "Using text extraction backend '%s' for %s", backend or "unknown", path
            )
            text = "\n".join(page for page in text_pages if page)
            return list(self._build_segments(text, source=str(path)))

        LOGGER.info("Falling back to OCR for %s", path)
        ocr_segments = self._parse_with_ocr(path)
        if ocr_segments:
            return ocr_segments

        LOGGER.warning("OCR fallback failed for %s. Returning empty result.", path)
        return [
            Segment.from_text(
                text="",
                source=str(path),
                metadata={"error": "ocr_unavailable"},
            )
        ]

    # -- Text extraction helpers -------------------------------------------------

    def _extract_text_pages(self, path: Path) -> Tuple[List[str], str | None]:
        pages = self._extract_with_pymupdf_pages(path)
        if any(page.strip() for page in pages):
            return pages, "pymupdf"

        pages = self._extract_with_pdfplumber_pages(path)
        if any(page.strip() for page in pages):
            return pages, "pdfplumber"

        return [], None

    def _extract_with_pymupdf_pages(self, path: Path) -> List[str]:
        try:
            import fitz  # type: ignore
        except Exception:
            return []

        text_pages: List[str] = []
        try:
            with fitz.open(path) as doc:  # pragma: no cover - requires dependency
                for page in doc:
                    text_pages.append(page.get_text())
        except Exception as exc:  # pragma: no cover - requires dependency
            LOGGER.debug("PyMuPDF failed to extract %s: %s", path, exc)
        return text_pages

    def _extract_with_pdfplumber_pages(self, path: Path) -> List[str]:
        try:
            import pdfplumber  # type: ignore
        except Exception:
            return []

        text_pages: List[str] = []
        try:
            with pdfplumber.open(path) as pdf:  # pragma: no cover - requires dependency
                for page in pdf.pages:
                    text_pages.append(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - requires dependency
            LOGGER.debug("pdfplumber failed to extract %s: %s", path, exc)
        return text_pages

    def _looks_like_text_document(self, pages: List[str]) -> bool:
        if not pages:
            return False

        non_whitespace_total = sum(
            1 for page in pages for ch in page if not ch.isspace()
        )
        if non_whitespace_total >= self._min_document_chars:
            return True

        for page in pages:
            non_whitespace_page = sum(1 for ch in page if not ch.isspace())
            if non_whitespace_page >= self._min_page_chars:
                return True

        return False

    # -- OCR helpers -------------------------------------------------------------

    def _parse_with_ocr(self, path: Path) -> List[Segment]:
        source = str(path)
        with TemporaryDirectory() as tmpdir:
            page_images = list(self._export_pages_to_images(path, Path(tmpdir)))

            if not page_images:
                LOGGER.warning("Unable to render PDF %s for OCR", path)
                return []

            ocr_parser = ImageOCRParser()
            segments: List[Segment] = []
            for page_number, image_path in page_images:
                text = self._run_ocr(ocr_parser, image_path)
                if not text.strip():
                    continue

                segments.extend(
                    self._build_segments(
                        text,
                        source=source,
                        metadata_factory=lambda block, page=page_number: {
                            "source_type": "ocr",
                            "page": page,
                        },
                        page_factory=lambda block, page=page_number: page,
                    )
                )

        return segments

    def _export_pages_to_images(
        self, path: Path, tmpdir: Path
    ) -> Iterator[Tuple[int, Path]]:  # pragma: no cover - requires dependencies
        try:
            import fitz  # type: ignore
        except Exception:
            fitz = None

        if fitz is not None:
            try:
                with fitz.open(path) as doc:
                    for index, page in enumerate(doc, start=1):
                        pix = page.get_pixmap()
                        image_path = tmpdir / f"page_{index:04d}.png"
                        pix.save(str(image_path))
                        yield index, image_path
            except Exception as exc:
                LOGGER.debug("PyMuPDF failed to render %s: %s", path, exc)
            return

        try:
            from pdf2image import convert_from_path  # type: ignore
        except Exception:
            return

        try:
            images = convert_from_path(path)
        except Exception as exc:
            LOGGER.debug("pdf2image failed to render %s: %s", path, exc)
            return

        for index, image in enumerate(images, start=1):
            image_path = tmpdir / f"page_{index:04d}.png"
            image.save(image_path, format="PNG")
            yield index, image_path

    def _run_ocr(self, parser: ImageOCRParser, image_path: Path) -> str:
        segments = parser.parse(image_path)
        return "\n".join(segment.text for segment in segments if segment.text)
