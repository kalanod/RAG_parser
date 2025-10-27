"""PDF parser built on top of optional third-party libraries."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Iterable, Iterator, List

from core.segment import Segment
from parsers.base_parser import SimpleParser
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
        cleaned = self._clean_text(text)
        if not cleaned:
            return

        blocks = list(self._split_into_blocks(cleaned))
        blocks = self._merge_blocks(blocks)
        blocks = self._enforce_max_size(blocks)

        for block in blocks:
            if not block.text:
                continue
            yield Segment.from_text(
                text=block.text,
                source=source,
                metadata={"type": block.type},
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

