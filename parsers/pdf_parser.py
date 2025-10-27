"""PDF parser that performs structural segmentation for RAG pipelines."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

from core.segment import Segment
from parsers.base_parser import SimpleParser
from utils.logger import get_logger

LOGGER = get_logger(__name__)


_RE_BULLET = re.compile(r"^(?:[-*\u2022\u2023\u25E6\u2043\u2219]|\d+[.)])\s+")
_RE_TABLE_SEP = re.compile(r"\t|\s{2,}")
_RE_HEADING = re.compile(r"^[^.!?]{0,120}$")
_RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9])")


@dataclass
class _Block:
    text: str
    type: str


class PDFParser(SimpleParser):
    """Parser that extracts and chunks PDF documents into semantic segments."""

    supported_extensions = (".pdf",)

    def parse(self, path: Path, **kwargs: object) -> List[Segment]:
        try:
            text = self.extract_text(path, **kwargs)
            return list(self.process_text(text, path=path, **kwargs))
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

    # ------------------------------------------------------------------
    # Text post-processing helpers

    def process_text(
        self,
        text: str,
        *,
        path: Path | None = None,
        max_block_chars: int = 1200,
        min_block_chars: int = 200,
        **_: object,
    ) -> Iterable[Segment]:
        if not text.strip():
            return []

        cleaned = self._clean_text(text)
        blocks = list(self._build_blocks(cleaned))
        sized_blocks = self._apply_size_constraints(
            blocks, max_block_chars=max_block_chars, min_block_chars=min_block_chars
        )

        source = str(path) if path else ""
        for index, block in enumerate(sized_blocks):
            metadata = {"type": block.type, "order": index}
            yield Segment.from_text(text=block.text, source=source, metadata=metadata)

    # ------------------------------------------------------------------
    # Cleaning

    def _clean_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
        pages = [page for page in text.split("\f")]
        repeated = self._detect_repeated_lines(pages)
        lines: List[str] = []
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if line and line in repeated:
                continue
            lines.append(raw_line)
        cleaned = "\n".join(lines)
        cleaned = re.sub(r"(\w)[-\u2010\u2011]\s*\n(\w)", r"\1\2", cleaned)
        cleaned = re.sub(r"\s+\n", "\n", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _detect_repeated_lines(self, pages: Sequence[str]) -> set[str]:
        if not pages:
            return set()

        page_count = max(len(pages), 1)
        counter: Counter[str] = Counter()
        for page in pages:
            lines = [line.strip() for line in page.splitlines() if line.strip()]
            if not lines:
                continue
            candidates = lines[:5] + lines[-5:]
            counter.update(candidates)

        threshold = max(2, int(page_count * 0.5))
        return {line for line, count in counter.items() if count >= threshold}

    # ------------------------------------------------------------------
    # Block construction

    def _build_blocks(self, text: str) -> Iterator[_Block]:
        current_lines: List[str] = []
        current_type: str | None = None

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                if current_lines:
                    yield self._flush_block(current_lines, current_type or "paragraph")
                    current_lines = []
                    current_type = None
                continue

            line_type = self._classify_line(line)
            if line_type == "heading":
                if current_lines:
                    yield self._flush_block(current_lines, current_type or "paragraph")
                    current_lines = []
                    current_type = None
                yield _Block(text=line, type="heading")
                continue

            if current_type is None:
                current_type = line_type
                current_lines = [line]
                continue

            if line_type == current_type:
                current_lines.append(line)
                continue

            yield self._flush_block(current_lines, current_type)
            current_type = line_type
            current_lines = [line]

        if current_lines:
            yield self._flush_block(current_lines, current_type or "paragraph")

    def _flush_block(self, lines: List[str], block_type: str) -> _Block:
        if block_type == "paragraph":
            text = " ".join(lines)
        else:
            text = "\n".join(lines)
        return _Block(text=text.strip(), type=block_type)

    def _classify_line(self, line: str) -> str:
        if self._is_list_item(line):
            return "list"
        if self._looks_like_table_row(line):
            return "table"
        if self._looks_like_heading(line):
            return "heading"
        return "paragraph"

    def _is_list_item(self, line: str) -> bool:
        return bool(_RE_BULLET.match(line))

    def _looks_like_table_row(self, line: str) -> bool:
        if "|" in line or "\t" in line:
            return True
        if _RE_TABLE_SEP.search(line):
            chunks = [chunk for chunk in re.split(r"\s{2,}", line) if chunk.strip()]
            return len(chunks) >= 2
        return False

    def _looks_like_heading(self, line: str) -> bool:
        words = line.split()
        if len(words) <= 1:
            return line.isupper() or line.istitle()
        if len(words) <= 12 and _RE_HEADING.match(line):
            if line.endswith(":"):
                return True
            if line.isupper():
                return True
            if line.istitle() and not line.endswith("."):
                return True
            if line[0].isupper() and not line.endswith("."):
                return True
        return False

    # ------------------------------------------------------------------
    # Block sizing

    def _apply_size_constraints(
        self,
        blocks: Sequence[_Block],
        *,
        max_block_chars: int,
        min_block_chars: int,
    ) -> List[_Block]:
        sized: List[_Block] = []
        for block in blocks:
            sized.extend(self._split_block(block, max_block_chars))

        merged: List[_Block] = []
        for block in sized:
            if (
                block.type != "heading"
                and merged
                and merged[-1].type == block.type
                and len(block.text) < min_block_chars
                and len(merged[-1].text) + len(block.text) <= max_block_chars
            ):
                merged[-1] = _Block(
                    text=self._merge_block_text(merged[-1], block),
                    type=merged[-1].type,
                )
                continue
            merged.append(block)

        return merged

    def _split_block(self, block: _Block, max_block_chars: int) -> List[_Block]:
        if len(block.text) <= max_block_chars:
            return [block]

        if block.type in {"list", "table"}:
            lines = block.text.split("\n")
            chunks: List[_Block] = []
            buffer: List[str] = []
            for line in lines:
                candidate = "\n".join(buffer + [line]).strip()
                if buffer and len(candidate) > max_block_chars:
                    chunks.append(_Block(text="\n".join(buffer).strip(), type=block.type))
                    buffer = [line]
                else:
                    buffer.append(line)
            if buffer:
                chunks.append(_Block(text="\n".join(buffer).strip(), type=block.type))
            return chunks

        sentences = _RE_SENTENCE_SPLIT.split(block.text)
        if len(sentences) == 1:
            words = block.text.split()
            chunks: List[_Block] = []
            buffer: List[str] = []
            for word in words:
                buffer.append(word)
                if len(" ".join(buffer)) > max_block_chars:
                    overflow = buffer.pop()
                    chunks.append(_Block(text=" ".join(buffer).strip(), type=block.type))
                    buffer = [overflow]
            if buffer:
                chunks.append(_Block(text=" ".join(buffer).strip(), type=block.type))
            return [chunk for chunk in chunks if chunk.text]

        chunks: List[_Block] = []
        buffer: List[str] = []
        for sentence in sentences:
            candidate = " ".join(buffer + [sentence]).strip()
            if buffer and len(candidate) > max_block_chars:
                chunks.append(_Block(text=" ".join(buffer).strip(), type=block.type))
                buffer = [sentence]
            else:
                buffer.append(sentence)
        if buffer:
            chunks.append(_Block(text=" ".join(buffer).strip(), type=block.type))
        return chunks

    def _merge_block_text(self, first: _Block, second: _Block) -> str:
        if first.type == "paragraph":
            return f"{first.text} {second.text}".strip()
        return f"{first.text}\n{second.text}".strip()

