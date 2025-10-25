"""Output writers for parsed segments."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from core.segment import Segment
from utils.formatting import to_jsonl


def write_jsonl(segments: Iterable[Segment], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in to_jsonl(segments):
            f.write(line + "\n")

