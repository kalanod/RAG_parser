"""Common data model for parsed document segments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import uuid


def _generate_id() -> str:
    return uuid.uuid4().hex


@dataclass
class Segment:
    """Represents a fragment of text extracted from a document."""

    text: str
    source: str
    page: Optional[int] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    segment_id: str = field(default_factory=_generate_id)

    @classmethod
    def from_text(
        cls,
        *,
        text: str,
        source: str,
        page: Optional[int] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> "Segment":
        return cls(
            text=text,
            source=source,
            page=page,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable representation of the segment."""

        return {
            "id": self.segment_id,
            "text": self.text,
            "source": self.source,
            "page": self.page,
            "metadata": self.metadata,
        }

