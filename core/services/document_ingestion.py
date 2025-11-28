from pathlib import Path
from typing import List

from langchain_core.documents import Document

from core.registry import resolve_parser
from utils.logger import get_logger


class DocumentIngestionService:
    def __init__(self):
        self.logger = get_logger(__name__)

    def validate_path(self, path: Path) -> Path:
        normalized = Path(path)
        if not normalized.exists():
            raise FileNotFoundError(normalized)
        return normalized

    def parse(self, path: Path) -> List[Document]:
        normalized_path = self.validate_path(path)
        self.logger.info("Parsing document: %s", normalized_path)
        parser = resolve_parser(normalized_path)
        segments = parser.parse(normalized_path)
        self.logger.debug("Produced %s raw segments", len(segments))
        return segments

    def clean(self, documents: List[Document]) -> List[Document]:
        return documents

    def parse_and_clean(self, path: Path) -> List[Document]:
        parsed = self.parse(path)
        return self.clean(parsed)