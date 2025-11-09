from __future__ import annotations

from abc import abstractmethod, ABC
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document

from core.segment import Segment


class AbstractDocumentParser(ABC):
    supported_extensions: Iterable[str] = ()

    @abstractmethod
    def parse(self, path: Path, **kwargs: object) -> List[Document]:
        raise NotImplementedError

    @abstractmethod
    def split_to_docs(self, documents):
        raise NotImplementedError

