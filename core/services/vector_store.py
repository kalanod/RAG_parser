"""Vector store service abstraction around Chroma.

This module stays intentionally lightweight. Additional CRUD operations
(such as filtered deletes or updates) should be implemented when the
vector lifecycle is defined.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from core.pipeline import load_from_chroma
from utils.logger import get_logger


class VectorStoreService:
    """CRUD wrapper around the Chroma vector store."""

    def __init__(
        self,
        collection_name: str,
        embedder,
        persist_directory: str = "./db",
    ):
        self.logger = get_logger(__name__)
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedder = embedder
        self._db: Optional[Chroma] = None
        self.id_cache = set()
        self._init_db()

    def _init_db(self) -> None:
        self._db = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embedder,
        )
        self.id_cache = set(self._db.get(include=[]).get("ids", []))

    @property
    def db(self) -> Chroma:
        if self._db is None:
            self._db = load_from_chroma(
                self.persist_directory, self.collection_name, self.embedder
            )
        return self._db

    def get_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def add_documents(self, documents: List[Document]) -> None:
        new_docs: List[Document] = []
        new_ids: List[str] = []

        for doc in documents:
            doc_id = self.get_hash(doc.page_content)
            if doc_id in self.id_cache:
                continue
            self.id_cache.add(doc_id)
            new_docs.append(doc)
            new_ids.append(doc_id)

        if new_docs:
            self.db.add_documents(new_docs, ids=new_ids)

    def search(self, query: str, top_k: int = 5):
        self.logger.info(
            "collection: %s - search in chroma: %s", self.collection_name, query
        )
        results = self.db.similarity_search(query, k=top_k)
        self.logger.info(
            "collection: %s - search in chroma results: %s", self.collection_name, results
        )
        return results

    def delete(self, ids: Optional[List[str]] = None) -> None:
        """Delete entries from the vector store.

        TODO: Implement delete logic for specific document ids or sources.
        """
        _ = ids
        # TODO: call Chroma delete API when deletion strategy is defined.
