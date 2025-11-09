"""High-level document parsing orchestration."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from tqdm import tqdm

from core.registry import resolve_parser
from core.segment import Segment
from utils.chunking import chunk_segments
from utils.cleaning import clean_segment
from utils.logger import get_logger


LOGGER = get_logger(__name__)


def parse_document(path: Path,) -> List[Document]:
    """Parse *path* into :class:`Segment` objects."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    LOGGER.info("Parsing document: %s", path)
    parser = resolve_parser(path)
    segments = parser.parse(path)
    LOGGER.debug("Produced %s raw segments", len(segments))
    return segments


def parse_many(paths: Iterable[Path], *, chunk: bool = True) -> List[Segment]:
    """Parse multiple *paths* and return a combined list of segments."""

    result: List[Segment] = []
    for path in paths:
        result.extend(parse_document(Path(path), chunk=chunk))
    return result

def create_embeddings(data, model: str = "text-embedding-3-small"):
    client = OpenAI(
        base_url="https://ai-for-finance-hack.up.railway.app",
        api_key="sk-fadOQ_Pb4hU8f73BQzeTwQ"
    )
    res = []
    for i in tqdm(range(len(data)), desc="Эмбединги",
            ncols=80):
        response = client.embeddings.create(
            input=data[i].page_content,
            model=model
        )
        res.append(response.data[0].embedding)
    return res
def save_to_chroma(
    documents: List[Document],
    embeddings: Embeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "pdf_collection"
):
    # Инициализация или создание коллекции
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    # Сохраняем на диск
    vectorstore.persist()
    print(f"✅ Chroma база успешно сохранена в '{persist_directory}' (коллекция: {collection_name})")

    return vectorstore


def load_from_chroma(
    embeddings: Embeddings,
    persist_directory: str = "./chroma_db",
    collection_name: str = "pdf_collection"
):
    """
    Загружает ранее сохранённую Chroma-базу для последующего поиска.
    """
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print(f"📂 Chroma база загружена из '{persist_directory}' (коллекция: {collection_name})")
    return vectorstore

from langchain.vectorstores import Chroma
from langchain.schema import Document
import numpy as np
from typing import List, Tuple


def _sanitize_metadata(metadata: dict) -> dict:
    safe_meta = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe_meta[k] = v
        else:
            safe_meta[k] = str(v)
    return safe_meta


def init_chroma_with_embeddings(
    persist_directory: str,
    documents: List[Document],
    embeddings: List[List[float]],
    collection_name: str = "rag_store"
) -> Chroma:
    vectors = np.array(embeddings, dtype=np.float32)
    if len(vectors) != len(documents):
        raise ValueError("Количество эмбеддингов и документов должно совпадать.")

    db = Chroma(
        collection_name=collection_name,
        embedding_function=None,
        persist_directory=persist_directory
    )
    metadatas = [_sanitize_metadata(d.metadata) for d in documents]

    db._collection.add(
        embeddings=vectors,
        metadatas=metadatas,
        documents=[d.page_content for d in documents],
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    db.persist()
    print(f"✅ Добавлено {len(documents)} документов в Chroma (путь: {persist_directory})")
    return db


def search_in_chroma(
    db: Chroma,
    query_vector: List[float],
    top_k: int = 5
) -> List[Tuple[Document, float]]:

    results = db._collection.query(
        query_embeddings=query_vector,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    docs = []
    for i in range(len(results["documents"][0])):
        doc = Document(
            page_content=results["documents"][0][i],
            metadata=results["metadatas"][0][i]
        )
        distance = results["distances"][0][i]
        docs.append((doc, distance))

    return docs
