import os
from pathlib import Path
from core.registry import resolve_parser
from utils.logger import get_logger
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List

LOGGER = get_logger(__name__)


def parse_document(path: Path) -> List[Document]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    LOGGER.info("Parsing document: %s", path)
    parser = resolve_parser(path)
    segments = parser.parse(path)
    LOGGER.debug("Produced %s raw segments", len(segments))
    return segments


def load_from_chroma(
        persist_directory: str = "./db",
        collection_name: str = "pdf_collection",
        embedding_function=None,
):
    vectorstore = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )
    print(f"Chroma база загружена из '{persist_directory}' (коллекция: {collection_name})")
    return vectorstore
