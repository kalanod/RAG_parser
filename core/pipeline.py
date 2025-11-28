from pathlib import Path
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from core.services.document_ingestion import DocumentIngestionService
from utils.logger import get_logger

LOGGER = get_logger(__name__)


def parse_document(path: Path) -> List[Document]:
    ingestion_service = DocumentIngestionService()
    return ingestion_service.parse_and_clean(path)


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
