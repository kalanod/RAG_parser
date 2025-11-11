import sys
import types
from pathlib import Path

from core.pipeline import parse_document, create_embeddings
from parsers.pdf_parser import PDFParser


def test_pdf_parser_returns_list(tmp_path: Path):
    documents = parse_document(tmp_path)
    assert isinstance(documents, list)
