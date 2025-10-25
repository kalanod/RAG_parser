from pathlib import Path

import pytest

from core.pipeline import parse_document
from core.registry import available_extensions


def test_available_extensions_contains_expected_types():
    expected = {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".csv", ".png", ".jpg", ".jpeg"}
    assert expected.issubset(set(available_extensions()))


def test_parse_document_handles_missing_file(tmp_path: Path):
    path = tmp_path / "missing.pdf"
    with pytest.raises(FileNotFoundError):
        parse_document(path)

