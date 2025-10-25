from pathlib import Path

from parsers.docx_parser import DOCXParser


def test_docx_parser_returns_list(tmp_path: Path):
    path = tmp_path / "sample.docx"
    path.write_bytes(b"")
    parser = DOCXParser()
    segments = parser.parse(path)
    assert isinstance(segments, list)

