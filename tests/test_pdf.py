from pathlib import Path

from parsers.pdf_parser import PDFParser


def test_pdf_parser_returns_list(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    parser = PDFParser()
    segments = parser.parse(pdf_path)
    assert isinstance(segments, list)

