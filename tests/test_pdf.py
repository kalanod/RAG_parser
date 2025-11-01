import sys
import types
from pathlib import Path

from parsers.pdf_parser import PDFParser


def test_pdf_parser_returns_list(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    parser = PDFParser()
    segments = parser.parse(pdf_path)
    assert isinstance(segments, list)


def test_pdf_parser_uses_pymupdf_with_ocr(monkeypatch, tmp_path: Path):
    parser = PDFParser()

    class DummyPage:
        def get_text(self, mode: str | None = None):
            if mode == "rawdict":
                return {
                    "blocks": [
                        {"type": 0, "lines": [{"spans": [{"text": "Hello"}]}]},
                        {"type": 1, "image": 42},
                        {"type": 0, "lines": [{"spans": [{"text": "World"}]}]},
                    ]
                }
            return "fallback"

    class DummyDoc:
        def __iter__(self):
            return iter([DummyPage()])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def extract_image(self, xref: int):
            assert xref == 42
            return {"image": b"fake-bytes"}

    dummy_fitz = types.SimpleNamespace(open=lambda path: DummyDoc())
    monkeypatch.setitem(sys.modules, "fitz", dummy_fitz)
    monkeypatch.setattr(parser, "_ocr_with_paddle", lambda data: "")
    monkeypatch.setattr(parser, "_ocr_with_pytesseract", lambda data: "ImageText")

    text = parser._extract_with_pymupdf(tmp_path / "dummy.pdf")
    assert "Hello" in text
    assert "ImageText" in text
    assert "World" in text
    assert "Hello\nImageText\nWorld" in text


def test_pdf_parser_falls_back_to_plain_text(monkeypatch, tmp_path: Path):
    parser = PDFParser()

    class DummyPage:
        def get_text(self, mode: str | None = None):
            if mode == "rawdict":
                raise ValueError("no rawdict")
            return "plain text"

    class DummyDoc:
        def __iter__(self):
            return iter([DummyPage()])

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    dummy_fitz = types.SimpleNamespace(open=lambda path: DummyDoc())
    monkeypatch.setitem(sys.modules, "fitz", dummy_fitz)
    monkeypatch.setattr(parser, "_perform_ocr", lambda data: "")

    text = parser._extract_with_pymupdf(tmp_path / "dummy.pdf")
    assert text == "plain text"

