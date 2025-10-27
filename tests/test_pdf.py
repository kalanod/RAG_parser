from pathlib import Path

from parsers.pdf_parser import PDFParser


def test_pdf_parser_returns_list(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    parser = PDFParser()
    segments = parser.parse(pdf_path)
    assert isinstance(segments, list)


def test_pdf_parser_processes_semantic_blocks():
    parser = PDFParser()
    raw_text = (
        "ООО «Пример»\n"
        "Отчет по продажам\n\n"
        "Раздел 1\n\n"
        "инфор-\n"
        "мация о продажах за период.\n\n"
        "ООО «Пример»\n"
        "- Первый пункт списка\n"
        "- Второй пункт списка\n\n"
        "Товар    Кол-во    Цена\n"
        "Стол     3         1200\n"
        "Стул     5         800\n\n"
        "ООО «Пример»\n"
    )

    segments = list(parser.process_text(raw_text, path=Path("doc.pdf")))

    assert [segment.metadata["type"] for segment in segments] == [
        "heading",
        "heading",
        "paragraph",
        "list",
        "table",
    ]

    # Headers should be removed during cleaning.
    assert all("ООО «Пример»" not in segment.text for segment in segments)

    # Hyphenated words are restored.
    paragraph_texts = [seg.text for seg in segments if seg.metadata["type"] == "paragraph"]
    assert paragraph_texts and "информация" in paragraph_texts[0]

