from pathlib import Path

import pytest

from parsers.pdf_parser import PDFParser


def test_pdf_parser_returns_list(tmp_path: Path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    parser = PDFParser()
    segments = parser.parse(pdf_path)
    assert isinstance(segments, list)


def test_pdf_process_text_cleans_and_segments():
    raw_text = (
        "Отчет 2023\n"
        "Конфиденциально\n"
        "\n"
        "1 Введение\n"
        "Эта инфор-\n"
        "мация описывает продукт.\n"
        "Продолжается описание.\n"
        "\n"
        "- первый пункт\n"
        "- второй пункт\n"
        "\n"
        "Имя    Значение\n"
        "foo    1\n"
        "bar    2\n"
        "\f"
        "Отчет 2023\n"
        "Конфиденциально\n"
    )

    parser = PDFParser()
    segments = list(parser.process_text(raw_text))

    assert [segment.metadata["type"] for segment in segments] == [
        "heading",
        "paragraph",
        "list",
        "table",
    ]
    assert segments[1].text.startswith("Эта информация описывает продукт.")
    assert "Конфиденциально" not in "\n".join(segment.text for segment in segments)


@pytest.mark.parametrize("max_chars", [80, 120])
def test_pdf_process_text_applies_size_constraints(max_chars: int):
    text = "Заголовок\n" + "Это предложение." * 20
    parser = PDFParser()
    segments = list(
        parser.process_text(
            text,
            max_block_chars=max_chars,
            min_block_chars=50,
        )
    )

    paragraph_segments = [segment for segment in segments if segment.metadata["type"] == "paragraph"]
    assert all(len(segment.text) <= max_chars for segment in paragraph_segments)


def test_pdf_process_text_preserves_paragraph_boundaries():
    text = "\n\n".join(
        [
            "Первый абзац короткий.",
            "Второй абзац тоже небольшой.",
            "Третий абзац завершает пример.",
        ]
    )

    parser = PDFParser()
    segments = list(
        parser.process_text(
            text,
            max_block_chars=70,
            min_block_chars=10,
        )
    )

    assert len(segments) == 2
    assert segments[0].metadata["type"] == "paragraph"
    assert segments[1].metadata["type"] == "paragraph"
    assert "Первый абзац" in segments[0].text
    assert "Второй абзац" in segments[0].text
    assert "Третий абзац" in segments[1].text

