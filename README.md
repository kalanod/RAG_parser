# RAG Parser

Модульная система парсинга документов для Retrieval-Augmented Generation.

## Возможности

- Общая модель сегмента (:mod:`core.segment`).
- Реестр парсеров для основных типов документов (:mod:`core.registry`).
- Единый пайплайн парсинга с очисткой и чанкингом (:mod:`core.pipeline`).
- Парсеры для PDF, DOCX, XLSX/CSV, PPTX и изображений с OCR.
- Утилиты для очистки текста, чанкинга и форматирования.
- Экспорт результатов в JSONL.

## Установка

```bash
pip install -e .
```

При необходимости установите дополнительные зависимости через extras:

```bash
pip install -e .[pdf,docs,spreadsheets,pptx,ocr]
```

## Использование

```bash
python examples/demo_parse.py path/to/document.pdf
```

## Тестирование

```bash
pytest
```

