from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Dict, Iterable, Type

from parsers.abstract_document_parser import AbstractDocumentParser

_REGISTRY: Dict[str, str] = {
    ".pdf": "parsers.pdf_parser.PDFParser",
    ".docx": "parsers.docx_parser.DOCXParser",
    ".pptx": "parsers.pptx_parser.PPTXParser",
    ".xlsx": "parsers.xlsx_parser.XLSXParser",
    ".xls": "parsers.xlsx_parser.XLSXParser",
    ".csv": "parsers.xlsx_parser.XLSXParser",
    ".png": "parsers.ocr_parser.ImageOCRParser",
    ".jpg": "parsers.ocr_parser.ImageOCRParser",
    ".jpeg": "parsers.ocr_parser.ImageOCRParser",
}


def get_parser_class(extension: str) -> Type[AbstractDocumentParser]:
    try:
        dotted_path = _REGISTRY[extension.lower()]
    except KeyError as exc:
        raise KeyError(f"No parser registered for extension '{extension}'.") from exc

    module_name, class_name = dotted_path.rsplit(".", 1)
    module = import_module(module_name)
    parser_class = getattr(module, class_name)
    if not issubclass(parser_class, AbstractDocumentParser):
        raise TypeError(f"{parser_class!r} is not a AbstractDocumentParser subclass.")
    return parser_class


def resolve_parser(path: Path) -> AbstractDocumentParser:
    extension = path.suffix.lower()
    parser_cls = get_parser_class(extension)
    return parser_cls()


def available_extensions() -> Iterable[str]:
    return tuple(sorted(_REGISTRY))

