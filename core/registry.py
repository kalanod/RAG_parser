"""Parser registry that maps file extensions to parser classes."""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Dict, Iterable, Type

from parsers.base_parser import DocumentParser

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


def get_parser_class(extension: str) -> Type[DocumentParser]:
    """Return the parser class registered for *extension*.

    Raises ``KeyError`` if the extension is unknown.
    """

    try:
        dotted_path = _REGISTRY[extension.lower()]
    except KeyError as exc:
        raise KeyError(f"No parser registered for extension '{extension}'.") from exc

    module_name, class_name = dotted_path.rsplit(".", 1)
    module = import_module(module_name)
    parser_class = getattr(module, class_name)
    if not issubclass(parser_class, DocumentParser):
        raise TypeError(f"{parser_class!r} is not a DocumentParser subclass.")
    return parser_class


def resolve_parser(path: Path) -> DocumentParser:
    """Instantiate the parser for the file located at *path*."""

    extension = path.suffix.lower()
    parser_cls = get_parser_class(extension)
    return parser_cls()


def available_extensions() -> Iterable[str]:
    """Yield the extensions currently registered."""

    return tuple(sorted(_REGISTRY))

