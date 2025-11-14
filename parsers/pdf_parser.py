from __future__ import annotations

import re

import pdfplumber
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List

from PIL import Image
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pytesseract import pytesseract

from parsers.abstract_document_parser import AbstractDocumentParser
from utils.logger import get_logger
import fitz

LOGGER = get_logger(__name__)


@dataclass
class PDFParserConfig:
    chunk_size: int = 500
    chunk_overlap: int = 80
    ocr_lang: str = "rus+eng"


class PDFParser(AbstractDocumentParser):
    supported_extensions = (".pdf",)
    config = PDFParserConfig()

    def parse(self, path: Path, **kwargs: object) -> List[Document]:
        try:
            documents = []
            documents.extend(self._extract_text_blocks(path))
            documents.extend(self._extract_ocr_blocks(path))
            documents.extend(self._extract_table_blocks(path))
            return self.split_to_docs(documents)
        except Exception as exc:
            LOGGER.exception("Failed to parse PDF %s: %s", path, exc)
            return [
                Document(
                    page_content=f"[ERROR] Failed to load document: {str(exc)}",
                    metadata={
                        "source": str(path),
                        "error": True,
                        "error_type": type(exc).__name__,
                    }
                )
            ]

    def split_to_docs(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ".", "?", "!", " "]
        )
        split_docs = []
        for doc in documents:
            chunks = splitter.split_text(self.clean_document_text(doc).page_content)
            for i, chunk in enumerate(chunks):
                split_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            "chunk_index": i,
                            "chunk_count": len(chunks)
                            # "full_context": doc.page_content
                        }
                    )
                )
        return split_docs

    def clean_document_text(self, doc: Document) -> Document:
        text = doc.page_content
        text = re.sub(r"[^\x00-\x7Fа-яА-ЯёЁ0-9.,!?;:'\"()\[\]{}«»\-–—/% ]+", " ", text)
        text = text.replace("\r", " ").replace("\t", " ")
        text = re.sub(r"\n+", "\n", text)
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        text = re.sub(r" {2,}", " ", text)
        text = text.strip()
        return Document(page_content=text, metadata=doc.metadata)

    # ------------------------ ТЕКСТ ------------------------
    def _extract_text_blocks(self, path) -> List[Document]:
        documents = []
        doc = fitz.open(path)
        pages = [page.get_text("text").strip() for page in doc]
        doc.close()
        for i in range(len(pages)):
            current_page = pages[i]
            next_page = pages[i + 1] if i + 1 < len(pages) else ""
            overlap_text = current_page
            if next_page:
                overlap_text += next_page[:300]

            if overlap_text.strip():
                documents.append(
                    Document(
                        page_content=overlap_text.strip(),
                        metadata={
                            "page": i + 1,
                            "source": str(path)
                        }
                    )
                )
        return documents

    # ------------------------ OCR ------------------------
    def _extract_ocr_blocks(self, path) -> List[Document]:
        doc = fitz.open(path)
        ocr_docs = []

        for page_num, page in enumerate(doc, start=1):
            for img_info in page.get_images(full=True):

                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(BytesIO(image_bytes))
                if image.width < 100 or image.height < 50:
                    continue
                try:
                    ocr_text = pytesseract.image_to_string(image, lang=self.config.ocr_lang).strip()
                except Exception as e:
                    print(f"Ошибка OCR на странице {page_num}: {e}")
                    ocr_text = ""

                if not ocr_text:
                    continue

                ocr_docs.append(
                    Document(
                        page_content=ocr_text,
                        metadata={
                            "page": page_num,
                            "type": "image_text",
                            "source": str(path)
                        }
                    )
                )

        doc.close()
        return ocr_docs

    # ------------------------ ТАБЛИЦЫ ------------------------
    def _extract_table_blocks(self, path) -> List[Document]:
        table_docs = []
        try:
            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables()
                    for t in tables:
                        # Преобразуем таблицу в читаемый текст
                        table_text = "\n".join([" | ".join(row) for row in t if any(row)])
                        if not table_text.strip():
                            continue

                        table_docs.append(
                            Document(
                                page_content=table_text.strip(),
                                metadata={
                                    "page": page_num,
                                    "type": "table",
                                    "source": str(path)
                                }
                            )
                        )
        except Exception as e:
            print(f"Ошибка при обработке таблиц: {e}")

        return table_docs
