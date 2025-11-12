from dataclasses import dataclass
from pathlib import Path
from typing import List
import pandas as pd
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

from parsers.abstract_document_parser import AbstractDocumentParser

LOGGER = logging.getLogger(__name__)


@dataclass
class XLSXParserConfig:
    chunk_size: int = 1500  # по токенам, не символам
    chunk_overlap: int = 200
    min_nonempty_cells: int = 2


class XLSXParser(AbstractDocumentParser):
    supported_extensions = (".xlsx", ".xls")
    config = XLSXParserConfig()

    def split_to_docs(self, documents):
        raise NotImplementedError

    def parse(self, path: Path) -> List[Document]:
        try:
            xls = pd.ExcelFile(path)
        except Exception as e:
            LOGGER.exception("Failed to open XLSX file: %s", e)
            return [
                Document(
                    page_content=f"[ERROR] Failed to read XLSX: {e}",
                    metadata={"source": str(path), "error": True}
                )
            ]

        all_docs: List[Document] = []
        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
                df = df.dropna(how="all").dropna(axis=1, how="all")
                if df.empty:
                    continue

                markdown = self._to_markdown(df, sheet_name)
                sheet_docs = self._split_to_docs(markdown, path, sheet_name)
                all_docs.extend(sheet_docs)
            except Exception as e:
                LOGGER.warning("Failed to parse sheet '%s' in %s: %s", sheet_name, path, e)
                continue

        return all_docs

    def _to_markdown(self, df: pd.DataFrame, sheet_name: str) -> str:
        df = df.dropna(how="all")
        if df.empty:
            return ""

        headers = [str(h).strip() for h in df.columns]
        md_lines = [
            f"### {sheet_name}",
            "",
            " | ".join(headers),
            " | ".join(["---"] * len(headers))
        ]

        for _, row in df.iterrows():
            cells = [str(x).strip().replace("\n", " ") if pd.notna(x) else "" for x in row]
            if sum(bool(c) for c in cells) < self.config.min_nonempty_cells:
                continue
            md_lines.append(" | ".join(cells))

        return "\n".join(md_lines)

    def _split_to_docs(self, markdown: str, path: Path, sheet_name: str) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )
        chunks = splitter.split_text(markdown)

        docs = []
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk.strip(),
                    metadata={
                        "source": str(path),
                        "sheet": sheet_name,
                        "chunk_index": i,
                        "chunk_count": len(chunks),
                        "type": "table",
                    }
                )
            )
        return docs
