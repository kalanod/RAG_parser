"""Example script showing how to parse documents using the pipeline."""
from __future__ import annotations

from pathlib import Path

from core.pipeline import parse_document
from utils.formatting import to_markdown


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse a document into segments")
    parser.add_argument("path", type=Path, help="Path to the document to parse")
    parser.add_argument("--no-chunk", action="store_true", help="Disable chunking")
    args = parser.parse_args()

    segments = parse_document(args.path, chunk=not args.no_chunk)
    print(to_markdown(segments))

