# Hybrid PDF Parser with OCR Preservation of Document Structure

This document outlines the architecture for extending the current `PDFParser`
so that it can automatically fall back to OCR for scanned PDFs while preserving
the structural segmentation (headings, paragraphs, lists, tables) that the
existing pipeline relies on.

## Overview

We introduce a `HybridPDFParser` that wraps the existing text-based
`PDFParser`. The new parser detects whether a PDF contains an extractable text
layer. If a text layer is found, it delegates to the standard `PDFParser`
behaviour. When the PDF is image-only, it applies OCR and reconstructs block
metadata before handing the text back to the inherited segmentation pipeline.

```
parse_document
    └── HybridPDFParser.parse(path)
          ├── _detect_text_layer(path)  ──► text? ──► super().parse()
          └── _parse_via_ocr(path)
                ├── _render_pages_to_images()
                ├── _run_ocr_on_pages()
                └── _reconstruct_blocks_from_ocr()
```

## Text-Layer Detection

* Open the document with PyMuPDF (already used in `PDFParser`).
* For each page, collect both the textual content and statistics such as the
  ratio of printable characters to page area, share of whitespace-only lines,
  and the average length of extracted lines.
* Declare the PDF text-rich when at least one of the following holds:
  * Total printable characters ≥ configurable threshold (e.g. 500).
  * Average density ≥ threshold (characters per cm²) on more than 60% of
    pages.
* If text-rich, call `super().parse(path, **kwargs)` and return.

These thresholds should be configurable (through parser kwargs or settings) so
operations can tune sensitivity without code changes.

## OCR Fallback Pipeline

When `_detect_text_layer` reports that the text layer is missing or too short,
`HybridPDFParser` switches to `_parse_via_ocr(path)`.

1. **Page Rendering**
   * Render each page to an image using PyMuPDF (`page.get_pixmap`) or, as a
     fallback, `pdf2image`.
   * Normalise DPI (e.g. 300) and colour mode to improve OCR accuracy.

2. **OCR Extraction**
   * Reuse the existing `ImageOCRParser` backends (PaddleOCR → Tesseract).
   * Collect not only recognised strings but also their bounding boxes and
     confidence scores. Both PaddleOCR and Tesseract expose this data.
   * Record per-page metadata: page number, rotation, and DPI.

3. **Block Reconstruction**
   * Convert the OCR line data into logical blocks compatible with the current
     `_Block` model used by `PDFParser`:
     * **Tables** – cluster boxes forming grid-like structures (uniform rows
       and columns) or detect characters separated by multiple spaces; keep the
       original row/column order.
     * **Lists** – detect bullet/number markers within bounding boxes aligned on
       the left margin; merge with following text boxes sharing baseline.
     * **Headings** – identify lines with high confidence, larger bounding boxes
       (font-size approximation via box height), or fully uppercase strings.
     * **Paragraphs** – merge consecutive lines with similar x-coordinates and
       spacing.
   * Feed these reconstructed `_Block` objects through the inherited helper
     methods: `_merge_blocks` and `_enforce_max_size`.

4. **Segment Emission**
   * Yield `Segment` objects using `Segment.from_text`, attaching metadata such
     as:

     ```python
     {
         "type": block.type,
         "page": page_number,
         "source_type": "ocr",
         "bbox": [x0, y0, x1, y1],  # optional for downstream QA/visualisation
         "confidence": min_line_confidence,
     }
     ```

   * This keeps downstream consumers aware that the content originated from OCR
     while preserving structural tags relied upon by the chunking pipeline.

## Error Handling and Configuration

* If OCR backends are unavailable, return a diagnostic `Segment` with metadata
  `{"error": "ocr_unavailable"}` to match current error handling in
  `PDFParser`.
* Expose feature flags/kwargs:
  * `enable_ocr: bool = True`
  * `min_text_characters: int`
  * `ocr_dpi: int`
  * `return_bboxes: bool`
* Log which branch (text vs OCR) was taken, total processing time, number of
  blocks, and aggregated confidence to aid monitoring.

## Testing Strategy

* **Unit Tests**
  * PDF with rich text → ensure `_detect_text_layer` routes to the standard
    path and metadata `source_type` is absent.
  * Image-only PDF → ensure OCR branch runs and emits segments with
    `source_type="ocr"` and block types preserved.
  * Mixed PDFs (some pages text, some image) → validate per-page detection and
    combined output.
  * Missing OCR dependencies → expect diagnostic segment.

* **Integration / Golden Tests**
  * Prepare sample PDFs with headings, bullet lists, and tables rendered as
    images, then assert that output block types match ground truth annotations.

This approach adds OCR support without bypassing the existing segmentation
logic, ensuring that downstream consumers continue to receive enriched segment
metadata even for scanned PDFs.
