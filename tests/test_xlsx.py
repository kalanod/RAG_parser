from pathlib import Path

from parsers.xlsx_parser import XLSXParser


def test_xlsx_parser_returns_list(tmp_path: Path):
    path = tmp_path / "sample.csv"
    path.write_text("col1,col2\n1,2")
    parser = XLSXParser()
    segments = parser.parse(path)
    assert isinstance(segments, list)

