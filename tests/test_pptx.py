from pathlib import Path

from parsers.pptx_parser import PPTXParser


def test_pptx_parser_returns_list(tmp_path: Path):
    path = tmp_path / "sample.pptx"
    path.write_bytes(b"")
    parser = PPTXParser()
    segments = parser.parse(path)
    assert isinstance(segments, list)

