"""Tests cho pdf_ingest.py (M0)."""
import os
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_pdf(tmp_path):
    """Tạo 1 file PDF tối giản bằng pypdf."""
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    p = tmp_path / "ch1_intro.pdf"
    with open(p, "wb") as f:
        writer.write(f)
    return p


@pytest.fixture()
def tmp_pdf_dir(tmp_path):
    """Tạo thư mục chứa 2 file PDF."""
    from pypdf import PdfWriter
    d = tmp_path / "pdfs"
    d.mkdir()
    for name in ("01_overview.pdf", "02_sorting.pdf"):
        w = PdfWriter()
        w.add_blank_page(72, 72)
        with open(d / name, "wb") as f:
            w.write(f)
    return d


def test_analyze_pdf_basic(tmp_pdf):
    from src.pdf_ingest import analyze_pdf
    report, chapters = analyze_pdf(tmp_pdf)
    assert report.num_pages == 1
    assert report.filename == "ch1_intro.pdf"
    assert not report.errors


def test_ingest_copies_to_corpus(tmp_pdf, tmp_path):
    from src.pdf_ingest import ingest_pdfs
    corpus = tmp_path / "corpus_out"
    reports, chapters = ingest_pdfs([tmp_pdf], "XX101", corpus)
    assert len(reports) == 1
    dest = corpus / "XX101" / "books" / "ch1_intro.pdf"
    assert dest.exists()


def test_no_bookmark_uses_filename(tmp_pdf_dir, tmp_path):
    from src.pdf_ingest import ingest_pdfs
    pdfs = sorted(tmp_pdf_dir.glob("*.pdf"))
    corpus = tmp_path / "corpus_out"
    reports, chapters = ingest_pdfs(pdfs, "XX101", corpus)
    assert len(chapters) == 2
    # Tên file bỏ prefix số → "overview", "sorting"
    titles = [c.title for c in chapters]
    assert "overview" in titles
    assert "sorting" in titles


def test_resolve_pdf_paths(tmp_pdf_dir, tmp_pdf):
    from src.pdf_ingest import resolve_pdf_paths
    # Thư mục
    result = resolve_pdf_paths([str(tmp_pdf_dir)])
    assert len(result) == 2
    # File đơn
    result = resolve_pdf_paths([str(tmp_pdf)])
    assert len(result) == 1
    # Không tồn tại
    result = resolve_pdf_paths(["nonexistent.pdf"])
    assert len(result) == 0
