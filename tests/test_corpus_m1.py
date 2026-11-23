"""Tests cho corpus theo môn (M1) + prompts.py (M3)."""
import os
from pathlib import Path

import pytest
import src.rag
from src.rag import Chunk, Retriever, load_corpus


def _make_corpus(tmp_path, structure: dict):
    """Tạo corpus giả. structure: {path_relative: content}."""
    for rel, content in structure.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return tmp_path


# --- M1: kind từ tên thư mục con ---

def test_dir_kind_ref(tmp_path):
    """corpus/XX101/ref/books.txt → kind=reference."""
    _make_corpus(tmp_path, {
        "XX101/ref/books.txt": "Cormen. Algorithms. MIT Press.\nSedgewick. Algorithms. AW."
    })
    chunks = load_corpus(tmp_path)
    assert all(c.kind == src.rag.REFERENCE for c in chunks)
    assert len(chunks) == 2


def test_dir_kind_books(tmp_path):
    """corpus/XX101/books/notes.txt → kind=general."""
    _make_corpus(tmp_path, {
        "XX101/books/notes.txt": "Nội dung chương 1 về cấu trúc dữ liệu cơ bản bla bla"
    })
    chunks = load_corpus(tmp_path)
    assert all(c.kind == src.rag.GENERAL for c in chunks)


def test_dir_kind_plo(tmp_path):
    _make_corpus(tmp_path, {
        "_shared/plo/plo_2024.txt": "PLO 01: Có khả năng phân tích"
    })
    chunks = load_corpus(tmp_path)
    assert all(c.kind == "plo" for c in chunks)


def test_dir_kind_toc(tmp_path):
    _make_corpus(tmp_path, {"XX101/toc/toc_data.txt": "Mục lục giáo trình"})
    chunks = load_corpus(tmp_path)
    assert all(c.kind == "toc" for c in chunks)


def test_dir_kind_reg(tmp_path):
    _make_corpus(tmp_path, {"_shared/reg/qc_2024.txt": "Quy chế đào tạo đại học"})
    chunks = load_corpus(tmp_path)
    assert all(c.kind == "regulation" for c in chunks)


def test_prefix_still_works(tmp_path):
    """Tiền tố tên file vẫn hoạt động khi không có thư mục con."""
    _make_corpus(tmp_path, {
        "ref_dsa.txt": "Cormen. Algorithms.",
        "plo_2024.txt": "PLO 01",
    })
    chunks = load_corpus(tmp_path)
    refs = [c for c in chunks if c.kind == src.rag.REFERENCE]
    plos = [c for c in chunks if c.kind == "plo"]
    assert len(refs) == 1
    assert len(plos) >= 1


# --- M1: course isolation ---

def test_course_isolation(tmp_path):
    """Tài liệu môn A không lọt vào prompt môn B."""
    _make_corpus(tmp_path, {
        "IT3011/books/dsa.txt": "Cấu trúc dữ liệu và giải thuật cơ bản " * 20,
        "IT3080/books/network.txt": "Mạng máy tính giao thức TCP UDP " * 20,
    })
    chunks = load_corpus(tmp_path)
    r = Retriever(chunks)
    # Tìm kiếm loại trừ IT3011 → không có chunk của IT3011
    hits = r.search("cấu trúc dữ liệu", k=5, exclude_course="IT3011")
    for h in hits:
        assert h.chunk.course != "IT3011", f"Chunk {h.chunk.id} của IT3011 lọt vào kết quả"


def test_course_detected_from_path(tmp_path):
    """Chunk trong corpus/<MÃ>/... có course = MÃ."""
    _make_corpus(tmp_path, {
        "IT3011/books/dsa.txt": "Cấu trúc dữ liệu",
        "_shared/plo/plo.txt": "PLO chung",
    })
    chunks = load_corpus(tmp_path)
    course_chunks = [c for c in chunks if c.source == "dsa.txt"]
    shared_chunks = [c for c in chunks if c.source == "plo.txt"]
    assert all(c.course == "IT3011" for c in course_chunks)
    assert all(c.course is None for c in shared_chunks)


# --- M3: prompts.py ---

def test_prompts_system():
    from src.prompts import SYSTEM, verb_table, method_ceiling_table
    assert "JSON" in SYSTEM
    assert "Nhớ" in verb_table()
    assert "trắc nghiệm" in method_ceiling_table()
