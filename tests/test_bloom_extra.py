"""Tests cho bloom_extra.py (M2)."""
from src.bloom_extra import warn_vague, VAGUE


def test_vague_set_contents():
    assert VAGUE == {"biết", "nắm được", "appreciate", "know"}


def test_warn_vague_detects():
    warns = warn_vague("Biết các khái niệm cơ bản")
    assert len(warns) == 1
    assert "biết" in warns[0].lower()


def test_warn_vague_en():
    warns = warn_vague("Appreciate the importance of testing")
    assert len(warns) == 1
    assert "appreciate" in warns[0].lower()


def test_warn_vague_know():
    warns = warn_vague("Know the basic concepts")
    assert len(warns) == 1
    assert "know" in warns[0].lower()


def test_warn_vague_nam_duoc():
    warns = warn_vague("Nắm được lý thuyết đồ thị")
    assert len(warns) == 1
    assert "nắm được" in warns[0].lower()


def test_no_warn_for_specific_verb():
    assert warn_vague("Phân tích độ phức tạp") == []
    assert warn_vague("Explain data structure concepts") == []
    assert warn_vague("Áp dụng thuật toán") == []


def test_no_warn_vague_with_lead():
    # "Có khả năng" prefix → vẫn phát hiện
    warns = warn_vague("Có khả năng biết các khái niệm")
    assert len(warns) == 1
