"""Mở rộng Bloom (KHÔNG sửa bloom.py).

Chứa tập động từ MƠ HỒ — những từ KHÔNG bị chặn (vẫn hợp lệ), chỉ CẢNH BÁO.
Lý do: "hiểu" là động từ Bloom hợp lệ (mức 2 — Understand), nhưng các từ trong VAGUE
thường bị coi là quá chung chung trong thực tế viết CLO.
"""
from __future__ import annotations

import re
import unicodedata

# Những từ gây cảnh báo "mơ hồ" khi xuất hiện ở đầu CLO.
# KHÔNG phải lỗi — chỉ là khuyến nghị xem xét thay bằng động từ cụ thể hơn.
VAGUE: set[str] = {"biết", "nắm được", "appreciate", "know"}

_LEAD_VI = re.compile(r"^(?:sinh viên\s+)?(?:có khả năng|có thể|được)\s+")
_LEAD_EN = re.compile(r"^(?:students?\s+(?:will\s+)?(?:be able to\s+)?|ability to\s+|"
                       r"be able to\s+|capable of\s+)")


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).lower().strip()


def warn_vague(clo_text: str) -> list[str]:
    """Trả danh sách cảnh báo nếu CLO chứa từ mơ hồ ở đầu câu.

    >>> warn_vague("Biết các khái niệm cơ bản")
    ['Từ "biết" thường quá mơ hồ cho CLO — xem xét thay bằng động từ cụ thể hơn (ví dụ: nhận diện, liệt kê, giải thích).']
    >>> warn_vague("Appreciate the importance of testing")
    ['Từ "appreciate" thường quá mơ hồ cho CLO — xem xét thay bằng động từ cụ thể hơn (ví dụ: nhận diện, liệt kê, giải thích).']
    >>> warn_vague("Phân tích độ phức tạp")
    []
    """
    t = _norm(clo_text)
    # Bỏ dẫn ngữ
    t = _LEAD_VI.sub("", t)
    t = _LEAD_EN.sub("", t)
    words = t.split()[:6]  # chỉ quét đầu câu
    head = " ".join(words)

    warnings: list[str] = []
    for v in VAGUE:
        v_norm = _norm(v)
        # Khớp từ nguyên (word boundary)
        if re.search(rf"(?<!\w){re.escape(v_norm)}(?!\w)", head):
            warnings.append(
                f'Từ "{v_norm}" thường quá mơ hồ cho CLO — xem xét thay bằng '
                f'động từ cụ thể hơn (ví dụ: nhận diện, liệt kê, giải thích).')
    return warnings
