"""Quy tắc Bloom (bản sửa đổi Anderson & Krathwohl, 2001) dạng code.

Hai chiều của A&K:
  - Quá trình nhận thức (Cog): Remember < Understand < Apply < Analyze < Evaluate < Create
  - Tri thức (KnowledgeType): factual, conceptual, procedural, metacognitive

LƯU Ý TRUNG THỰC:
  * A&K KHÔNG đưa ra danh sách động từ tiếng Việt. Bảng dưới đây là bản dịch/diễn giải
    của người viết code, cần giảng viên/bộ môn rà lại (sửa trực tiếp VERBS_VI / VERBS_EN).
  * Động từ -> mức là heuristic: cùng một động từ ("đánh giá", "tối ưu") có thể ở mức khác
    nhau tuỳ đối tượng. Kiểm tra động từ là điều kiện CẦN, không phải điều kiện ĐỦ.
  * METHOD_RULES (trần nhận thức của phương pháp đánh giá) là giả định của người viết code,
    không phải kết quả từ A&K.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import IntEnum


class Cog(IntEnum):
    REMEMBER = 1
    UNDERSTAND = 2
    APPLY = 3
    ANALYZE = 4
    EVALUATE = 5
    CREATE = 6

    @property
    def vi(self) -> str:
        return _LABEL_VI[self]


_LABEL_VI = {
    Cog.REMEMBER: "Nhớ", Cog.UNDERSTAND: "Hiểu", Cog.APPLY: "Vận dụng",
    Cog.ANALYZE: "Phân tích", Cog.EVALUATE: "Đánh giá", Cog.CREATE: "Sáng tạo",
}

KNOWLEDGE_TYPES = ("factual", "conceptual", "procedural", "metacognitive")

# --- Từ điển động từ -------------------------------------------------------
VERBS_VI: dict[Cog, list[str]] = {
    Cog.REMEMBER: ["nhớ", "nhắc lại", "liệt kê", "nhận diện", "nhận biết", "nhận dạng",
                   "xác định", "nêu", "gọi tên", "định nghĩa", "kể tên", "trích dẫn"],
    Cog.UNDERSTAND: ["hiểu", "giải thích", "phân loại", "tóm tắt", "diễn giải", "minh họa",
                     "minh hoạ", "so sánh", "suy luận", "mô tả", "trình bày", "thảo luận",
                     "nắm được"],
    Cog.APPLY: ["áp dụng", "vận dụng", "cài đặt", "thực hiện", "sử dụng", "triển khai",
                "lập trình", "giải quyết", "thực thi"],
    Cog.ANALYZE: ["phân tích", "phân biệt", "đối chiếu", "phân rã", "phân tách",
                  "tìm ra nguyên nhân", "gỡ lỗi"],
    Cog.EVALUATE: ["đánh giá", "phê bình", "nhận xét", "lựa chọn", "biện luận", "thẩm định",
                   "kiểm định", "lập luận", "bảo vệ"],
    Cog.CREATE: ["thiết kế", "đề xuất", "xây dựng", "phát triển", "sáng tạo", "tạo ra",
                 "lập kế hoạch", "tối ưu hoá", "tối ưu hóa", "đề ra"],
}

# Tránh danh từ trùng dạng động từ (list, state, name, plan, test, use...) vì
# ví dụ "linked lists" sẽ bị nhận nhầm.
VERBS_EN: dict[Cog, list[str]] = {
    Cog.REMEMBER: [r"defin\w+", r"recall\w*", r"identif(?:y|ies|ying)", r"recogni[sz]\w*",
                   r"enumerat\w+", r"memori[sz]\w+"],
    Cog.UNDERSTAND: [r"understand\w*", r"explain\w*", r"describ\w+", r"classif(?:y|ies|ying)",
                     r"summari[sz]\w+", r"interpret\w*", r"compar(?:e|es|ing)", r"illustrat\w+",
                     r"discuss\w*", r"infer\w*"],
    Cog.APPLY: [r"appl(?:y|ies|ying)", r"implement\w*", r"execut\w+", r"solv(?:e|es|ing)",
                r"demonstrat\w+", r"program(?:ming)?", r"comput(?:e|es|ing)"],
    Cog.ANALYZE: [r"analy[sz]\w+", r"differentiat\w+", r"distinguish\w*", r"examin\w+",
                  r"decompos\w+", r"debug\w*"],
    Cog.EVALUATE: [r"evaluat\w+", r"assess\w*", r"critiqu\w+", r"judg\w+", r"justif(?:y|ies|ying)",
                   r"argu(?:e|es|ing)", r"appraise\w*"],
    Cog.CREATE: [r"design\w*", r"propos(?:e|es|ing)", r"creat(?:e|es|ing)", r"construct\w*",
                 r"develop\w*", r"formulat\w+", r"optimi[sz]\w+", r"devis\w+", r"invent\w*"],
}

_LEAD = {
    "vi": re.compile(r"^(?:sinh viên\s+)?(?:có khả năng|có thể|được)\s+"),
    "en": re.compile(r"^(?:students?\s+(?:will\s+)?(?:be able to\s+)?|ability to\s+|"
                     r"be able to\s+|capable of\s+)"),
}
_WINDOW = {"vi": 12, "en": 9}  # chỉ quét đầu câu: động từ chính thường ở đây, giảm nhận nhầm danh từ


def _compile() -> dict[str, dict[Cog, re.Pattern]]:
    out: dict[str, dict[Cog, re.Pattern]] = {"vi": {}, "en": {}}
    for lvl, words in VERBS_VI.items():
        alts = sorted((re.escape(w) for w in words), key=len, reverse=True)
        out["vi"][lvl] = re.compile(r"(?<!\w)(?:%s)(?!\w)" % "|".join(alts))
    for lvl, pats in VERBS_EN.items():
        out["en"][lvl] = re.compile(r"(?<!\w)(?:%s)(?!\w)" % "|".join(pats))
    return out


_RX = _compile()


def norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).lower()


@dataclass(frozen=True)
class BloomResult:
    level: Cog                      # mức CAO NHẤT trong câu (quy ước bảo thủ)
    lowest: Cog
    verbs: tuple[tuple[str, Cog], ...]

    @property
    def spread(self) -> int:
        return int(self.level) - int(self.lowest)


def classify(text: str, lang: str) -> BloomResult | None:
    """Tìm động từ Bloom ở đầu câu. lang: 'vi' | 'en'."""
    t = _LEAD[lang].sub("", norm(text).strip())
    window = " ".join(t.split()[: _WINDOW[lang]])
    hits: list[tuple[int, str, Cog]] = []
    for lvl, rx in _RX[lang].items():
        for m in rx.finditer(window):
            hits.append((m.start(), m.group(0), lvl))
    if not hits:
        return None
    hits.sort()
    lv = [h[2] for h in hits]
    return BloomResult(level=max(lv), lowest=min(lv),
                       verbs=tuple((h[1], h[2]) for h in hits))


def analyze_clo(clo) -> dict:
    """Trả {'vi': BloomResult|None, 'en': BloomResult|None, 'level': Cog|None}."""
    vi = classify(clo.description.vi, "vi")
    en = classify(clo.description.en, "en")
    level = vi.level if vi else (en.level if en else None)
    return {"vi": vi, "en": en, "level": level}


def verb_table_vi() -> str:
    return "\n".join(f"- {lvl.vi} ({lvl.name.title()}): {', '.join(w)}"
                     for lvl, w in VERBS_VI.items())


# --- Trần nhận thức của phương pháp đánh giá (GIẢ ĐỊNH, chỉnh được) --------
METHOD_RULES: list[tuple[tuple[str, ...], Cog]] = [
    (("trắc nghiệm", "quiz", "mcq", "multiple choice"), Cog.APPLY),
    (("vấn đáp", "oral", "thuyết trình", "presentation"), Cog.EVALUATE),
    (("đồ án", "bài tập lớn", "project", "thực hành", "lab", "lập trình", "programming",
      "portfolio"), Cog.CREATE),
    (("kiểm tra viết", "written", "tự luận", "essay"), Cog.CREATE),
]


def method_ceiling(name: str) -> Cog | None:
    n = norm(name)
    for keys, ceil in METHOD_RULES:
        if any(k in n for k in keys):
            return ceil
    return None
