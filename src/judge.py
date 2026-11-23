"""LLM-as-judge cho các tiêu chí không kiểm tra bằng code được (L1-L3 trong rubric.py).

Điểm yếu đã biết của phương pháp này (đừng bỏ qua khi báo cáo kết quả):
  - Thiên vị tự ưu tiên: judge cùng họ model với generator có xu hướng chấm cao -> đặt
    LLM_JUDGE_MODEL khác LLM_MODEL.
  - Thiên vị độ dài/văn phong; điểm dao động giữa các lần chạy.
  - Chưa hiệu chuẩn: cần đối chiếu với điểm của ≥2 giảng viên trên một tập đề cương
    (tính độ đồng thuận, ví dụ Cohen's kappa có trọng số) trước khi tin con số.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from pydantic import BaseModel, ValidationError, field_validator

from rubric import CRIT, CRITERIA
from schema import Syllabus

LLM_IDS = [c.id for c in CRITERIA if c.kind == "llm"]

JUDGE_SYSTEM = """Bạn là hội đồng thẩm định đề cương học phần đại học, chấm nghiêm và có căn cứ.
Với MỖI tiêu chí: trước hết trích bằng chứng cụ thể từ đề cương (evidence), nêu vấn đề chính (issue),
RỒI MỚI cho điểm nguyên 1-4 theo anchors. Không thưởng cho độ dài hay văn phong. Chỉ trả một JSON."""


class Verdict(BaseModel):
    evidence: str
    issue: str
    score: int

    @field_validator("score")
    @classmethod
    def _rng(cls, v: int) -> int:
        if not 1 <= v <= 4:
            raise ValueError("score phải trong 1..4")
        return v


def build_prompt(s: Syllabus, ctx: str) -> str:
    rubric = "\n".join(
        f"{c.id} — {c.name}\n" + "\n".join(f"   {k}: {v}" for k, v in c.anchors.items())
        for c in CRITERIA if c.kind == "llm")
    body = {
        "title": s.info.title.model_dump(), "credits": s.info.credits,
        "workload": s.info.workload.model_dump(), "prerequisites": s.info.prerequisites,
        "description": s.description.vi,
        "clos": [{"code": c.code, "vi": c.description.vi, "bloom": c.bloom_level} for c in s.clos],
        "chapters": [{"n": ch.number, "title": ch.title.vi, "topics": [t.vi for t in ch.topics],
                      "clos": ch.clo_codes} for ch in s.chapters],
    }
    return f"""RUBRIC (anchors):
{rubric}

NGỮ CẢNH THAM CHIẾU (có thể trống; nếu trống, chấm theo kiến thức chuyên môn và ghi 'chưa kiểm chứng'):
{ctx or '(trống)'}

ĐỀ CƯƠNG:
{json.dumps(body, ensure_ascii=False)}

Trả JSON dạng {{"L1": {{"evidence": "...", "issue": "...", "score": 1-4}}, "L2": {{...}}, "L3": {{...}}}}"""


def judge(s: Syllabus, ctx: str, llm: Callable[[str, str], str], max_retry: int = 2) -> dict[str, dict]:
    prompt, err = build_prompt(s, ctx), ""
    for _ in range(max_retry + 1):
        raw = llm(JUDGE_SYSTEM, prompt + (f"\n\nLần trước lỗi: {err}\nChỉ trả JSON đúng định dạng." if err else ""))
        try:
            t = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
            data = json.loads(t[t.find("{"): t.rfind("}") + 1])
            out = {i: Verdict.model_validate(data[i]).model_dump() for i in LLM_IDS}
            return out
        except (ValueError, KeyError, ValidationError) as e:
            err = str(e)[:400]
    raise RuntimeError(f"Judge trả về định dạng sai sau {max_retry + 1} lần: {err}")
