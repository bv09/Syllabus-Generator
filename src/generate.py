import json, re
from pydantic import BaseModel, ValidationError
from llm import chat
from schema import (Syllabus, CourseInfo, Bi, CLO, AssessmentComponent,
                    BlueprintCell, Chapter, Reference)


class Generated(BaseModel):
    """Phần LLM sinh. Phần 'info' do người dùng nhập."""
    description: Bi
    clos: list[CLO]
    assessments: list[AssessmentComponent]
    blueprint: list[BlueprintCell]
    chapters: list[Chapter]
    required_readings: list[Reference]
    recommended_readings: list[Reference]


SYSTEM = """Bạn là giảng viên đại học có kinh nghiệm viết đề cương chi tiết học phần.
Chỉ trả về MỘT đối tượng JSON hợp lệ theo schema. Không giải thích, không markdown.
Mọi trường kiểu {vi, en} phải có cả tiếng Việt và tiếng Anh. Nội dung là văn bản thuần."""


def clean(text: str) -> str:
    return re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()


def build_prompt(info: CourseInfo, plos: list[str], n_chapters: int, hint: str) -> str:
    schema = json.dumps(Generated.model_json_schema(), ensure_ascii=False)
    return f"""Viết đề cương cho học phần:
- Tên: {info.title.vi} ({info.title.en}), mã {info.code}, {info.credits} tín chỉ
- Gợi ý nội dung: {hint}

Quy tắc:
- Đúng {n_chapters} chương, đánh số từ 1.
- 4-6 CLO, mã CLO1, CLO2...; linked_plos CHỈ chọn trong: {plos}; level là I, R hoặc M.
- assessments: tổng weight đúng bằng 100.
- blueprint: PHẢI có một mục cho MỖI cặp (thành phần đánh giá × CLO);
  trường component phải trùng đúng assessments[].name.vi.
- Mỗi chương gắn CLO trong clo_codes (chỉ dùng mã CLO đã tạo).
- Tài liệu tham khảo: KHÔNG bịa. Nếu không chắc chắn, để danh sách rỗng.

Schema JSON:
{schema}"""


def generate_syllabus(info: CourseInfo, plos, n_chapters, hint, max_retry=3) -> Syllabus:
    prompt = build_prompt(info, plos, n_chapters, hint)
    error = None
    for attempt in range(1, max_retry + 1):
        p = prompt if error is None else (
            prompt + f"\n\nLần trước bị lỗi:\n{error}\nHãy sửa và chỉ trả JSON.")
        raw = chat(SYSTEM, p)
        try:
            gen = Generated.model_validate_json(clean(raw))
            print(f"OK ở lần thử {attempt}")
            return Syllabus(info=info, **gen.model_dump())
        except ValidationError as e:
            error = str(e)[:1500]
            print(f"Lần {attempt} lỗi")
    raise RuntimeError("Hết số lần retry")