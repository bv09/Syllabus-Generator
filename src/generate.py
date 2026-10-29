import json, re
from pathlib import Path
from pydantic import ValidationError
from llm import chat
from schema import Syllabus

SYSTEM = """Bạn là giảng viên đại học có kinh nghiệm viết đề cương môn học.
Chỉ trả về MỘT đối tượng JSON hợp lệ theo schema được cung cấp.
Không thêm giải thích, không dùng markdown. Viết nội dung bằng tiếng Việt."""

def clean(text: str) -> str:
    return re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()

def generate_syllabus(name, code, credits, weeks, desc, max_retry=3) -> Syllabus:
    schema = json.dumps(Syllabus.model_json_schema(), ensure_ascii=False)
    user = f"""Tạo đề cương môn học:
                - Tên môn: {name}
                - Mã môn: {code}
                - Số tín chỉ: {credits}
                - Số tuần: {weeks} (weekly_plan phải có đủ {weeks} tuần)
                - Mô tả: {desc}
                
                
                Schema JSON:
                {schema}"""
    error = None
    for attempt in range(1, max_retry + 1):
        prompt = user if error is None else (
            user + f"\n\nLần trước bị lỗi:\n{error}\nHãy sửa và chỉ trả JSON.")
        raw = chat(SYSTEM, prompt)
        try:
            result = Syllabus.model_validate_json(clean(raw))
            print(f"OK ở lần thử {attempt}")
            return result
        except ValidationError as e:
            error = str(e)[:1500]
            print(f"Lần {attempt} lỗi")
    raise RuntimeError("Hết số lần retry")

if __name__ == "__main__":
    s = generate_syllabus("Cấu trúc dữ liệu và giải thuật", 
                          "IT3011", 
                          3, 
                          15,
                          "Các cấu trúc dữ liệu cơ bản, thuật toán sắp xếp, tìm kiếm, đồ thị.")
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/ctdl.json").write_text(s.model_dump_json(indent=2), encoding="utf-8")