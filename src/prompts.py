"""Prompt 4 giai đoạn gom vào một file.

Nội suy {verb_table} và {schema} từ code — không tạo file .md riêng.
Hành vi KHÔNG đổi so với prompt inline trong generate.py.
"""
from __future__ import annotations

import json

import bloom
from schema import CLO, Bi


def _schema(cls) -> str:
    return json.dumps(cls.model_json_schema(), ensure_ascii=False)


def verb_table() -> str:
    return bloom.verb_table_vi()


def method_ceiling_table() -> str:
    return "\n".join(f"- {', '.join(k[:2])}: đo được tới mức {v.vi}" for k, v in bloom.METHOD_RULES)


SYSTEM = """Bạn là giảng viên đại học có kinh nghiệm viết đề cương chi tiết học phần.
Chỉ trả về MỘT đối tượng JSON hợp lệ theo schema. Không giải thích, không markdown.
Mọi trường kiểu {vi, en} phải có cả tiếng Việt và tiếng Anh. Nội dung là văn bản thuần.
Chỉ dùng thông tin trong NGỮ CẢNH khi nêu tên sách, tên chương, quy định; không bịa."""


def prompt_clos(head: str, ctx: str, clo_min: int, clo_max: int, plos: list[str], schema_json: str) -> str:
    n_clos = clo_min if clo_min == clo_max else f"{clo_min}-{clo_max}"
    return f"""{head}

NGỮ CẢNH (trích từ kho tài liệu của bộ môn):
{ctx}

Viết MÔ TẢ học phần và các CHUẨN ĐẦU RA (CLO). Quy tắc:
- BẮT BUỘC viết ĐÚNG {n_clos} CLO (CLO1, CLO2, ...), TUYỆT ĐỐI KHÔNG sinh thừa hay thiếu.
- Sắp xếp các CLO theo tiến trình nhận thức Bloom tăng dần từ mức thấp đến mức cao.
- Mỗi CLO tương ứng với MỘT bậc nhận thức khác nhau trong bảng Bloom bên dưới. TUYỆT ĐỐI KHÔNG để nhiều CLO cùng một mức nhận thức và KHÔNG lặp lại cùng một động từ hành động (ví dụ không để nhiều CLO cùng là 'cài đặt' / Vận dụng).
- Mỗi CLO bắt đầu bằng MỘT động từ hành động đo được lấy từ bảng Bloom sửa đổi (hai bản vi và en phải cùng mức nhận thức):
{verb_table()}
- Đảm bảo CLO chỉ đề cập đến nội dung THỰC SỰ được dạy trong học phần (constructive alignment).
- linked_plos: danh sách các PLO mà CLO đóng góp (chọn 1 hoặc nhiều PLO trong: {plos}); level là I, R hoặc M (mức đóng góp cho PLO, KHÔNG phải mức Bloom).
- knowledge_type: factual | conceptual | procedural | metacognitive.
- KHÔNG điền bloom_level (hệ thống tự gán bằng bloom.py).

Schema JSON:
{schema_json}"""


def prompt_chapters(head: str, clo_txt: str, ctx: str, n: int, schema_json: str) -> str:
    return f"""{head}

CLO đã chốt:
{clo_txt}

NGỮ CẢNH:
{ctx}

Viết ĐÚNG {n} chương, đánh số từ 1, mỗi chương có title, topics (4-6 mục) và activities (song ngữ).
Quy tắc:
- clo_codes của chương = các CLO mà chương đó THỰC SỰ phát triển (thường 1-3). Mỗi CLO phải được
  ít nhất một chương phát triển. KHÔNG gắn mọi chương với mọi CLO.
- sources = id các đoạn NGỮ CẢNH bạn dựa vào (ví dụ "abc:3"); để [] nếu dùng kiến thức chung.
- Thứ tự chương theo logic sư phạm (khái niệm nền trước, ứng dụng sau).

Schema JSON:
{schema_json}"""


def prompt_assessment(head: str, clo_txt: str, schema_json: str, fixed: str = "") -> str:
    rule_timing = f"""- BẮT BUỘC SỬ DỤNG ĐÚNG các thành phần đánh giá, trọng số (weight) và thời điểm (timing) sau do bộ môn quy định:
{fixed}
Giữ nguyên 100% name, weight, timing; bạn chỉ thiết kế phần coverage (chọn CLO nào đo bằng phương pháp tương ứng).""" if fixed else """- Tổng weight đúng bằng 100.
- timing phải phân bổ hợp lý theo tiến trình học tập (TUYỆT ĐỐI không để tất cả thành phần cùng tuần 17-19):
  + Kiểm tra thường xuyên / Quiz / Bài tập: tuần 5-10 (hoặc rải rác giữa kỳ)
  + Đồ án / Bài tập lớn / Project: tuần 15-17
  + Kiểm tra viết cuối kỳ: tuần 17-19"""

    return f"""{head}

CLO (kèm mức Bloom đã gán):
{clo_txt}

Thiết kế các THÀNH PHẦN ĐÁNH GIÁ. Quy tắc:
{rule_timing}
- coverage: với mỗi thành phần liệt kê các CLO mà nó đo và phương pháp (song ngữ, ví dụ {{"vi":"Kiểm tra viết","en":"Written test"}}).
  + Đồ án/Bài tập lớn đo được hầu hết các CLO thực hành và sáng tạo.
  + Kiểm tra viết cuối kỳ đo mức Hiểu, Vận dụng, Phân tích (không dùng để đo mức Sáng tạo).
  + Quiz/Trắc nghiệm đo mức Hiểu và Vận dụng cơ bản.
  Mỗi CLO phải được ít nhất một thành phần đo.
- Phương pháp phải đủ "trần" cho mức Bloom của CLO (CLO mức Sáng tạo/Đánh giá không chỉ đo bằng trắc
  nghiệm). Trần của từng phương pháp:
{method_ceiling_table()}

Schema JSON:
{schema_json}"""


def prompt_refs(head: str, cand_list: str, schema_json: str) -> str:
    return f"""{head}

DANH SÁCH TÀI LIỆU ỨNG VIÊN (từ kho của bộ môn):
{cand_list}

Chọn id các tài liệu phù hợp: required_ids (1-2 tài liệu chính), recommended_ids (còn lại, nếu phù hợp).
Chỉ dùng id có trong danh sách; một id không xuất hiện ở cả hai danh sách; để [] nếu không tài liệu nào phù hợp.

Schema JSON:
{schema_json}"""
