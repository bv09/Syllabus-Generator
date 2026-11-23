"""Chạy pipeline bằng LLM giả: lần đầu CỐ TÌNH vi phạm luật, kiểm tra phản hồi được gửi ngược lại."""
import json
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
SRC = str(Path(__file__).resolve().parents[1] / "src")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import src.schema
sys.modules["schema"] = src.schema

import src.generate as generate
import src.rag as rag
from src.rag import Chunk, Retriever
from src.schema import CourseInfo, Workload, Bi

INFO = CourseInfo(title=Bi(vi="Cấu trúc dữ liệu", en="Data Structures"), code="XX101", credits=2,
                  workload=Workload(lecture_periods=30, exercise_periods=15, lab_periods=0, self_study_periods=60),
                  teaching_methods=[Bi(vi="Trực tiếp", en="Offline")],
                  relation_to_curriculum=Bi(vi="Bắt buộc", en="Mandatory"), prerequisites=[],
                  semester=2, lecturers=["A"])
PLOS = ["PLO 01", "PLO 02"]


def clo(i, vi, en, kt="conceptual"):
    return {"code": f"CLO{i}", "description": {"vi": vi, "en": en}, "linked_plos": ["PLO 02"],
            "level": "R", "knowledge_type": kt, "bloom_level": None}


GOOD_CLO = {"description": {"vi": "Mô tả", "en": "Desc"}, "clos": [
    clo(1, "Giải thích khái niệm cấu trúc dữ liệu", "Explain data structure concepts"),
    clo(2, "Cài đặt ngăn xếp và hàng đợi", "Implement stacks and queues", "procedural"),
    clo(3, "Phân tích độ phức tạp thuật toán", "Analyze algorithm complexity"),
    clo(4, "Đề xuất cấu trúc dữ liệu phù hợp bài toán", "Propose suitable data structures"),
]}
BAD_CLO = json.loads(json.dumps(GOOD_CLO))
BAD_CLO["clos"][0]["linked_plos"] = ["PLO 99"]           # sai PLO
BAD_CLO["clos"][1]["description"] = {"vi": "Cấu trúc ngăn xếp", "en": "Stack structure"}  # không có động từ ở cả hai ngôn ngữ

CH = {"chapters": [{"number": i, "title": {"vi": f"Chương {i}", "en": f"Chapter {i}"},
                    "topics": [{"vi": "mục", "en": "item"}], "clo_codes": [f"CLO{(i % 4) + 1}"],
                    "activities": {"vi": "Giảng bài", "en": "Lecture"}, "sources": []} for i in (1, 2, 3, 4)]}
M = lambda v, e: [{"vi": v, "en": e}]
AS_BAD = {"assessments": [  # tổng 90 và CLO4 (Create) chỉ đo bằng trắc nghiệm
    {"name": {"vi": "Cuối kỳ", "en": "Final"}, "weight": 60, "timing": {"vi": "tuần 17", "en": "week 17"},
     "coverage": [{"clo_code": c, "methods": M("Trắc nghiệm", "Quiz")} for c in ("CLO1", "CLO2", "CLO3", "CLO4")]},
    {"name": {"vi": "Quá trình", "en": "Process"}, "weight": 30, "timing": {"vi": "tuần 1-15", "en": "weeks 1-15"},
     "coverage": [{"clo_code": "CLO1", "methods": M("Trắc nghiệm", "Quiz")}]}]}
AS_OK = json.loads(json.dumps(AS_BAD))
AS_OK["assessments"][1]["weight"] = 40
for i in (2, 3):  # CLO3 (Analyze) và CLO4 (Create) cần phương pháp có trần cao hơn trắc nghiệm
    AS_OK["assessments"][0]["coverage"][i]["methods"] = M("Kiểm tra viết", "Written test")


class Fake:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def __call__(self, system, user):
        self.prompts.append(user)
        return json.dumps(self.replies.pop(0), ensure_ascii=False)


def test_full_pipeline_with_retry_feedback_and_provenance():
    r = Retriever([Chunk("ref:0", "Cormen. Introduction to Algorithms. MIT Press, 2022.", rag.REFERENCE, "ref"),
                   Chunk("toc:0", "Cấu trúc dữ liệu: ngăn xếp, hàng đợi, cây", "toc", "toc")])
    fake = Fake([BAD_CLO, GOOD_CLO, CH, AS_BAD, AS_OK, {"required_ids": ["ref:0"], "recommended_ids": []}])
    s = generate.generate_syllabus(INFO, PLOS, 4, "cấu trúc dữ liệu", retriever=r, llm=fake)
    # phản hồi cụ thể đã được gửi ngược lại ở lần retry
    assert "PLO 99" in fake.prompts[1] and "động từ Bloom" in fake.prompts[1]
    assert "Tổng tỉ trọng = 90" in fake.prompts[4]
    # bloom_level do code gán; blueprint dựng bằng code
    assert [c.bloom_level for c in s.clos] == ["understand", "apply", "analyze", "create"]
    assert len(s.blueprint) == 5
    assert s.required_readings[0].source_id == "ref:0"
    assert not [n for n in s.notes if "Corpus" in n]


def test_no_corpus_gives_empty_refs_and_note():
    fake = Fake([GOOD_CLO, CH, AS_OK])
    s = generate.generate_syllabus(INFO, PLOS, 4, "x", retriever=None, llm=fake)
    assert s.required_readings == [] and any("giảng viên cần bổ sung" in n for n in s.notes)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
