"""Tests cho M4: Truy hồi theo chương + post-filter."""
import json
from src.rag import Chunk, Retriever
from src.schema import Bi, CourseInfo, Workload
from src.generate import generate_syllabus

INFO = CourseInfo(
    title=Bi(vi="Cấu trúc dữ liệu", en="Data Structures"), code="XX101", credits=2,
    workload=Workload(lecture_periods=30, exercise_periods=15, lab_periods=0, self_study_periods=60),
    teaching_methods=[Bi(vi="Trực tiếp", en="Offline")],
    relation_to_curriculum=Bi(vi="Bắt buộc", en="Mandatory"), prerequisites=[],
    semester=2, lecturers=["A"])
PLOS = ["PLO 01", "PLO 02"]


def _clo(i, vi, en):
    return {"code": f"CLO{i}", "description": {"vi": vi, "en": en}, "linked_plos": ["PLO 01"],
            "level": "R", "knowledge_type": "conceptual", "bloom_level": None}


GOOD_CLO = {"description": {"vi": "Mô tả", "en": "Desc"}, "clos": [
    _clo(1, "Giải thích khái niệm cấu trúc dữ liệu", "Explain data structure concepts"),
    _clo(2, "Cài đặt thuật toán sắp xếp", "Implement sorting algorithms"),
    _clo(3, "Phân tích độ phức tạp thuật toán", "Analyze algorithm complexity"),
    _clo(4, "Đề xuất cấu trúc dữ liệu phù hợp", "Propose suitable data structures"),
]}

CH = {"chapters": [
    {"number": 1, "title": {"vi": "Tổng quan", "en": "Overview"},
     "topics": [{"vi": "Khái niệm", "en": "Concepts"}], "clo_codes": ["CLO1", "CLO3"],
     "activities": {"vi": "Giảng bài", "en": "Lecture"}, "sources": []},
    {"number": 2, "title": {"vi": "Sắp xếp", "en": "Sorting"},
     "topics": [{"vi": "Quick sort", "en": "Quick sort"}], "clo_codes": ["CLO2", "CLO4"],
     "activities": {"vi": "Giảng bài", "en": "Lecture"}, "sources": []},
]}

M = lambda v, e: [{"vi": v, "en": e}]
AS = {"assessments": [
    {"name": {"vi": "Giữa kỳ", "en": "Midterm"}, "weight": 40, "timing": {"vi": "tuần 8", "en": "week 8"},
     "coverage": [{"clo_code": "CLO1", "methods": M("Kiểm tra viết", "Written test")},
                  {"clo_code": "CLO2", "methods": M("Kiểm tra viết", "Written test")}]},
    {"name": {"vi": "Cuối kỳ", "en": "Final"}, "weight": 60, "timing": {"vi": "tuần 17", "en": "week 17"},
     "coverage": [{"clo_code": "CLO3", "methods": M("Kiểm tra viết", "Written test")},
                  {"clo_code": "CLO4", "methods": M("Đồ án", "Project")}]},
]}

REFS = {"required_ids": ["ref:0"], "recommended_ids": []}


class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, system, user):
        self.prompts.append(user)
        return json.dumps(self.replies.pop(0), ensure_ascii=False)


def test_chapter_post_filter_retrieval():
    """Kiểm tra post-filter theo chunk.meta['chapter'] == chapter_num."""
    chunks = [
        Chunk("ref:0", "Cormen. Introduction to Algorithms. 2022.", "reference", "ref.txt", course="XX101"),
        # Chunk thuộc chương 1
        Chunk("ch1:0", "Tổng quan về cấu trúc dữ liệu và giải thuật cơ bản", "general", "ch1.pdf",
              course="XX101", meta={"chapter": 1, "source_pdf": "ch1.pdf"}),
        # Chunk thuộc chương 2
        Chunk("ch2:0", "Các thuật toán sắp xếp: quick sort, merge sort, heap sort", "general", "ch2.pdf",
              course="XX101", meta={"chapter": 2, "source_pdf": "ch2.pdf"}),
        # Chunk không có meta chapter
        Chunk("gen:0", "Kiến thức chung về lập trình", "general", "gen.txt", course="XX101"),
    ]
    r = Retriever(chunks)

    chapter_suggs = [
        {"chapter_num": 1, "title": "Tổng quan", "source_pdf": "ch1.pdf"},
        {"chapter_num": 2, "title": "Sắp xếp", "source_pdf": "ch2.pdf"},
    ]

    fake = FakeLLM([GOOD_CLO, CH, AS, REFS])
    s = generate_syllabus(INFO, PLOS, 2, "cấu trúc dữ liệu", retriever=r, llm=fake,
                          chapter_suggestions=chapter_suggs)

    assert len(s.chapters) == 2
    assert s.required_readings[0].source_id == "ref:0"
    # Prompt của giai đoạn chương phải nhận được ngữ cảnh từ đúng các chương
    ch_prompt = fake.prompts[1]
    assert "ch1:0" in ch_prompt or "ch2:0" in ch_prompt


def test_chapter_filter_by_source_pdf_only():
    """Hướng A: Chỉ dùng source_pdf trong meta (1 PDF = 1 chương), không cần chapter."""
    chunks = [
        Chunk("ref:0", "Cormen. Introduction to Algorithms. 2022.", "reference", "ref.txt", course="XX101"),
        # Chunk KHÔNG có meta chapter, chỉ có source_pdf
        Chunk("c1:0", "Tổng quan về danh sách liên kết và ngăn xếp", "general", "01_intro.pdf",
              course="XX101", meta={"source_pdf": "01_intro.pdf"}),
        Chunk("c2:0", "Các thuật toán sắp xếp nhanh quicksort mergesort", "general", "02_sort.pdf",
              course="XX101", meta={"source_pdf": "02_sort.pdf"}),
    ]
    r = Retriever(chunks)

    chapter_suggs = [
        {"chapter_num": 1, "title": "Tổng quan", "source_pdf": "01_intro.pdf"},
        {"chapter_num": 2, "title": "Sắp xếp", "source_pdf": "02_sort.pdf"},
    ]

    fake = FakeLLM([GOOD_CLO, CH, AS, REFS])
    s = generate_syllabus(INFO, PLOS, 2, "cấu trúc dữ liệu", retriever=r, llm=fake,
                          chapter_suggestions=chapter_suggs)

    assert len(s.chapters) == 2
    ch_prompt = fake.prompts[1]
    assert "c1:0" in ch_prompt or "c2:0" in ch_prompt
