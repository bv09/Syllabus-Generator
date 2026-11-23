"""Sinh đề cương:  python src/main.py inputs/it3011.json --corpus corpus [--evaluate] [--judge]
                  python src/main.py inputs/it3011.json --pdfs data/pdfs/ --corpus corpus
"""
import argparse
import json
import logging
from pathlib import Path

import rag
from evaluate import evaluate, print_report
from generate import generate_syllabus
from rubric import Constraints
from schema import CourseInfo

logging.basicConfig(level=logging.INFO, format="%(message)s")

ap = argparse.ArgumentParser()
ap.add_argument("path", nargs="?", default="inputs/it3011.json")
ap.add_argument("--corpus", default="corpus")
ap.add_argument("--pdfs", nargs="*", default=None, help="thư mục/file PDF nạp vào corpus/<MÃ>/books/")
ap.add_argument("--evaluate", action="store_true", help="chấm ngay bằng rubric xác định")
ap.add_argument("--judge", action="store_true", help="thêm LLM-judge (tốn API)")
ap.add_argument("--allow-self", action="store_true",
                help="KHÔNG loại đề cương cùng mã học phần khỏi ngữ cảnh (gây rò rỉ khi đánh giá)")
a = ap.parse_args()

cfg = json.loads(Path(a.path).read_text(encoding="utf-8"))
info = CourseInfo(**cfg["info"])
cons = Constraints.from_dict(cfg.get("constraints"))

# --- M0: nạp PDF vào corpus nếu có --pdfs ------------------------------------
suggested_chapters = None
if a.pdfs:
    from pdf_ingest import resolve_pdf_paths, ingest_pdfs, print_ingest_report
    pdf_files = resolve_pdf_paths(a.pdfs)
    if pdf_files:
        reports, suggested_chapters = ingest_pdfs(pdf_files, info.code, Path(a.corpus))
        print_ingest_report(reports, suggested_chapters)

retr = rag.build_retriever(a.corpus)
n_chunks = len(retr.chunks) if retr else 0
print(f"[RAG] Đã nạp thành công {n_chunks} đoạn trích từ '{a.corpus}' (BM25: sẵn sàng)", flush=True)
n_ch = len(suggested_chapters) if suggested_chapters else cfg["num_chapters"]

print(f"Bắt đầu sinh đề cương: {info.code} - {info.title.vi} ({n_ch} chương)...", flush=True)
print("Đang gọi LLM lần lượt qua 4 giai đoạn, quá trình này mất khoảng 20-40 giây...", flush=True)

s = generate_syllabus(info, cfg["plos"], n_ch, cfg.get("hint", ""), retriever=retr,
                      constraints=cons, exclude_course="" if a.allow_self else None,
                      chapter_suggestions=suggested_chapters,
                      fixed_assessments=cfg.get("assessments"))

Path("outputs").mkdir(exist_ok=True)
out = Path(f"outputs/{info.code}.json")
out.write_text(s.model_dump_json(indent=2), encoding="utf-8")
print("Đã ghi:", out, flush=True)
for n in s.notes:
    print("  ! cần người duyệt:", n, flush=True)

if a.evaluate or a.judge:
    jl = None
    if a.judge:
        from llm import judge_chat
        jl = judge_chat
    print_report(evaluate(s, cfg["plos"], cfg["num_chapters"], retr, cons, jl))
