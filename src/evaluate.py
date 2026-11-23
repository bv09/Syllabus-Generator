"""Chấm một đề cương:
    python src/evaluate.py outputs/IT3011.json --plos "PLO 01,PLO 02,PLO 03,PLO 04" --corpus corpus
    python src/evaluate.py corpus/syllabus_IT3011_mau.docx            # chấm đề cương mẫu (baseline)
    thêm --judge để chạy LLM-judge (tốn API)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import rag
import rubric
from rubric import Constraints
from schema import Syllabus


def load(path: Path) -> Syllabus:
    if path.suffix.lower() == ".docx":
        from ingest_docx import parse_syllabus_docx
        return parse_syllabus_docx(path)
    return Syllabus.model_validate(json.loads(path.read_text(encoding="utf-8")))


def evaluate(s: Syllabus, plos=None, n_chapters=None, retriever=None, constraints=None,
             judge_llm=None) -> dict:
    det = rubric.run_deterministic(s, constraints, plos, n_chapters, rag.reference_texts(retriever))
    verdicts = None
    if judge_llm:
        import judge as J
        ctx = ""
        if retriever:
            q = f"{s.info.title.vi} {s.info.title.en} " + " ".join(c.title.vi for c in s.chapters)
            hits = retriever.search(q, k=6, exclude_course=s.info.code)
            ctx = "\n".join(f"[{h.chunk.id}] {h.chunk.text[:600]}" for h in hits)
        verdicts = J.judge(s, ctx, judge_llm)
    return rubric.aggregate(det, verdicts)


def print_report(rep: dict) -> None:
    tag = " (CHỈ tiêu chí xác định; chưa có LLM-judge)" if rep["partial"] else ""
    print(f"\nĐiểm: {rep['total']}/100{tag} | cổng chặn: "
          f"{'ĐẠT' if rep['passed'] else 'KHÔNG ĐẠT ' + str(rep['gate_failed'])}")
    for c in rep["criteria"]:
        print(f"  {c['id']} {c['name']:<58} {c['score']*c['weight']:>4.1f}/{c['weight']:g}")
        for f in c.get("findings", []):
            w = f" [{f['where']}]" if f["where"] else ""
            print(f"      - {f['severity']:<5}{w} {f['message']}")
        if c["kind"] == "llm":
            j = c["judge"]
            print(f"      judge {j['score']}/4: {j['issue']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--plos", default="")
    ap.add_argument("--chapters", type=int)
    ap.add_argument("--corpus")
    ap.add_argument("--constraints", help="file JSON ghi đè Constraints")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()

    s = load(Path(a.path))
    plos = [p.strip() for p in a.plos.split(",") if p.strip()] or None
    cons = Constraints.from_dict(json.loads(Path(a.constraints).read_text("utf-8"))) if a.constraints else None
    retr = rag.build_retriever(a.corpus, use_dense=False)
    jl = None
    if a.judge:
        from llm import judge_chat
        jl = judge_chat
    rep = evaluate(s, plos, a.chapters, retr, cons, jl)
    print_report(rep)
    out = Path(a.out) if a.out else Path("outputs") / f"{s.info.code}.report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nĐã ghi:", out)


if __name__ == "__main__":
    main()
