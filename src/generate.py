"""Sinh đề cương theo 4 giai đoạn. Nguyên tắc thiết kế:

  1. Cái gì suy ra được bằng code thì KHÔNG nhờ LLM: blueprint dựng từ 'coverage' của từng thành phần
     đánh giá; bloom_level do bloom.py gán; tài liệu tham khảo chỉ được CHỌN theo id từ corpus.
  2. Mỗi giai đoạn có validator = các hàm check_* của rubric.py; lỗi 'error' bị gửi ngược lại LLM
     để sửa (retry có phản hồi cụ thể thay vì 'thử lại').
  3. Cảnh báo còn sót sau khi hết retry được ghi vào Syllabus.notes cho người duyệt.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Callable

from pydantic import BaseModel, ValidationError

import bloom
import rag
import rubric
from rubric import Constraints, Finding
from schema import (CLO, AssessmentComponent, Bi, BlueprintCell, Chapter, CourseInfo, Reference,
                    Syllabus)

log = logging.getLogger("generate")
LLM = Callable[[str, str], str]


# --- Mô hình dữ liệu của từng giai đoạn (cái LLM trả về) -----------------------------
class StageCLO(BaseModel):
    description: Bi
    clos: list[CLO]


class StageChapters(BaseModel):
    chapters: list[Chapter]


class Coverage(BaseModel):
    clo_code: str
    methods: list[Bi]


class AssessmentDraft(BaseModel):
    name: Bi
    weight: float
    timing: Bi
    coverage: list[Coverage]


class StageAssess(BaseModel):
    assessments: list[AssessmentDraft]


class StageRefs(BaseModel):
    required_ids: list[str]
    recommended_ids: list[str]


from prompts import SYSTEM, prompt_clos, prompt_assessment


class StageError(RuntimeError):
    pass


def parse_model(raw: str, cls):
    t = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    a, b = t.find("{"), t.rfind("}")
    if a == -1 or b <= a:
        raise ValueError(f"Không tìm thấy đối tượng JSON (đoạn đầu: {raw[:120]!r})")
    content = re.sub(r",\s*([\]}])", r"\1", t[a:b + 1])
    return cls.model_validate_json(content)


def run_stage(name: str, prompt: str, cls, check: Callable[[BaseModel], list[Finding]],
              llm: LLM, max_retry: int = 3):
    """Trả (đối tượng tốt nhất, các cảnh báo còn lại). Raise StageError nếu vẫn còn 'error'."""
    best, best_key, best_fs = None, None, []
    feedback = ""
    for attempt in range(1, max_retry + 1):
        user = prompt if not feedback else (
            prompt + f"\n\nLần trước có vấn đề:\n{feedback}\nSửa TẤT CẢ các lỗi trên, giữ đúng cấu trúc schema (mọi trường song ngữ đều là object {{vi, en}}), rồi trả lại TOÀN BỘ JSON.")
        print(f"  -> Đang xử lý giai đoạn: {name} (lần {attempt})...", flush=True)
        raw = llm(SYSTEM, user)
        try:
            obj = parse_model(raw, cls)
        except (ValidationError, ValueError) as e:
            feedback = f"JSON không hợp lệ hoặc sai schema: {str(e)[:800]}"
            log.warning("[%s] lần %d: %s", name, attempt, feedback[:120])
            continue
        fs = check(obj)
        key = (sum(f.severity == "error" for f in fs), sum(f.severity == "warn" for f in fs))
        if best is None or key < best_key:
            best, best_key, best_fs = obj, key, fs
        log.info("[%s] lần %d: %d error, %d warn", name, attempt, *key)
        if key == (0, 0):
            break
        feedback = "\n".join(f"- [{f.severity}] {f.where + ': ' if f.where else ''}{f.message}"
                             for f in fs if f.severity != "info")[:1800]
    if best is None or best_key[0] > 0:
        raise StageError(f"Giai đoạn '{name}' vẫn lỗi sau {max_retry} lần:\n"
                         + "\n".join(f"- {f.message}" for f in best_fs if f.severity == "error"))
    return best, [f for f in best_fs if f.severity == "warn"]


# --- Ngữ cảnh RAG ----------------------------------------------------------------------
def fmt_ctx(hits: list[rag.Hit], limit: int = 700) -> str:
    if not hits:
        return "(trống: không có tài liệu nội bộ phù hợp; dùng kiến thức chung và nêu rõ trong notes)"
    return "\n".join(f"[{h.chunk.id}] ({h.chunk.kind}) {h.chunk.text[:limit]}" for h in hits)


def _schema(cls) -> str:
    return json.dumps(cls.model_json_schema(), ensure_ascii=False)


def _head(info: CourseInfo, hint: str) -> str:
    return (f"Học phần: {info.title.vi} ({info.title.en}), mã {info.code}, {info.credits} tín chỉ; "
            f"học kỳ {info.semester}; số tiết LT/BT/TN: {info.workload.lecture_periods:g}/"
            f"{info.workload.exercise_periods:g}/{info.workload.lab_periods:g}\n"
            f"Gợi ý nội dung: {hint}")


# --- Giai đoạn A: mô tả + CLO ------------------------------------------------------------
def stage_clos(info, plos, hint, ctx, c: Constraints, llm, max_retry):
    prompt = prompt_clos(_head(info, hint), ctx, c.clo_min, c.clo_max, plos, _schema(StageCLO))

    def check(o: StageCLO) -> list[Finding]:
        fs = rubric.check_plo_links(o.clos, plos)
        fs += rubric.check_bloom_verbs(o.clos)
        fs += rubric.check_progression(o.clos, c)
        if not o.description.vi.strip() or not o.description.en.strip():
            fs.append(Finding("desc_empty", "error", "Thiếu mô tả vi hoặc en"))
        if c.clo_min == c.clo_max and len(o.clos) != c.clo_min:
            fs.append(Finding("clo_count_exact", "error", f"Yêu cầu ĐÚNG {c.clo_min} CLO, hiện có {len(o.clos)} CLO"))
        levels = [bloom.analyze_clo(x)["level"] for x in o.clos if bloom.analyze_clo(x)["level"]]
        if len(levels) >= 4 and len(levels) - len(set(levels)) >= 2:
            fs.append(Finding("clo_dup_level", "error",
                              "Có nhiều CLO trùng mức nhận thức hoặc lặp động từ (ví dụ nhiều CLO cùng là 'cài đặt'). "
                              "Mỗi CLO phải tương ứng một bậc nhận thức Bloom riêng: Hiểu, Vận dụng, Phân tích, Đánh giá, Sáng tạo."))
        for i, x in enumerate(o.clos, 1):
            if x.code != f"CLO{i}":
                fs.append(Finding("clo_code", "error", f"Mã phải là CLO{i}, đang là {x.code}"))
        return fs

    obj, warns = run_stage("CLO", prompt, StageCLO, check, llm, max_retry)
    for x in obj.clos:  # gán bằng code, không tin LLM
        lv = bloom.analyze_clo(x)["level"]
        x.bloom_level = lv.name.lower() if lv else None
    obj.clos.sort(key=lambda x: (bloom.analyze_clo(x)["level"] or 0, x.code))
    for i, x in enumerate(obj.clos, 1):
        x.code = f"CLO{i}"
    return obj, warns


# --- Giai đoạn B: chương -------------------------------------------------------------------
def stage_chapters(info, n, hint, clos, hits, c, llm, max_retry):
    ids = {h.chunk.id for h in hits}
    clo_txt = "\n".join(f"{x.code} [{x.bloom_level}]: {x.description.vi}" for x in clos)
    prompt = f"""{_head(info, hint)}

CLO đã chốt:
{clo_txt}

NGỮ CẢNH:
{fmt_ctx(hits)}

Viết ĐÚNG {n} chương, đánh số từ 1, mỗi chương có title, topics (4-6 mục) và activities (song ngữ).
Quy tắc:
- BẮT BUỘC: Mỗi chương PHẢI có ít nhất 1-3 mã CLO trong clo_codes (ví dụ ["CLO1"]), TUYỆT ĐỐI KHÔNG để mảng rỗng [].
- clo_codes chỉ chọn trong: {[x.code for x in clos]}. Mỗi CLO phải được ít nhất một chương phát triển.
- sources = id các đoạn NGỮ CẢNH bạn dựa vào (ví dụ "abc:3"); để [] nếu dùng kiến thức chung.
- Thứ tự chương theo logic sư phạm (khái niệm nền trước, ứng dụng sau).

Schema JSON:
{_schema(StageChapters)}"""

    def check(o: StageChapters) -> list[Finding]:
        fs = rubric.check_chapters(o.chapters, [x.code for x in clos], n, c)
        for ch in o.chapters:
            fixed_src = []
            for s in ch.sources:
                if s in ids:
                    fixed_src.append(s)
                else:
                    matched = next((i for i in ids if s.lower().replace("0", "") in i.lower().replace("0", "")), None)
                    if matched:
                        fixed_src.append(matched)
                    else:
                        fs.append(Finding("ch_source", "warn", f"sources không có trong ngữ cảnh: {s}",
                                          f"Chương {ch.number}"))
            ch.sources = fixed_src
        return fs

    return run_stage("chương", prompt, StageChapters, check, llm, max_retry)


# --- Giai đoạn C: đánh giá (blueprint dựng bằng code) -----------------------------------------
def _to_assessment(o: StageAssess):
    comps, cells = [], []
    for a in o.assessments:
        comps.append(AssessmentComponent(name=a.name, weight=a.weight, timing=a.timing))
        cells += [BlueprintCell(component=a.name.vi, clo_code=cv.clo_code, methods=cv.methods)
                  for cv in a.coverage]
    return comps, cells


def stage_assessment(info, clos, hint, c, llm, max_retry, fixed_assessments: list | None = None):
    clo_txt = "\n".join(f"{x.code} [{x.bloom_level}]: {x.description.vi}" for x in clos)
    fixed_txt = json.dumps(fixed_assessments, ensure_ascii=False, indent=2) if fixed_assessments else ""
    prompt = prompt_assessment(_head(info, hint), clo_txt, _schema(StageAssess), fixed_txt)

    def check(o: StageAssess) -> list[Finding]:
        if fixed_assessments and len(o.assessments) == len(fixed_assessments):
            for draft, fa in zip(o.assessments, fixed_assessments):
                draft.name = Bi(**fa["name"])
                draft.weight = float(fa["weight"])
                draft.timing = Bi(**fa["timing"])
        comps, cells = _to_assessment(o)
        return (rubric.check_weights(comps) + rubric.check_coverage(comps, cells, [x.code for x in clos])
                + rubric.check_alignment(clos, cells))

    obj, warns = run_stage("đánh giá", prompt, StageAssess, check, llm, max_retry)
    return _to_assessment(obj), warns


# --- Giai đoạn D: tài liệu (chỉ CHỌN id từ corpus) ----------------------------------------------
def stage_refs(info, hint, cands: list[rag.Hit], llm, max_retry):
    if not cands:
        return [], [], ["Corpus không có tài liệu tham khảo nào cho học phần này: "
                        "giảng viên cần bổ sung (hệ thống không tự sinh để tránh bịa)."]
    by_id = {h.chunk.id: h.chunk for h in cands}
    lst = "\n".join(f"[{i}] {c.text}" for i, c in by_id.items())
    prompt = f"""{_head(info, hint)}

DANH SÁCH TÀI LIỆU ỨNG VIÊN (từ kho của bộ môn):
{lst}

Chọn id các tài liệu phù hợp: required_ids (1-2 tài liệu chính), recommended_ids (còn lại, nếu phù hợp).
Chỉ dùng id có trong danh sách; một id không xuất hiện ở cả hai danh sách; để [] nếu không tài liệu nào phù hợp.

Schema JSON:
{_schema(StageRefs)}"""

    def check(o: StageRefs) -> list[Finding]:
        fs = []
        allid = [*o.required_ids, *o.recommended_ids]
        bad = [i for i in allid if i not in by_id]
        if bad:
            fs.append(Finding("ref_id", "error", f"id không có trong danh sách: {bad}"))
        if len(set(allid)) != len(allid):
            fs.append(Finding("ref_dup", "error", "Một id xuất hiện nhiều lần"))
        return fs

    obj, _ = run_stage("tài liệu", prompt, StageRefs, check, llm, max_retry)
    mk = lambda ids: [Reference(citation=by_id[i].text, source_id=i) for i in ids]
    notes = [] if obj.required_ids else ["Chưa chọn được tài liệu bắt buộc: cần giảng viên bổ sung."]
    return mk(obj.required_ids), mk(obj.recommended_ids), notes


# --- Điều phối ------------------------------------------------------------------------------------
def generate_syllabus(info: CourseInfo, plos: list[str], n_chapters: int, hint: str,
                      retriever: rag.Retriever | None = None, constraints: Constraints | None = None,
                      llm: LLM | None = None, max_retry: int = 3,
                      exclude_course: str | None = None,
                      chapter_suggestions: list | None = None,
                      fixed_assessments: list | None = None) -> Syllabus:
    if llm is None:
        from llm import chat
        llm = lambda s, u: chat(s, u, temperature=0.2)  # noqa: E731
    c = constraints or Constraints()
    q = f"{info.title.vi} {info.title.en} {hint}"
    # Mặc định loại chính đề cương cùng mã học phần khỏi ngữ cảnh để tránh rò rỉ đáp án khi đánh giá.
    excl = exclude_course if exclude_course is not None else info.code
    ctx_hits = (retriever.search(q, k=8, kinds={rag.SYLLABUS, rag.GENERAL, "toc", "plo", "regulation"},
                                  exclude_course=excl) if retriever else [])
    if chapter_suggestions and retriever:
        ch_hits = []
        for ch in chapter_suggestions:
            c_title = ch.get("title") if isinstance(ch, dict) else getattr(ch, "title", "")
            c_src = ch.get("source_pdf") if isinstance(ch, dict) else getattr(ch, "source_pdf", None)
            c_num = ch.get("chapter_num") if isinstance(ch, dict) else getattr(ch, "chapter_num", None)
            sh = retriever.search(f"{info.title.vi} {c_title}", k=20, kinds={rag.GENERAL})
            filtered = [h for h in sh if (c_src and h.chunk.meta.get("source_pdf") == c_src)
                        or (c_num is not None and h.chunk.meta.get("chapter") == c_num)]
            ch_hits.extend(filtered[:5] if filtered else sh[:5])
        if ch_hits:
            ctx_hits = ch_hits
    if ctx_hits:
        print(f"[RAG] Đã truy hồi {len(ctx_hits)} đoạn trích làm căn cứ nội dung cho LLM", flush=True)
    ref_hits = []
    if retriever:
        all_refs = [x for x in retriever.chunks if x.kind == rag.REFERENCE]
        ref_hits = ([rag.Hit(x, 0.0) for x in all_refs] if len(all_refs) <= 10
                    else retriever.search(q, k=10, kinds={rag.REFERENCE}))
    notes: list[str] = []
    if retriever is None:
        notes.append("Không có corpus: toàn bộ nội dung dựa trên kiến thức chung của LLM, chưa được đối chiếu.")
    elif not ctx_hits:
        notes.append("Corpus không có ngữ cảnh liên quan: nội dung dựa trên kiến thức chung của LLM.")

    a, w = stage_clos(info, plos, hint, fmt_ctx(ctx_hits), c, llm, max_retry)
    notes += [f"[CLO] {x.message}" for x in w]
    b, w = stage_chapters(info, n_chapters, hint, a.clos, ctx_hits, c, llm, max_retry)
    notes += [f"[chương] {x.message}" for x in w]
    (comps, cells), w = stage_assessment(info, a.clos, hint, c, llm, max_retry, fixed_assessments)
    notes += [f"[đánh giá] {x.message}" for x in w]
    req, rec, n = stage_refs(info, hint, ref_hits, llm, max_retry)
    notes += n

    return Syllabus(info=info, description=a.description, clos=a.clos, assessments=comps, blueprint=cells,
                    chapters=b.chapters, required_readings=req, recommended_readings=rec, notes=notes)
