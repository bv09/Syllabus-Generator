"""Đọc đề cương .docx theo mẫu HUST (như IT3011) -> Syllabus.

Dò bảng theo TIÊU ĐỀ cột chứ không theo chỉ số bảng. Best-effort: mẫu khác có thể cần chỉnh.
Giới hạn đã biết:
  - Mức đóng góp (I/R/M) trong mẫu ghi THEO TỪNG PLO ("R R"); schema hiện chỉ giữ 1 mức/CLO
    (lấy mức đầu tiên).
  - Tên thành phần đánh giá trong mẫu không có bản tiếng Anh -> en = "".
"""
from __future__ import annotations

import re
import sys
from itertools import zip_longest
from pathlib import Path

from docx import Document

from schema import (CLO, AssessmentComponent, Bi, BlueprintCell, Chapter, CourseInfo,
                    Reference, Syllabus, Workload)


def _uniq(row):
    return list(dict.fromkeys(row.cells))


def _txt(cell) -> str:
    return cell.text.replace("\xa0", " ").strip()


def _paren_bi(s: str) -> Bi:
    s = re.sub(r"\s+", " ", s).strip()
    m = re.match(r"^(.*?)\s*\(([^()]*)\)\s*$", s)
    return Bi(vi=m.group(1).strip(), en=m.group(2).strip()) if m else Bi(vi=s, en="")


def _split_bi(text: str) -> Bi:
    parts = re.split(r"\n\s*\n\s*", text.strip(), maxsplit=1)
    vi = re.sub(r"\s+", " ", parts[0]).strip()
    en = re.sub(r"\s+", " ", parts[1]).strip() if len(parts) > 1 else ""
    return Bi(vi=vi, en=en)


def _num(s: str, default=0.0) -> float:
    m = re.search(r"\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else default


def _find_table(doc, pred):
    for t in doc.tables:
        if t.rows and pred([_txt(c) for c in _uniq(t.rows[0])], t):
            return t
    return None


def _parse_info(t) -> CourseInfo:
    d: dict = {"wl": {}}
    for row in t.rows:
        cells = _uniq(row)
        if len(cells) == 3:                       # dòng con của "Khối lượng"
            label, val = _txt(cells[1]).lower(), _txt(cells[2])
            for key, name in (("lý thuyết", "lecture_periods"), ("bài tập", "exercise_periods"),
                              ("thí nghiệm", "lab_periods"), ("tự học", "self_study_periods")):
                if key in label:
                    d["wl"][name] = _num(val)
            continue
        if len(cells) < 2:
            continue
        label, val = _txt(cells[0]).lower(), _txt(cells[-1])
        if "tên học phần" in label:
            d["title"] = _paren_bi(val)
        elif "mã học phần" in label:
            d["code"] = val.strip()
        elif "khối lượng" in label and "tín chỉ" in val:
            d["credits"] = _num(val)
            m = re.search(r"(\d+(?:\.\d+)?)\s*ECTS", val)
            d["ects"] = float(m.group(1)) if m else None
        elif "phương pháp giảng dạy" in label:
            d["methods"] = [_paren_bi(x) for x in val.split("\n") if x.strip()]
        elif "vai trò" in label:
            d["relation"] = _paren_bi(val)
        elif "điều kiện" in label:
            d["prereq"] = [x.strip() for x in re.split(r"\s+(?:and|và)\s+|,|;", val) if x.strip()]
        elif "ngôn ngữ" in label:
            d["lang"] = val or None
        elif "học kỳ" in label:
            d["semester"] = int(_num(val)) if val else None
        elif "giảng viên" in label:
            d["lecturers"] = [x.strip() for x in re.split(r"[;\n,]", val) if x.strip()]
        elif "qui mô" in label or "quy mô" in label:
            d["group"] = int(_num(val)) if val else None
    wl = d["wl"]
    return CourseInfo(
        title=d["title"], code=d["code"], credits=d.get("credits", 0.0), ects=d.get("ects"),
        workload=Workload(lecture_periods=wl.get("lecture_periods", 0),
                          exercise_periods=wl.get("exercise_periods", 0),
                          lab_periods=wl.get("lab_periods", 0),
                          self_study_periods=wl.get("self_study_periods", 0)),
        teaching_methods=d.get("methods", []),
        relation_to_curriculum=d.get("relation", Bi(vi="", en="")),
        prerequisites=d.get("prereq", []), language_of_instruction=d.get("lang"),
        semester=d.get("semester"), lecturers=d.get("lecturers", []),
        max_group_size=d.get("group"),
    )


def _parse_clos(t) -> list[CLO]:
    out = []
    for row in t.rows[1:]:
        c = _uniq(row)
        if len(c) < 4:
            continue
        plos = [x.strip() for x in _txt(c[2]).split("\n") if x.strip()]
        lvls = [x.strip() for x in _txt(c[3]).split("\n") if x.strip()]
        out.append(CLO(code=_txt(c[0]), description=_split_bi(_txt(c[1])), linked_plos=plos,
                       level=lvls[0] if lvls and lvls[0] in "IRM" else "R"))
    return out


def _parse_assessments(t) -> list[AssessmentComponent]:
    out = []
    for row in t.rows[1:]:
        c = _uniq(row)
        if len(c) >= 3:
            out.append(AssessmentComponent(name=Bi(vi=_txt(c[0]), en=""), weight=_num(_txt(c[1])),
                                           timing=Bi(vi=_txt(c[2]), en="")))
    return out


def _methods(text: str) -> list[Bi]:
    pairs = re.findall(r"([^,()]+?)\s*\(([^()]*)\)", text)
    if pairs:
        return [Bi(vi=a.strip(" ,"), en=b.strip()) for a, b in pairs]
    return [Bi(vi=text.strip(), en="")] if text.strip() else []


def _parse_blueprint(t) -> list[BlueprintCell]:
    codes = [_txt(c) for c in _uniq(t.rows[0])][1:]
    out = []
    for row in t.rows[1:]:
        c = _uniq(row)
        for code, cell in zip(codes, c[1:]):
            ms = _methods(_txt(cell))
            if ms:
                out.append(BlueprintCell(component=_txt(c[0]), clo_code=code, methods=ms))
    return out


def _parse_chapters(t) -> list[Chapter]:
    out = []
    for row in t.rows[1:]:
        c = _uniq(row)
        if len(c) < 3:
            continue
        parts = re.split(r"\n\s*(?=Chapter\s+\d+)", c[0].text, maxsplit=1, flags=re.I)
        vi_lines = [re.sub(r"^[·•\-\s]+", "", x).strip() for x in parts[0].split("\n") if x.strip(" ·•\xa0")]
        en_lines = ([re.sub(r"^[·•\-\s]+", "", x).strip() for x in parts[1].split("\n") if x.strip(" ·•\xa0")]
                    if len(parts) > 1 else [])
        if not vi_lines:
            continue
        m = re.match(r"^CHƯƠNG\s+(\d+)\.?\s*(.*)$", vi_lines[0], re.I)
        e = re.match(r"^Chapter\s+(\d+)\.?\s*(.*)$", en_lines[0], re.I) if en_lines else None
        if not m:
            continue
        topics = [Bi(vi=a or "", en=b or "") for a, b in
                  zip_longest(vi_lines[1:], en_lines[1:] if en_lines else [])]
        out.append(Chapter(
            number=int(m.group(1)), title=Bi(vi=m.group(2).strip(), en=e.group(2).strip() if e else ""),
            topics=topics, clo_codes=re.findall(r"CLO\d+", _txt(c[1])),
            activities=_split_bi(_txt(c[2]))))
    return out


def _parse_refs(t) -> list[Reference]:
    out = []
    for row in t.rows:
        texts = [x for x in (_txt(c) for c in _uniq(row)) if x]
        if texts:
            out.append(Reference(citation=" ".join(texts)))
    return out


def _description(doc) -> Bi:
    vi = en = ""
    on = False
    got: list[str] = []
    for p in doc.paragraphs:
        if p.style.name.startswith("Heading"):
            if on:
                break
            on = "Mô tả học phần" in p.text
            continue
        if on and p.text.strip():
            got.append(re.sub(r"\s+", " ", p.text).strip())
    if got:
        vi = got[0]
        en = got[1] if len(got) > 1 else ""
    return Bi(vi=vi, en=en)


def parse_syllabus_docx(path: str | Path) -> Syllabus:
    doc = Document(str(path))
    info_t = _find_table(doc, lambda h, t: h and "Tên học phần" in h[0])
    clo_t = _find_table(doc, lambda h, t: h and "Mã CĐR" in h[0])
    as_t = _find_table(doc, lambda h, t: len(h) == 3 and "Thành phần đánh giá" in h[0])
    bp_t = _find_table(doc, lambda h, t: len(h) >= 4 and "Thành phần đánh giá" in h[0])
    ch_t = _find_table(doc, lambda h, t: len(h) == 3 and h[0].startswith("Nội dung"))
    if not all([info_t, clo_t, as_t, bp_t, ch_t]):
        raise ValueError("Không nhận ra cấu trúc bảng của đề cương mẫu")
    idx = [id(t._tbl) for t in doc.tables].index(id(ch_t._tbl))
    ref_tables = doc.tables[idx + 1: idx + 3]
    refs = [_parse_refs(t) for t in ref_tables] + [[], []]
    return Syllabus(
        info=_parse_info(info_t), description=_description(doc), clos=_parse_clos(clo_t),
        assessments=_parse_assessments(as_t), blueprint=_parse_blueprint(bp_t),
        chapters=_parse_chapters(ch_t), required_readings=refs[0], recommended_readings=refs[1])


if __name__ == "__main__":
    s = parse_syllabus_docx(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    js = s.model_dump_json(indent=2)
    (out.write_text(js, encoding="utf-8") if out else print(js))
