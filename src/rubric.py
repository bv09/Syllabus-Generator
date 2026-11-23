"""Rubric đánh giá đề cương. Một nguồn sự thật duy nhất:

  - Các hàm check_* vừa là VALIDATOR khi sinh (generate.py), vừa là tiêu chí chấm (evaluate.py).
  - Tiêu chí 'det' tính bằng code (tái lập 100%). Tiêu chí 'llm' do judge.py chấm theo anchors.

CẢNH BÁO: vì generator dùng chính các luật này làm ràng buộc, điểm 'det' của đề cương do máy sinh
gần như chắc chắn cao -> nó là CỔNG CHẶN lỗi cấu trúc, KHÔNG phải thước đo chất lượng. Chất lượng
thật phải đo bằng LLM-judge đã hiệu chuẩn + chấm tay của giảng viên (xem README).
Trọng số và công thức điểm là chủ quan, cần hiệu chuẩn với dữ liệu thật.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Literal

import bloom
from bloom import Cog
from schema import CLO, AssessmentComponent, BlueprintCell, Chapter, Syllabus

Sev = Literal["error", "warn", "info"]


@dataclass
class Finding:
    rule: str
    severity: Sev
    message: str
    where: str = ""


@dataclass
class Constraints:
    clo_min: int = 4
    clo_max: int = 6
    top_level_min: int = 3                 # ít nhất 1 CLO từ mức Vận dụng trở lên
    max_all_clo_chapter_ratio: float = 0.5  # >50% chương gắn với TẤT CẢ CLO -> ánh xạ vô nghĩa

    @classmethod
    def from_dict(cls, d: dict | None) -> "Constraints":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _plo(x: str) -> str:
    return re.sub(r"\s+", "", x).upper()


_VI_CHARS = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.I)


# --- Kiểm tra cấu trúc --------------------------------------------------------
def check_weights(assessments: list[AssessmentComponent]) -> list[Finding]:
    f: list[Finding] = []
    total = sum(a.weight for a in assessments)
    if abs(total - 100) > 0.01:
        f.append(Finding("weights_sum", "error", f"Tổng tỉ trọng = {total:g}, phải bằng 100"))
    for a in assessments:
        if a.weight <= 0:
            f.append(Finding("weights_positive", "error", "Tỉ trọng phải > 0", a.name.vi))
    dup = [n for n, c in Counter(a.name.vi for a in assessments).items() if c > 1]
    for n in dup:
        f.append(Finding("weights_dup", "error", "Trùng tên thành phần đánh giá", n))
    return f


def check_coverage(assessments: list[AssessmentComponent], blueprint: list[BlueprintCell],
                   clo_codes: Iterable[str]) -> list[Finding]:
    f: list[Finding] = []
    clo_codes = list(clo_codes)
    names = {a.name.vi for a in assessments}
    by_comp: dict[str, set[str]] = defaultdict(set)
    by_clo: dict[str, set[str]] = defaultdict(set)
    pairs: Counter = Counter()
    for c in blueprint:
        pairs[(c.component, c.clo_code)] += 1
        if c.component not in names:
            f.append(Finding("bp_component", "error", "component không khớp assessments[].name.vi",
                             c.component))
        if c.clo_code not in clo_codes:
            f.append(Finding("bp_clo", "error", "clo_code không tồn tại", c.clo_code))
        if not c.methods:
            f.append(Finding("bp_methods", "error", "Ô ma trận không có phương pháp",
                             f"{c.component} × {c.clo_code}"))
        by_comp[c.component].add(c.clo_code)
        by_clo[c.clo_code].add(c.component)
    for code in clo_codes:
        if not by_clo[code]:
            f.append(Finding("clo_unassessed", "error", "CLO không được thành phần đánh giá nào đo",
                             code))
    for n in names:
        if not by_comp[n]:
            f.append(Finding("comp_empty", "error", "Thành phần đánh giá không đo CLO nào", n))
    for (comp, code), n in pairs.items():
        if n > 1:
            f.append(Finding("bp_dup", "warn", "Ô ma trận bị lặp", f"{comp} × {code}"))
    return f


def check_chapters(chapters: list[Chapter], clo_codes: Iterable[str], n_expected: int | None,
                   c: Constraints) -> list[Finding]:
    f: list[Finding] = []
    clo_codes = list(clo_codes)
    nums = [ch.number for ch in chapters]
    if nums != list(range(1, len(nums) + 1)):
        f.append(Finding("ch_numbering", "error", f"Số chương không liên tục từ 1: {nums}"))
    if n_expected and len(chapters) != n_expected:
        f.append(Finding("ch_count", "error", f"Có {len(chapters)} chương, yêu cầu {n_expected}"))
    covered: set[str] = set()
    for ch in chapters:
        w = f"Chương {ch.number}"
        if not ch.clo_codes:
            f.append(Finding("ch_no_clo", "error", "Chương không gắn CLO nào", w))
        if not ch.topics:
            f.append(Finding("ch_no_topics", "error", "Chương không có mục nội dung", w))
        for code in ch.clo_codes:
            if code not in clo_codes:
                f.append(Finding("ch_clo_unknown", "error", f"CLO không tồn tại: {code}", w))
        covered |= set(ch.clo_codes)
    for code in clo_codes:
        if code not in covered:
            f.append(Finding("clo_no_chapter", "error", "CLO không được chương nào phát triển", code))
    if chapters and len(clo_codes) > 1:
        full = sum(set(clo_codes) <= set(ch.clo_codes) for ch in chapters) / len(chapters)
        if full > c.max_all_clo_chapter_ratio:
            f.append(Finding(
                "ch_mapping_flat", "warn",
                f"{full:.0%} chương được gắn với TẤT CẢ CLO -> ánh xạ chương-CLO không còn khả năng "
                f"phân biệt (không cho biết chương nào thực sự dạy CLO nào)."))
    return f


def check_plo_links(clos: list[CLO], allowed: list[str] | None) -> list[Finding]:
    f: list[Finding] = []
    codes = Counter(c.code for c in clos)
    for code, n in codes.items():
        if n > 1:
            f.append(Finding("clo_dup", "error", "Mã CLO bị lặp", code))
    allowed_n = {_plo(a) for a in allowed} if allowed else None
    for c in clos:
        if not c.linked_plos:
            f.append(Finding("plo_empty", "error", "CLO không liên kết PLO nào", c.code))
        elif allowed_n is not None:
            bad = [p for p in c.linked_plos if _plo(p) not in allowed_n]
            if bad:
                f.append(Finding("plo_invalid", "error", f"PLO ngoài danh sách cho phép: {bad}", c.code))
    return f


def _iter_bi(s: Syllabus):
    yield "info.title", s.info.title
    yield "info.relation_to_curriculum", s.info.relation_to_curriculum
    for i, m in enumerate(s.info.teaching_methods):
        yield f"info.teaching_methods[{i}]", m
    yield "description", s.description
    for c in s.clos:
        yield f"{c.code}.description", c.description
    for a in s.assessments:
        yield f"assessment '{a.name.vi}'.name", a.name
        yield f"assessment '{a.name.vi}'.timing", a.timing
    for ch in s.chapters:
        yield f"Chương {ch.number}.title", ch.title
        for i, t in enumerate(ch.topics):
            yield f"Chương {ch.number}.topics[{i}]", t
        yield f"Chương {ch.number}.activities", ch.activities


def check_bilingual(s: Syllabus) -> list[Finding]:
    f: list[Finding] = []
    for where, bi in _iter_bi(s):
        if not bi.vi.strip():
            f.append(Finding("bi_vi_empty", "error", "Thiếu bản tiếng Việt", where))
        if not bi.en.strip():
            f.append(Finding("bi_en_empty", "warn", "Thiếu bản tiếng Anh", where))
        elif _VI_CHARS.search(bi.en):
            f.append(Finding("bi_en_has_vi", "warn", "Bản 'en' chứa ký tự tiếng Việt", where))
        elif bi.vi.strip() == bi.en.strip() and len(bi.vi.split()) > 3:
            f.append(Finding("bi_same", "warn", "Bản vi và en giống hệt nhau", where))
    if s.info.language_of_instruction is None:
        f.append(Finding("info_lang", "warn", "Chưa có ngôn ngữ giảng dạy", "info"))
    if s.info.max_group_size is None:
        f.append(Finding("info_group", "info", "Chưa có qui mô lớp tối đa", "info"))
    return f


# --- Bloom ----------------------------------------------------------------------
def check_bloom_verbs(clos: list[CLO]) -> list[Finding]:
    f: list[Finding] = []
    for c in clos:
        r = bloom.analyze_clo(c)
        if r["vi"] is None and r["en"] is None:
            f.append(Finding("bloom_no_verb", "error",
                             "Không tìm thấy động từ Bloom đo được ở đầu CLO "
                             "(dùng động từ trong bảng hoặc bổ sung từ điển bloom.py)", c.code))
            continue
        if r["vi"] is None or r["en"] is None:
            f.append(Finding("bloom_one_lang", "warn",
                             "Chỉ nhận được động từ ở một ngôn ngữ", c.code))
        elif r["vi"].level != r["en"].level:
            f.append(Finding("bloom_vi_en_mismatch", "warn",
                             f"vi = {r['vi'].level.name}, en = {r['en'].level.name}: "
                             f"bản dịch lệch mức nhận thức", c.code))
        main = r["vi"] or r["en"]
        if main.spread >= 3:
            f.append(Finding("bloom_compound", "info",
                             f"CLO gộp nhiều mức ({main.lowest.name}→{main.level.name}); "
                             f"khó đánh giá riêng từng mức", c.code))
    return f


def check_progression(clos: list[CLO], c: Constraints) -> list[Finding]:
    f: list[Finding] = []
    if not (c.clo_min <= len(clos) <= c.clo_max):
        f.append(Finding("clo_count", "warn",
                         f"Có {len(clos)} CLO, khoảng khuyến nghị {c.clo_min}-{c.clo_max}"))
    levels = {x.code: bloom.analyze_clo(x)["level"] for x in clos}
    known = [v for v in levels.values() if v is not None]
    if known and max(known) < c.top_level_min:
        f.append(Finding("bloom_ceiling", "warn",
                         f"CLO cao nhất chỉ ở mức {max(known).name}; kỳ vọng ≥ {Cog(c.top_level_min).name}"))
    if len(known) >= 3 and len(set(known)) == 1:
        f.append(Finding("bloom_flat", "warn",
                         f"Tất cả CLO cùng mức {known[0].name}: không có tiến triển nhận thức"))
    for x in clos:  # I/R/M là thang đóng góp CTĐT, KHÔNG phải Bloom: chỉ nhắc khi lệch hẳn
        lv = levels[x.code]
        if lv is None:
            continue
        if x.level == "I" and lv >= Cog.ANALYZE:
            f.append(Finding("irm_vs_bloom", "info", "Mức 'I' nhưng động từ ở mức cao", x.code))
        if x.level == "M" and lv <= Cog.UNDERSTAND:
            f.append(Finding("irm_vs_bloom", "info", "Mức 'M' nhưng động từ ở mức thấp", x.code))
    return f


def check_alignment(clos: list[CLO], blueprint: list[BlueprintCell]) -> list[Finding]:
    """Constructive alignment (mức thô): phương pháp đo CLO phải có 'trần' >= mức CLO.
    Giới hạn: chỉ nhìn TÊN phương pháp, không nhìn đề thi thật."""
    f: list[Finding] = []
    for c in clos:
        lv = bloom.analyze_clo(c)["level"]
        if lv is None:
            continue
        ceils: list[Cog] = []
        unknown: set[str] = set()
        for cell in blueprint:
            if cell.clo_code != c.code:
                continue
            for m in cell.methods:
                cl = bloom.method_ceiling(m.vi) or bloom.method_ceiling(m.en)
                (ceils.append(cl) if cl else unknown.add(m.vi))
        if ceils and max(ceils) < lv:
            f.append(Finding("align_ceiling", "warn",
                             f"CLO mức {lv.name} nhưng phương pháp đo chỉ tới {max(ceils).name}", c.code))
        if not ceils and unknown:
            f.append(Finding("align_unknown", "info",
                             f"Không nhận ra phương pháp: {sorted(unknown)}", c.code))
    return f


def check_references(s: Syllabus, corpus_refs: set[str] | None) -> list[Finding]:
    f: list[Finding] = []
    if not s.required_readings:
        f.append(Finding("ref_required_empty", "warn", "Chưa có tài liệu bắt buộc (cần giảng viên bổ sung)"))
    if corpus_refs is None:
        f.append(Finding("ref_provenance_skipped", "info", "Không kiểm tra nguồn gốc tài liệu (không có corpus)"))
        return f
    norm = lambda x: re.sub(r"\s+", " ", x).strip().lower()
    known = {norm(x) for x in corpus_refs}
    for r in [*s.required_readings, *s.recommended_readings]:
        if norm(r.citation) not in known:
            f.append(Finding("ref_not_in_corpus", "error",
                             "Tài liệu không có trong corpus -> nghi bịa", r.citation[:80]))
    return f


# --- Đăng ký tiêu chí & tổng hợp điểm -----------------------------------------
@dataclass(frozen=True)
class Criterion:
    id: str
    name: str
    weight: float
    kind: Literal["det", "llm"]
    gate: bool = False
    anchors: dict | None = None


_L = lambda a, b, c, d: {1: a, 2: b, 3: c, 4: d}
CRITERIA: list[Criterion] = [
    Criterion("S1", "Tỉ trọng đánh giá hợp lệ", 10, "det", True),
    Criterion("S2", "Ma trận đánh giá phủ đủ CLO", 10, "det", True),
    Criterion("S3", "Ánh xạ chương–CLO đầy đủ & có phân biệt", 10, "det", True),
    Criterion("S4", "Liên kết PLO hợp lệ", 5, "det", True),
    Criterion("S5", "Song ngữ & đủ trường thông tin", 5, "det"),
    Criterion("B1", "CLO có động từ Bloom đo được, vi/en nhất quán", 10, "det", True),
    Criterion("B2", "Tiến triển nhận thức giữa các CLO", 5, "det"),
    Criterion("B3", "Phương pháp đánh giá đủ 'trần' cho mức CLO", 10, "det"),
    Criterion("G1", "Tài liệu tham khảo có nguồn trong corpus", 10, "det", True),
    Criterion("L1", "Nội dung chương đúng, đủ, bám ngữ cảnh/mô tả", 10, "llm", anchors=_L(
        "Nhiều chủ đề sai/lạc đề hoặc thiếu mảng chính của học phần",
        "Đa số đúng nhưng thiếu/thừa rõ rệt hoặc có chủ đề không có căn cứ",
        "Đúng và đủ, sai lệch nhỏ", "Đúng, đủ, bám sát ngữ cảnh, thứ tự hợp lý")),
    Criterion("L2", "CLO cụ thể, đo được, phù hợp học phần", 10, "llm", anchors=_L(
        "Mơ hồ/chung chung, không đo được hoặc không thuộc học phần",
        "Đo được một phần; nhiều CLO trùng ý hoặc quá rộng",
        "Đa số cụ thể, đo được; vài CLO còn chung", "Mỗi CLO cụ thể, đo được, không trùng, đúng mức"
    )),
    Criterion("L3", "Mạch sư phạm: thứ tự chương, điều kiện tiên quyết, khối lượng", 5, "llm", anchors=_L(
        "Thứ tự phi lý hoặc khối lượng không khả thi",
        "Có chỗ bất hợp lý đáng kể",
        "Hợp lý, vài điểm nhỏ cần chỉnh", "Mạch chặt chẽ, khả thi trong số tiết")),
]
CRIT = {c.id: c for c in CRITERIA}


def run_deterministic(s: Syllabus, c: Constraints | None = None, allowed_plos: list[str] | None = None,
                      n_chapters: int | None = None,
                      corpus_refs: set[str] | None = None) -> dict[str, list[Finding]]:
    c = c or Constraints()
    codes = [x.code for x in s.clos]
    return {
        "S1": check_weights(s.assessments),
        "S2": check_coverage(s.assessments, s.blueprint, codes),
        "S3": check_chapters(s.chapters, codes, n_chapters, c),
        "S4": check_plo_links(s.clos, allowed_plos),
        "S5": check_bilingual(s),
        "B1": check_bloom_verbs(s.clos),
        "B2": check_progression(s.clos, c),
        "B3": check_alignment(s.clos, s.blueprint),
        "G1": check_references(s, corpus_refs),
    }


def score_findings(fs: list[Finding]) -> float:
    """0 nếu có error; ngược lại trừ 0.25 cho mỗi warn (sàn 0). info không trừ."""
    if any(x.severity == "error" for x in fs):
        return 0.0
    return max(0.0, 1.0 - 0.25 * sum(x.severity == "warn" for x in fs))


def aggregate(det: dict[str, list[Finding]], llm: dict[str, dict] | None = None) -> dict:
    rows, num, den = [], 0.0, 0.0
    for cr in CRITERIA:
        if cr.kind == "det":
            fs = det.get(cr.id, [])
            sc = score_findings(fs)
            rows.append({"id": cr.id, "name": cr.name, "weight": cr.weight, "kind": "det",
                         "score": sc, "gate": cr.gate, "findings": [asdict(x) for x in fs]})
        else:
            if not llm or cr.id not in llm:
                continue
            raw = llm[cr.id]
            sc = (int(raw["score"]) - 1) / 3
            rows.append({"id": cr.id, "name": cr.name, "weight": cr.weight, "kind": "llm",
                         "score": sc, "gate": False, "judge": raw})
        num += sc * cr.weight
        den += cr.weight
    gate_fail = [r["id"] for r in rows if r["gate"] and
                 any(x["severity"] == "error" for x in r.get("findings", []))]
    return {
        "total": round(100 * num / den, 1) if den else 0.0,
        "partial": llm is None,
        "passed": not gate_fail,
        "gate_failed": gate_fail,
        "criteria": rows,
    }
