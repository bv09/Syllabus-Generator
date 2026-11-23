"""RAG tối giản, không phụ thuộc thư viện ngoài.

Tinh thần Lewis et al. (2020): sinh có điều kiện trên đoạn văn truy hồi từ bộ nhớ ngoài
(non-parametric). KHÁC bài báo: bài báo huấn luyện đồng thời retriever (DPR) và generator (BART);
ở đây dùng LLM có sẵn + retrieval lúc suy luận (inference-time RAG), không fine-tune.

Quy ước tên file trong corpus/ (đoán 'kind' từ tiền tố):
  ref*.txt|md   mỗi dòng không rỗng = MỘT tài liệu tham khảo (kind=reference). Đây là nguồn DUY NHẤT
                cho mục tài liệu học tập.
  plo*          chuẩn đầu ra CTĐT        toc*  mục lục giáo trình      reg*  quy chế/biểu mẫu
  *.docx|*.json đề cương cũ (parse thành chunk theo CLO/chương, kind=syllabus)
  khác          kind=general
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REFERENCE, SYLLABUS, GENERAL = "reference", "syllabus", "general"


@dataclass
class Chunk:
    id: str
    text: str
    kind: str
    source: str
    course: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Hit:
    chunk: Chunk
    score: float


_TOK = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Âm tiết + bigram: xấp xỉ từ ghép tiếng Việt khi không có bộ tách từ."""
    t = _TOK.findall(unicodedata.normalize("NFC", text).lower())
    return t + [f"{a}_{b}" for a, b in zip(t, t[1:])]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(docs)) if docs else 0.0
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, q: list[str], i: int) -> float:
        tf, s = self.tf[i], 0.0
        for t in set(q):
            f = tf.get(t, 0)
            if f:
                s += self.idf[t] * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.len[i] / (self.avg or 1)))
        return s


def _cos(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


class Retriever:
    def __init__(self, chunks: list[Chunk],
                 embed_fn: Callable[[list[str]], list[list[float]]] | None = None):
        self.chunks = chunks
        self.embed_fn = embed_fn
        self._bm = BM25([tokenize(c.text) for c in chunks])
        self._vec: list[list[float]] | None = None

    def _dense(self, q: str, idx: list[int]) -> list[int]:
        if self._vec is None:
            self._vec = self.embed_fn([c.text for c in self.chunks])
        qv = self.embed_fn([q])[0]
        return sorted(idx, key=lambda i: _cos(qv, self._vec[i]), reverse=True)

    def search(self, query: str, k: int = 5, kinds: set[str] | None = None,
               exclude_course: str | None = None) -> list[Hit]:
        idx = [i for i, c in enumerate(self.chunks)
               if (kinds is None or c.kind in kinds)
               and not (exclude_course and c.course and c.course.upper() == exclude_course.upper())]
        if not idx:
            return []
        qt = tokenize(query)
        bm = {i: self._bm.score(qt, i) for i in idx}
        rank_bm = [i for i in sorted(idx, key=lambda i: bm[i], reverse=True) if bm[i] > 0]
        fused: Counter = Counter()
        for r, i in enumerate(rank_bm):               # Reciprocal Rank Fusion (k=60)
            fused[i] += 1 / (60 + r)
        if self.embed_fn:
            for r, i in enumerate(self._dense(query, idx)):
                fused[i] += 1 / (60 + r)
        top = sorted(fused, key=lambda i: fused[i], reverse=True)[:k]
        return [Hit(self.chunks[i], float(fused[i])) for i in top]


# --- Nạp corpus ----------------------------------------------------------------------
_DIR_KIND = {"ref": REFERENCE, "plo": "plo", "toc": "toc", "reg": "regulation",
             "books": GENERAL}
_PREFIX_KIND = [("ref", REFERENCE), ("plo", "plo"), ("toc", "toc"), ("reg", "regulation"),
                ("syllabus", SYLLABUS)]


def _kind_from_name(stem: str) -> str:
    s = stem.lower()
    for pre, kind in _PREFIX_KIND:
        if s.startswith(pre):
            return kind
    return GENERAL


def _kind_from_dir(dir_name: str) -> str | None:
    """Đoán kind từ tên thư mục con (corpus/<MÃ>/<dir>/). Trả None nếu không khớp."""
    return _DIR_KIND.get(dir_name.lower())


def _window(text: str, max_words: int = 120, overlap: int = 20) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, cur = [], []
    for p in paras:
        w = p.split()
        if cur and len(cur) + len(w) > max_words:
            out.append(" ".join(cur))
            cur = cur[-overlap:]
        cur += w
        while len(cur) > max_words * 2:               # đoạn quá dài
            out.append(" ".join(cur[:max_words]))
            cur = cur[max_words - overlap:]
    if cur:
        out.append(" ".join(cur))
    return out


def _syllabus_chunks(s, source: str) -> list[Chunk]:
    code, stem, out = s.info.code, Path(source).stem, []
    n = 0

    def add(text: str, kind: str, **meta):
        nonlocal n
        out.append(Chunk(f"{stem}:{n}", text, kind, source, code, meta))
        n += 1

    add(f"{s.info.title.vi} ({s.info.title.en}). {s.description.vi}", SYLLABUS, part="description")
    for c in s.clos:
        add(f"{c.code}: {c.description.vi} | {c.description.en}", SYLLABUS, part="clo")
    for ch in s.chapters:
        add(f"Chương {ch.number}. {ch.title.vi} ({ch.title.en}): "
            + "; ".join(t.vi for t in ch.topics), SYLLABUS, part="chapter")
    for r in [*s.required_readings, *s.recommended_readings]:
        add(r.citation, REFERENCE, part="reference")
    return out


def _detect_chapter_num(stem: str) -> int | None:
    """Rút trích số chương từ tên file (vd: ch1_intro -> 1, 02_sorting -> 2)."""
    m = re.search(r"(?:ch(?:apter)?\.?|chuong|bài|bai)[\s._-]*(\d+)", stem, re.I)
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d+)[\s._-]", stem)
    if m:
        return int(m.group(1))
    return None


def _detect_course(path: Path, corpus_root: Path) -> str | None:
    """Suy ra mã môn từ đường dẫn: corpus/<MÃ>/... → MÃ, trừ _shared và _syllabus."""
    try:
        rel = path.relative_to(corpus_root)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) >= 2 and not parts[0].startswith("_"):
        return parts[0]
    return None


def load_corpus(directory: str | Path) -> list[Chunk]:
    from schema import Syllabus  # import muộn để tránh vòng lặp

    chunks: list[Chunk] = []
    corpus_root = Path(directory)
    for p in sorted(corpus_root.rglob("*")):
        if not p.is_file() or p.name.startswith("~$"):
            continue
        ext, stem = p.suffix.lower(), p.stem

        # Đoán kind: ưu tiên tên thư mục con, rồi tiền tố tên file
        dir_kind = None
        try:
            rel = p.relative_to(corpus_root)
            for part in rel.parts[:-1]:  # các thư mục cha (trừ tên file)
                dk = _kind_from_dir(part)
                if dk is not None:
                    dir_kind = dk
                    break
        except ValueError:
            pass

        # Suy ra course code
        course = _detect_course(p, corpus_root)

        if ext in (".txt", ".md"):
            text = p.read_text(encoding="utf-8")
            kind = dir_kind if dir_kind is not None else _kind_from_name(stem)
            if kind == REFERENCE:
                lines = [x.strip(" -*\t") for x in text.splitlines()
                         if x.strip() and not x.lstrip().startswith("#")]
                chunks += [Chunk(f"{stem}:{i}", ln, REFERENCE, p.name, course=course)
                           for i, ln in enumerate(lines)]
            else:
                chunks += [Chunk(f"{stem}:{i}", w, kind, p.name, course=course)
                           for i, w in enumerate(_window(text))]
        elif ext == ".pdf":
            # Chunk PDF (text extraction)
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(p))
                full_text = ""
                for page in reader.pages:
                    try:
                        full_text += (page.extract_text() or "") + "\n"
                    except Exception:  # noqa: BLE001
                        pass
                kind = dir_kind if dir_kind is not None else GENERAL
                ch_num = _detect_chapter_num(stem)
                meta = {"source_pdf": p.name}
                if ch_num is not None:
                    meta["chapter"] = ch_num
                for i, w in enumerate(_window(full_text)):
                    chunks.append(Chunk(f"{stem}:{i}", w, kind, p.name, course=course,
                                        meta={**meta}))
            except Exception as e:  # noqa: BLE001
                print(f"[rag] bỏ qua PDF {p.name}: {e}")
        elif ext == ".docx":
            from ingest_docx import parse_syllabus_docx
            try:
                chunks += _syllabus_chunks(parse_syllabus_docx(p), p.name)
            except Exception as e:  # noqa: BLE001
                print(f"[rag] bỏ qua {p.name}: {e}")
        elif ext == ".json":
            try:
                chunks += _syllabus_chunks(Syllabus.model_validate(json.loads(p.read_text("utf-8"))), p.name)
            except Exception:  # noqa: BLE001
                pass
    return chunks


def build_retriever(directory: str | Path | None, use_dense: bool = True) -> Retriever | None:
    if not directory or not Path(directory).exists():
        return None
    chunks = load_corpus(directory)
    if not chunks:
        return None
    embed_fn = None
    if use_dense:
        import llm
        if llm.has_embeddings():
            embed_fn = llm.embed
    return Retriever(chunks, embed_fn)


def reference_texts(r: Retriever | None) -> set[str] | None:
    return None if r is None else {c.text for c in r.chunks if c.kind == REFERENCE}
