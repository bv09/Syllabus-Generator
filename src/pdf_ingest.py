"""Nạp PDF vào corpus: sao chép, phát hiện chương, trích text, báo cáo.

Lệnh:  python src/main.py inputs/<MÃ>.json --pdfs <dir_or_files...>

Mỗi PDF được:
  1. Sao chép nguyên vẹn vào corpus/<MÃ>/books/ (giữ tên gốc).
  2. Trích text theo trang → phát hiện scan (trang có text rỗng).
  3. Phát hiện cấu trúc chương:
     a) Từ bookmark/outline nếu PDF có (ưu tiên).
     b) Nếu không có bookmark và mỗi PDF là 1 chương → tên file (bỏ đuôi) làm tiêu đề gợi ý.
  4. Trả danh sách gợi ý [{chapter_num, title, source_pdf}] — KHÔNG tự quyết,
     người dùng xác nhận qua inputs hoặc flag.
"""
from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

log = logging.getLogger("pdf_ingest")


@dataclass
class ChapterSuggestion:
    chapter_num: int
    title: str
    source_pdf: str
    page_start: int | None = None  # 1-based, None nếu không xác định


@dataclass
class PdfReport:
    filename: str
    num_pages: int
    is_scan: bool  # True nếu ≥50% trang text rỗng
    has_bookmarks: bool
    empty_pages: list[int] = field(default_factory=list)  # 1-based
    errors: list[str] = field(default_factory=list)


def _extract_text_pages(reader: PdfReader) -> list[str]:
    """Trích text mỗi trang, trả list[str] dài = số trang."""
    out: list[str] = []
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            out.append("")
    return out


def _outline_chapters(reader: PdfReader, source_pdf: str) -> list[ChapterSuggestion]:
    """Lấy chương từ outline/bookmark của PDF.

    Chỉ lấy mục cấp 1 (top-level) của outline. Mục lồng sâu hơn bị bỏ qua.
    """
    try:
        outline = reader.outline
    except Exception:  # noqa: BLE001
        return []
    if not outline:
        return []

    chapters: list[ChapterSuggestion] = []
    num = 0
    for item in outline:
        # outline là list; mục con nằm trong list lồng (list[list]) → bỏ qua
        if isinstance(item, list):
            continue
        try:
            title = str(item.title).strip()
        except AttributeError:
            continue
        if not title:
            continue
        # Thử lấy page number
        page_start = None
        try:
            page_idx = reader.get_destination_page_number(item)
            page_start = page_idx + 1  # 1-based
        except Exception:  # noqa: BLE001
            pass
        num += 1
        chapters.append(ChapterSuggestion(
            chapter_num=num, title=title, source_pdf=source_pdf,
            page_start=page_start))
    return chapters


def analyze_pdf(path: Path) -> tuple[PdfReport, list[ChapterSuggestion]]:
    """Phân tích 1 file PDF. Trả (báo cáo, danh sách chương gợi ý)."""
    report = PdfReport(filename=path.name, num_pages=0, is_scan=False, has_bookmarks=False)
    chapters: list[ChapterSuggestion] = []

    try:
        reader = PdfReader(str(path))
    except PdfReadError as e:
        report.errors.append(f"Không đọc được PDF: {e}")
        return report, chapters

    report.num_pages = len(reader.pages)

    # Text extraction & scan detection
    texts = _extract_text_pages(reader)
    for i, t in enumerate(texts, 1):
        if len(t.strip()) < 20:  # trang gần rỗng
            report.empty_pages.append(i)
    if report.num_pages > 0:
        report.is_scan = len(report.empty_pages) / report.num_pages >= 0.5

    # Bookmark detection
    chapters = _outline_chapters(reader, path.name)
    report.has_bookmarks = bool(chapters)

    return report, chapters


def ingest_pdfs(pdf_paths: list[Path], course_code: str,
                corpus_root: Path = Path("corpus")) -> tuple[list[PdfReport], list[ChapterSuggestion]]:
    """Nạp danh sách PDF vào corpus/<MÃ>/books/.

    Trả (danh sách báo cáo, danh sách gợi ý chương tổng hợp).
    Nếu không PDF nào có bookmark → gợi ý 1 PDF = 1 chương (dùng tên file).
    """
    dest = corpus_root / course_code / "books"
    dest.mkdir(parents=True, exist_ok=True)

    all_reports: list[PdfReport] = []
    all_chapters: list[ChapterSuggestion] = []
    any_bookmarks = False

    for src in pdf_paths:
        if not src.is_file() or src.suffix.lower() != ".pdf":
            log.warning("Bỏ qua %s (không phải file PDF)", src)
            continue
        # Sao chép vào corpus/<MÃ>/books/
        dst = dest / src.name
        if dst.resolve() != src.resolve():
            shutil.copy2(src, dst)
            log.info("Sao chép %s → %s", src.name, dst)

        report, chapters = analyze_pdf(src)
        all_reports.append(report)
        if chapters:
            any_bookmarks = True
            all_chapters.extend(chapters)

    # Nếu không PDF nào có bookmark → mỗi PDF = 1 chương, dùng tên file
    if not any_bookmarks and all_reports:
        for i, r in enumerate(all_reports, 1):
            if not r.errors:
                title = Path(r.filename).stem
                # Bỏ prefix kiểu "01_", "ch1_", "chapter1_"
                title = re.sub(r"^(?:ch(?:apter)?\.?\s*\d+[\s._-]*|(\d+)[\s._-]+)", "", title, flags=re.I)
                if not title:
                    title = Path(r.filename).stem
                all_chapters.append(ChapterSuggestion(
                    chapter_num=i, title=title, source_pdf=r.filename))

    return all_reports, all_chapters


def print_ingest_report(reports: list[PdfReport], chapters: list[ChapterSuggestion]) -> None:
    """In báo cáo nạp PDF ra stdout."""
    print(f"\n{'='*60}")
    print(f"BÁO CÁO NẠP PDF ({len(reports)} file)")
    print(f"{'='*60}")
    for r in reports:
        flags = []
        if r.is_scan:
            flags.append("⚠ SCAN (text rỗng)")
        if r.errors:
            flags.append(f"⚠ LỖI: {'; '.join(r.errors)}")
        status = " | ".join(flags) if flags else "OK"
        bm = "có bookmark" if r.has_bookmarks else "không có bookmark"
        print(f"  {r.filename}: {r.num_pages} trang, {bm} — {status}")
        if r.empty_pages and not r.is_scan:
            print(f"    trang text rỗng: {r.empty_pages[:10]}{'...' if len(r.empty_pages) > 10 else ''}")

    if chapters:
        print(f"\nGỢI Ý CẤU TRÚC CHƯƠNG ({len(chapters)} chương):")
        for ch in chapters:
            pg = f", trang {ch.page_start}" if ch.page_start else ""
            print(f"  {ch.chapter_num}. {ch.title} [{ch.source_pdf}{pg}]")
        print("\n⚠ ĐÂY LÀ GỢI Ý — người dùng cần xác nhận qua inputs hoặc flag.")
    else:
        print("\n⚠ Không xác định được cấu trúc chương từ PDF.")
        print("  Người dùng cần chỉ định thủ công.")


def resolve_pdf_paths(args: list[str]) -> list[Path]:
    """Chuyển danh sách đường dẫn (file hoặc thư mục) thành list[Path] các file PDF."""
    out: list[Path] = []
    for a in args:
        p = Path(a)
        if p.is_dir():
            out.extend(sorted(p.glob("*.pdf"), key=lambda x: x.name))
        elif p.is_file() and p.suffix.lower() == ".pdf":
            out.append(p)
        else:
            log.warning("Bỏ qua %s", a)
    return out
