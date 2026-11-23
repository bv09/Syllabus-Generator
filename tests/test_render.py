"""Test render đề cương ra Word bằng template."""
from pathlib import Path
import pytest

TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "syllabus_template.docx"
SAMPLE = Path(__file__).resolve().parents[1] / "corpus" / "syllabus_IT3011_mau.docx"


@pytest.mark.skipif(not TEMPLATE.exists(), reason="chưa có template docx")
def test_render_template(tmp_path):
    from src.ingest_docx import parse_syllabus_docx
    from src.render import build_blueprint_rows
    from docxtpl import DocxTemplate

    syllabus = parse_syllabus_docx(SAMPLE)
    context = syllabus.model_dump()
    context["blueprint_rows"] = build_blueprint_rows(syllabus)
    context["version"] = "1.0"
    context["decision_no"] = "1234/QĐ"
    context["ects_text"] = " (3.25 ECTS)"
    context.pop("blueprint", None)

    out_file = tmp_path / "rendered.docx"
    doc = DocxTemplate(str(TEMPLATE))
    doc.render(context)
    doc.save(str(out_file))

    assert out_file.exists()
    assert out_file.stat().st_size > 5000
