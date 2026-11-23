"""Chạy rubric trên đề cương mẫu thật của trường (baseline)."""
from pathlib import Path
import pytest
import src.rubric
from src.ingest_docx import parse_syllabus_docx

SAMPLE = Path(__file__).resolve().parents[1] / "corpus" / "syllabus_IT3011_mau.docx"
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="thiếu đề cương mẫu")


def test_parse_sample():
    s = parse_syllabus_docx(SAMPLE)
    assert s.info.code == "IT3011" and len(s.clos) == 5 and len(s.chapters) == 9
    assert abs(sum(a.weight for a in s.assessments) - 100) < 1e-6


def test_sample_findings():
    s = parse_syllabus_docx(SAMPLE)
    det = src.rubric.run_deterministic(s, allowed_plos=["PLO 01", "PLO 02", "PLO 03", "PLO 04"], n_chapters=9)
    assert not [f for fs in det.values() for f in fs if f.severity == "error"]
    # phát hiện có chủ đích: 8/9 chương gắn đủ 5 CLO
    assert any(f.rule == "ch_mapping_flat" for f in det["S3"])
    assert any(f.rule == "bi_en_empty" for f in det["S5"])
