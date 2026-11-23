import argparse
import json
from pathlib import Path

from docxtpl import DocxTemplate
from pydantic import ValidationError

from schema import Syllabus


def build_blueprint_rows(syllabus: Syllabus) -> list[dict]:
    """Chuyển blueprint (danh sách ô theo CLO) thành các dòng ma trận cho template."""
    clos = syllabus.clos
    assessment_by_name = {a.name.vi: a for a in syllabus.assessments}

    rows_map: dict[str, dict] = {}
    order: list[str] = []

    for cell in syllabus.blueprint:
        component = cell.component
        if component not in rows_map:
            assessment = assessment_by_name.get(component)
            rows_map[component] = {
                "name_vi": component,
                "name_en": assessment.name.en if assessment is not None else "",
                "cells_by_clo": {},
            }
            order.append(component)

        methods = [m.vi for m in cell.methods]
        rows_map[component]["cells_by_clo"].setdefault(cell.clo_code, []).extend(methods)

    rows: list[dict] = []
    for component in order:
        row = rows_map[component]
        cells = [
            ", ".join(row["cells_by_clo"].get(clo.code, [])) for clo in clos
        ]
        rows.append(
            {
                "name_vi": row["name_vi"],
                "name_en": row["name_en"],
                "cells": cells,
            }
        )

    # Fallback: nếu không có blueprint, vẫn tạo dòng trống cho từng thành phần đánh giá.
    if not rows:
        for assessment in syllabus.assessments:
            rows.append(
                {
                    "name_vi": assessment.name.vi,
                    "name_en": assessment.name.en,
                    "cells": [""] * len(clos),
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render file JSON đề cương thành file .docx theo template."
    )
    parser.add_argument("json_path", help="Đường dẫn file JSON đầu vào")
    parser.add_argument(
        "template_path",
        nargs="?",
        default="templates/syllabus_template.docx",
        help="Đường dẫn template .docx",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        default="outputs/syllabus.docx",
        help="Đường dẫn file .docx đầu ra",
    )
    parser.add_argument("--version", default="1.0", help="Phiên bản đề cương")
    parser.add_argument(
        "--decision-no",
        default="............",
        help="Số quyết định phê duyệt/ban hành",
    )
    args = parser.parse_args()

    json_path = Path(args.json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))

    try:
        syllabus = Syllabus.model_validate(data)
    except ValidationError as e:
        raise SystemExit(f"JSON không khớp schema:\n{e}")

    context = syllabus.model_dump()
    context["blueprint_rows"] = build_blueprint_rows(syllabus)
    context["version"] = args.version
    context["ects_text"] = (
        f" ({syllabus.info.ects} ECTS)" if syllabus.info.ects is not None else ""
    )
    context["decision_no"] = args.decision_no
    context.pop("blueprint", None)

    template_path = Path(args.template_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = DocxTemplate(template_path)
    doc.render(context)
    doc.save(output_path)
    print(f"Đã tạo: {output_path}")


if __name__ == "__main__":
    main()
