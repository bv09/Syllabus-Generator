import json, sys
from pathlib import Path
from schema import CourseInfo
from generate import generate_syllabus

path = sys.argv[1] if len(sys.argv) > 1 else "inputs/it3011.json"
cfg = json.loads(Path(path).read_text(encoding="utf-8"))
info = CourseInfo(**cfg["info"])

s = generate_syllabus(info, cfg["plos"], cfg["num_chapters"], cfg.get("hint", ""))

Path("outputs").mkdir(exist_ok=True)
out = Path(f"outputs/{info.code}.json")
out.write_text(s.model_dump_json(indent=2), encoding="utf-8")
print("Đã ghi:", out)