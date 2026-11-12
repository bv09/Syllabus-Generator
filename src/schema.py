from typing import Literal
from pydantic import BaseModel, Field


class Bi(BaseModel):
    """Văn bản song ngữ."""
    vi: str
    en: str


class Workload(BaseModel):
    lecture_periods: float = Field(description="Số tiết lý thuyết")
    exercise_periods: float = Field(description="Số tiết bài tập")
    lab_periods: float = Field(description="Số tiết thí nghiệm")
    self_study_periods: float = Field(description="Số giờ tự học")


class CourseInfo(BaseModel):
    title: Bi
    code: str
    credits: float
    ects: float | None = None
    workload: Workload
    teaching_methods: list[Bi]           # vd: Offline, Blended learning
    relation_to_curriculum: Bi           # Bắt buộc / Tự chọn
    prerequisites: list[str]             # mã học phần
    language_of_instruction: str | None = None
    semester: int | None = None
    lecturers: list[str]
    max_group_size: int | None = None


class CLO(BaseModel):
    code: str                            # CLO1, CLO2...
    description: Bi
    linked_plos: list[str]               # ["PLO02", "PLO03"]
    level: Literal["I", "R", "M"]        # Introduced / Reinforced / Mastered


class AssessmentComponent(BaseModel):
    name: Bi
    weight: float = Field(description="Tỉ trọng %, tổng các thành phần = 100")
    timing: Bi                           # vd: tuần 17-19


class BlueprintCell(BaseModel):
    component: str                       # tên thành phần (khớp AssessmentComponent.name.vi)
    clo_code: str
    methods: list[Bi]                    # Kiểm tra viết, Trắc nghiệm, Vấn đáp...


class Chapter(BaseModel):
    number: int
    title: Bi
    topics: list[Bi]                     # các mục con
    clo_codes: list[str]
    activities: Bi                       # Giảng bài; bài tập; thảo luận


class Reference(BaseModel):
    citation: str                        # tác giả, tên sách, NXB, năm


class Syllabus(BaseModel):
    info: CourseInfo
    description: Bi
    clos: list[CLO] = Field(min_length=1)
    assessments: list[AssessmentComponent] = Field(min_length=1)
    blueprint: list[BlueprintCell]
    chapters: list[Chapter] = Field(min_length=1)
    required_readings: list[Reference]
    recommended_readings: list[Reference]