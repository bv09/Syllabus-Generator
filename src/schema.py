from pydantic import BaseModel, Field


class CLO(BaseModel):
    code: str = Field(description="Mã chuẩn đầu ra, ví dụ CLO1")
    description: str = Field(description="Năng lực người học đạt được")


class WeekContent(BaseModel):
    week: int = Field(ge=1)
    topic: str
    content: str
    hours: float | None = None
    learning_activities: list[str] = Field(default_factory=list)
    clo_codes: list[str] = Field(default_factory=list)


class Assessment(BaseModel):
    component: str
    weight: float = Field(ge=0, le=100)
    method: str | None = None
    description: str | None = None
    clo_codes: list[str] = Field(default_factory=list)


class Reference(BaseModel):
    title: str
    author: str | None = None
    year: int | None = None
    source_pages: list[int] = Field(default_factory=list)


class Syllabus(BaseModel):
    course_name: str
    course_code: str | None = None
    credits: int | None = Field(default=None, ge=1)
    description: str
    objectives: list[str] = Field(min_length=1)
    clos: list[CLO] = Field(min_length=1)
    weekly_plan: list[WeekContent] = Field(min_length=1)
    assessments: list[Assessment] = Field(min_length=1)
    references: list[Reference] = Field(min_length=1)