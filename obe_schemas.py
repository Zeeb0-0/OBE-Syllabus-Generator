"""
obe_schemas.py
Pydantic schemas that define what a VALID OBE syllabus looks like.

The LLM is untrusted: nothing reaches SQLite unless it passes these models.
"""
import re
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EXPECTED_WEEKS = 18

# K = Knowledge, S = Skills, A = Attitudes
Domain = Literal["K", "S", "A"]

# Outcomes must not START with these vague, unmeasurable verbs.
VAGUE_VERBS = ("understand", "know", "learn", "be familiar with")

MIN_OUTCOME_WORDS = 4


def check_measurable(text: str) -> str:
    """Shared rule for every outcome text (CLOs and weekly lesson outcomes)."""
    cleaned = " ".join(text.split())  # collapse extra whitespace/newlines

    if len(cleaned.split()) < MIN_OUTCOME_WORDS:
        raise ValueError(
            f"Outcome is too short to be measurable (minimum {MIN_OUTCOME_WORDS} words): '{cleaned}'"
        )
    if not cleaned[0].isalpha():
        raise ValueError(f"Outcome must start with an action verb, got: '{cleaned}'")

    lowered = cleaned.lower()
    if lowered.startswith("to "):  # tolerate "To understand ..."
        lowered = lowered[3:]

    for phrase in VAGUE_VERBS:
        # \b = word boundary, so "knowledge" does NOT match "know"
        if re.match(rf"{re.escape(phrase)}\b", lowered):
            raise ValueError(
                f"Outcome starts with the vague verb '{phrase}'. "
                f"Use an active Bloom's verb (e.g. describe, apply, implement): '{cleaned}'"
            )
    return cleaned


class _Base(BaseModel):
    # Trim whitespace on every string field; ignore unexpected extra keys.
    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")


class CourseMetadataSchema(_Base):
    course_code: str = Field(min_length=2, max_length=20)
    course_title: str = Field(min_length=3, max_length=150)
    description: str = Field(min_length=20, max_length=2000)
    units: int = Field(ge=1, le=6)
    prerequisite: str = Field(default="None", max_length=200)


class CourseOutcomeSchema(_Base):
    number: int = Field(ge=1, le=20)
    domain: Domain
    text: str = Field(min_length=15, max_length=300)

    @field_validator("text")
    @classmethod
    def text_must_be_measurable(cls, v: str) -> str:
        return check_measurable(v)


class LessonOutcomeSchema(_Base):
    domain: Domain
    text: str = Field(min_length=15, max_length=300)

    @field_validator("text")
    @classmethod
    def text_must_be_measurable(cls, v: str) -> str:
        return check_measurable(v)


class WeeklyScheduleSchema(_Base):
    week: int = Field(ge=1, le=EXPECTED_WEEKS)
    topic: str = Field(min_length=3, max_length=150)
    description: str = Field(min_length=10, max_length=600)
    lesson_outcomes: List[LessonOutcomeSchema] = Field(min_length=3, max_length=9)

    @model_validator(mode="after")
    def must_cover_k_s_a(self):
        present = {lo.domain for lo in self.lesson_outcomes}
        missing = {"K", "S", "A"} - present
        if missing:
            raise ValueError(
                f"Week {self.week} is missing lesson outcomes for domain(s): {sorted(missing)}"
            )
        return self


class GeneratedContentSchema(_Base):
    """Exactly what Qwen is asked to produce (no institutional metadata)."""

    course_outcomes: List[CourseOutcomeSchema] = Field(min_length=3, max_length=8)
    weekly_schedule: List[WeeklyScheduleSchema] = Field(
        min_length=EXPECTED_WEEKS, max_length=EXPECTED_WEEKS
    )

    @model_validator(mode="after")
    def numbering_must_be_sequential(self):
        clo_numbers = [c.number for c in self.course_outcomes]
        if clo_numbers != list(range(1, len(clo_numbers) + 1)):
            raise ValueError(f"CLO numbers must run 1..n in order, got {clo_numbers}")

        weeks = [w.week for w in self.weekly_schedule]
        if weeks != list(range(1, EXPECTED_WEEKS + 1)):
            raise ValueError(f"Weeks must be 1..{EXPECTED_WEEKS} in order, got {weeks}")
        return self


class SyllabusSchema(GeneratedContentSchema):
    """Full syllabus = user-supplied course metadata + generated content."""

    course: CourseMetadataSchema


if __name__ == "__main__":
    # Quick manual check:  python obe_schemas.py [file.json]
    import json
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "sample_validated_output.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    s = SyllabusSchema.model_validate(data)
    n_lessons = sum(len(w.lesson_outcomes) for w in s.weekly_schedule)
    print(
        f"VALID: {path} -> '{s.course.course_title}' | "
        f"{len(s.course_outcomes)} CLOs | {len(s.weekly_schedule)} weeks | "
        f"{n_lessons} lesson outcomes"
    )