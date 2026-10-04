"""Tests for obe_schemas.py (covers spec TEST 1 and TEST 2, plus extra rules)."""
import copy
import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from obe_schemas import CourseOutcomeSchema, SyllabusSchema

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FILES = ["sample_validated_output.json", "sample_dsa_validated_output.json"]


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


class TestSchemas(unittest.TestCase):
    def setUp(self):
        self.data = load("sample_validated_output.json")

    def test_T1_valid_samples_pass(self):
        for name in SAMPLE_FILES:
            with self.subTest(file=name):
                SyllabusSchema.model_validate(load(name))

    def test_T2_invalid_course_domain_fails(self):
        self.data["course_outcomes"][0]["domain"] = "X"
        with self.assertRaises(ValidationError):
            SyllabusSchema.model_validate(self.data)

    def test_T2b_invalid_lesson_domain_fails(self):
        self.data["weekly_schedule"][0]["lesson_outcomes"][0]["domain"] = "X"
        with self.assertRaises(ValidationError):
            SyllabusSchema.model_validate(self.data)

    def test_vague_verb_rejected(self):
        self.data["course_outcomes"][0]["text"] = "Understand the basics of artificial intelligence."
        with self.assertRaises(ValidationError):
            SyllabusSchema.model_validate(self.data)

    def test_vague_verb_lookalike_allowed(self):
        # "knowledge" contains "know" but is NOT the vague verb
        CourseOutcomeSchema(
            number=1, domain="K",
            text="Apply knowledge of search algorithms to solve routing problems.",
        )

    def test_17_weeks_rejected(self):
        self.data["weekly_schedule"].pop()
        with self.assertRaises(ValidationError):
            SyllabusSchema.model_validate(self.data)

    def test_week_missing_skill_domain_rejected(self):
        week = self.data["weekly_schedule"][2]
        week["lesson_outcomes"] = [lo for lo in week["lesson_outcomes"] if lo["domain"] != "S"]
        with self.assertRaises(ValidationError):
            SyllabusSchema.model_validate(self.data)

    def test_weeks_out_of_order_rejected(self):
        self.data["weekly_schedule"][0]["week"] = 2
        self.data["weekly_schedule"][1]["week"] = 1
        with self.assertRaises(ValidationError):
            SyllabusSchema.model_validate(self.data)

    def test_units_out_of_range_rejected(self):
        self.data["course"]["units"] = 0
        with self.assertRaises(ValidationError):
            SyllabusSchema.model_validate(self.data)


if __name__ == "__main__":
    unittest.main()