"""Tests for schema.sql + db_manager.py (spec TEST 5, 7, 10 and constraint checks)."""
import json
import logging
import sqlite3
import tempfile
import unittest
from pathlib import Path

import db_manager as db
from obe_schemas import SyllabusSchema

ROOT = Path(__file__).resolve().parent.parent


def load_syllabus(name="sample_validated_output.json") -> SyllabusSchema:
    data = json.loads((ROOT / name).read_text(encoding="utf-8"))
    return SyllabusSchema.model_validate(data)


class TestDatabase(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "test.db"
        db.init_db(self.db_path)
        self.ai = load_syllabus()

    def tearDown(self):
        self._tmp.cleanup()
        logging.disable(logging.NOTSET)

    def count(self, table):
        with db.get_connection(self.db_path) as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def counts(self):
        return [self.count(t) for t in
                ("courses", "course_outcomes", "weekly_schedules", "lesson_outcomes")]

    # ---- TEST 5: a course can be inserted into SQLite
    def test_T5_course_inserted(self):
        cid = db.create_course(self.ai, db_path=self.db_path)
        course = db.get_course(cid, self.db_path)
        self.assertEqual(course["course_code"], self.ai.course.course_code)
        self.assertEqual(course["units"], 3)
        listed = db.list_courses(self.db_path)
        self.assertEqual(len(listed), 1)
        self.assertEqual((listed[0]["clo_count"], listed[0]["week_count"]), (5, 18))
        outcomes = db.get_course_outcomes(cid, self.db_path)
        self.assertEqual(outcomes[0]["text"], self.ai.course_outcomes[0].text)
        self.assertEqual(outcomes[0]["text"], outcomes[0]["original_text"])
        self.assertFalse(outcomes[0]["is_edited"])

    # ---- TEST 7: weekly schedules are persisted correctly
    def test_T7_weekly_schedule_persisted_in_order(self):
        cid = db.create_course(self.ai, db_path=self.db_path)
        schedule = db.get_weekly_schedule(cid, self.db_path)
        self.assertEqual([w["week"] for w in schedule], list(range(1, 19)))
        self.assertEqual(schedule[0]["topic"], self.ai.weekly_schedule[0].topic)
        for w in schedule:
            self.assertEqual([lo["domain"] for lo in w["lesson_outcomes"]], ["K", "S", "A"])
        self.assertEqual(self.count("lesson_outcomes"), 54)

    # ---- TEST 10: deleting a course handles related records
    def test_T10_delete_cascades_to_all_children(self):
        cid = db.create_course(self.ai, db_path=self.db_path)
        self.assertEqual(self.counts(), [1, 5, 18, 54])
        db.delete_course(cid, self.db_path)
        self.assertEqual(self.counts(), [0, 0, 0, 0])

    def test_T10b_delete_leaves_other_courses_untouched(self):
        ai_id = db.create_course(self.ai, db_path=self.db_path)
        db.create_course(load_syllabus("sample_dsa_validated_output.json"), db_path=self.db_path)
        db.delete_course(ai_id, self.db_path)
        self.assertEqual(self.counts(), [1, 5, 18, 54])

    def test_duplicate_course_code_rejected(self):
        db.create_course(self.ai, db_path=self.db_path)
        with self.assertRaises(db.DuplicateCourseError):
            db.create_course(self.ai, db_path=self.db_path)
        self.assertEqual(self.counts(), [1, 5, 18, 54])

    def test_replace_overwrites_without_duplicates(self):
        first = db.create_course(self.ai, db_path=self.db_path)
        second = db.create_course(self.ai, replace=True, db_path=self.db_path)
        self.assertNotEqual(first, second)
        self.assertEqual(self.counts(), [1, 5, 18, 54])

    def test_invalid_course_id(self):
        with self.assertRaises(db.CourseNotFoundError):
            db.get_course(999, self.db_path)
        with self.assertRaises(db.CourseNotFoundError):
            db.get_course_outcomes(999, self.db_path)
        with self.assertRaises(db.CourseNotFoundError):
            db.delete_course(999, self.db_path)

    # ---- second wall: the database itself rejects bad data
    def test_db_check_constraint_rejects_bad_domain(self):
        cid = db.create_course(self.ai, db_path=self.db_path)
        with self.assertRaises(db.DBError):
            with db.get_connection(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO course_outcomes (course_id, number, domain, text, original_text) "
                    "VALUES (?, 99, 'X', 'Bad domain outcome text', 'Bad domain outcome text')",
                    (cid,),
                )

    def test_foreign_keys_enforced(self):
        with self.assertRaises(db.DBError):
            with db.get_connection(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO course_outcomes (course_id, number, domain, text, original_text) "
                    "VALUES (999, 1, 'K', 'Orphan outcome text here', 'Orphan outcome text here')"
                )

    def test_create_course_requires_validated_schema(self):
        with self.assertRaises(db.DBError):
            db.create_course({"course": {}}, db_path=self.db_path)

    def test_failed_insert_rolls_back_everything(self):
        bad = self.ai.model_copy(deep=True)
        bad.weekly_schedule[5].lesson_outcomes[0].domain = "X"  # bypasses Pydantic on purpose
        with self.assertRaises(db.DBError):
            db.create_course(bad, db_path=self.db_path)
        self.assertEqual(self.counts(), [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()