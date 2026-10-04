"""Tests for human-in-the-loop editing (spec TEST 6 and related checks)."""
import json
import logging
import tempfile
import unittest
from pathlib import Path

import db_manager as db
from obe_schemas import SyllabusSchema

ROOT = Path(__file__).resolve().parent.parent
SYLLABUS = SyllabusSchema.model_validate(
    json.loads((ROOT / "sample_validated_output.json").read_text(encoding="utf-8"))
)
EDITED_CLO = "Analyze fundamental concepts and applications of artificial intelligence."


class TestEditing(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "edit.db"
        db.init_db(self.db_path)
        self.course_id = db.create_course(SYLLABUS, db_path=self.db_path)
        self.clos = db.get_course_outcomes(self.course_id, self.db_path)
        self.weeks = db.get_weekly_schedule(self.course_id, self.db_path)

    def tearDown(self):
        self._tmp.cleanup()
        logging.disable(logging.NOTSET)

    # ---- TEST 6: course outcomes can be edited
    def test_T6_edit_clo_saved_and_original_preserved(self):
        clo = self.clos[0]
        db.update_course_outcome(clo["id"], EDITED_CLO, self.db_path)

        reread = db.get_course_outcomes(self.course_id, self.db_path)
        self.assertEqual(reread[0]["text"], EDITED_CLO)
        self.assertEqual(reread[0]["original_text"], clo["original_text"])
        self.assertTrue(reread[0]["is_edited"])
        self.assertFalse(reread[1]["is_edited"])  # other CLOs untouched

    def test_edit_lesson_outcome_saved(self):
        lesson = self.weeks[0]["lesson_outcomes"][1]
        new_text = "Implement uninformed search algorithms in Python for a maze problem."
        db.update_lesson_outcome(lesson["id"], new_text, self.db_path)

        reread = db.get_weekly_schedule(self.course_id, self.db_path)
        self.assertEqual(reread[0]["lesson_outcomes"][1]["text"], new_text)
        self.assertTrue(reread[0]["lesson_outcomes"][1]["is_edited"])
        self.assertFalse(reread[0]["lesson_outcomes"][0]["is_edited"])
        self.assertFalse(reread[1]["lesson_outcomes"][1]["is_edited"])

    def test_edit_rejects_vague_verb_and_changes_nothing(self):
        clo = self.clos[0]
        with self.assertRaises(db.InvalidEditError):
            db.update_course_outcome(
                clo["id"], "Understand the basics of artificial intelligence.", self.db_path
            )
        unchanged = db.get_course_outcomes(self.course_id, self.db_path)[0]
        self.assertEqual(unchanged["text"], clo["text"])
        self.assertFalse(unchanged["is_edited"])

    def test_edit_rejects_too_short_text(self):
        with self.assertRaises(db.InvalidEditError):
            db.update_lesson_outcome(self.weeks[0]["lesson_outcomes"][0]["id"], "Describe AI.", self.db_path)

    def test_edit_unknown_ids(self):
        with self.assertRaises(db.OutcomeNotFoundError):
            db.update_course_outcome(99999, EDITED_CLO, self.db_path)
        with self.assertRaises(db.OutcomeNotFoundError):
            db.update_lesson_outcome(99999, EDITED_CLO, self.db_path)
        with self.assertRaises(db.OutcomeNotFoundError):
            db.reset_course_outcome(99999, self.db_path)

    def test_edit_whitespace_is_normalized(self):
        messy = "   Analyze   fundamental concepts \n and applications of artificial intelligence.  "
        out = db.update_course_outcome(self.clos[0]["id"], messy, self.db_path)
        self.assertEqual(out["text"], EDITED_CLO)

    def test_reset_restores_ai_wording(self):
        clo = self.clos[0]
        lesson = self.weeks[0]["lesson_outcomes"][0]
        db.update_course_outcome(clo["id"], EDITED_CLO, self.db_path)
        db.update_lesson_outcome(
            lesson["id"], "Define the goals and major application areas of AI systems.", self.db_path
        )
        self.assertFalse(db.reset_course_outcome(clo["id"], self.db_path)["is_edited"])
        self.assertFalse(db.reset_lesson_outcome(lesson["id"], self.db_path)["is_edited"])
        self.assertEqual(db.get_course_outcomes(self.course_id, self.db_path)[0]["text"], clo["text"])

    def test_sql_injection_text_is_stored_as_plain_text(self):
        evil = "Describe things'; DROP TABLE courses; -- as plain text only."
        db.update_course_outcome(self.clos[0]["id"], evil, self.db_path)
        self.assertEqual(len(db.list_courses(self.db_path)), 1)  # table still exists
        self.assertEqual(db.get_course_outcomes(self.course_id, self.db_path)[0]["text"], evil)


if __name__ == "__main__":
    unittest.main()