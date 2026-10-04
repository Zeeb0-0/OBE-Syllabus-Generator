"""Tests for the controller in main.py: validated data reaches SQLite, bad data does not."""
import json
import logging
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

import db_manager as db
from llm_engine import GenerationFailedError
from main import make_replay_fn, run_pipeline

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_FILE = ROOT / "sample_validated_output.json"
SAMPLE = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
COURSE_INFO = SAMPLE["course"]
VALID_JSON = json.dumps(
    {"course_outcomes": SAMPLE["course_outcomes"], "weekly_schedule": SAMPLE["weekly_schedule"]}
)


class ScriptedLLM:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def __call__(self, prompt, system):
        self.calls += 1
        return self.answers.pop(0)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "pipeline.db"

    def tearDown(self):
        self._tmp.cleanup()
        logging.disable(logging.NOTSET)

    def test_valid_output_is_saved_to_sqlite(self):
        llm = ScriptedLLM([VALID_JSON])
        result = run_pipeline(COURSE_INFO, generate_fn=llm, db_path=self.db_path)
        self.assertEqual(result.generation.attempts, 1)
        schedule = db.get_weekly_schedule(result.course_id, self.db_path)
        self.assertEqual(len(schedule), 18)
        self.assertEqual(len(db.get_course_outcomes(result.course_id, self.db_path)), 5)

    def test_retry_then_save_only_once(self):
        llm = ScriptedLLM(["{broken json", VALID_JSON])
        result = run_pipeline(COURSE_INFO, generate_fn=llm, db_path=self.db_path)
        self.assertEqual(result.generation.attempts, 2)
        self.assertEqual(len(db.list_courses(self.db_path)), 1)

    def test_invalid_output_never_reaches_database(self):
        llm = ScriptedLLM(["bad", "{still bad", "nope"])
        with self.assertRaises(GenerationFailedError):
            run_pipeline(COURSE_INFO, generate_fn=llm, db_path=self.db_path)
        self.assertEqual(db.list_courses(self.db_path), [])

    def test_invalid_form_data_rejected_before_llm_call(self):
        llm = ScriptedLLM([VALID_JSON])
        bad_info = dict(COURSE_INFO, description="too short")
        with self.assertRaises(ValidationError):
            run_pipeline(bad_info, generate_fn=llm, db_path=self.db_path)
        self.assertEqual(llm.calls, 0)

    def test_duplicate_detected_before_llm_call(self):
        run_pipeline(COURSE_INFO, generate_fn=ScriptedLLM([VALID_JSON]), db_path=self.db_path)
        llm = ScriptedLLM([VALID_JSON])
        with self.assertRaises(db.DuplicateCourseError):
            run_pipeline(COURSE_INFO, generate_fn=llm, db_path=self.db_path)
        self.assertEqual(llm.calls, 0)
        # with replace=True it succeeds and there is still exactly one course
        run_pipeline(COURSE_INFO, replace=True, generate_fn=llm, db_path=self.db_path)
        self.assertEqual(len(db.list_courses(self.db_path)), 1)

    def test_stage_callback_order(self):
        stages = []
        run_pipeline(COURSE_INFO, generate_fn=ScriptedLLM([VALID_JSON]),
                     db_path=self.db_path, on_stage=lambda s, m: stages.append(s))
        self.assertEqual(stages, ["init", "generate", "validated", "saved"])

    def test_replay_runs_validation_and_saves(self):
        # the sample file also contains a "course" key; the generated-content
        # schema ignores it, so replaying a full saved file works
        result = run_pipeline(COURSE_INFO, generate_fn=make_replay_fn(SAMPLE_FILE),
                              db_path=self.db_path)
        self.assertEqual(result.generation.attempts, 1)
        self.assertEqual(len(db.list_courses(self.db_path)), 1)


if __name__ == "__main__":
    unittest.main()