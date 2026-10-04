"""End-to-end integration tests:
LLM -> JSON -> Pydantic -> SQLite -> Human edit -> Jinja2 -> HTML
(the LLM is a scripted stand-in so the tests are fast and repeatable)."""
import json
import logging
import sqlite3
import tempfile
import time
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import requests
from markupsafe import escape

import db_manager as db
from export_engine import export_syllabus
from main import create_app
from obe_schemas import SyllabusSchema

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = json.loads((ROOT / "sample_validated_output.json").read_text(encoding="utf-8"))
VALID_JSON = json.dumps(
    {"course_outcomes": SAMPLE["course_outcomes"], "weekly_schedule": SAMPLE["weekly_schedule"]}
)
EDITED_CLO = "Analyze fundamental concepts and applications of artificial intelligence."
FORM = {
    "course_code": "INT-101",
    "course_title": "Integration Test Course",
    "description": "A course used only by the integration tests of the whole pipeline.",
    "units": "3",
    "prerequisite": "None",
    "mode": "live",
}


def esc(text: str) -> str:
    return str(escape(text))


class ScriptedLLM:
    """Stand-in for Qwen: returns scripted answers in order and counts the calls."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0
        self.prompts = []

    def __call__(self, prompt, system):
        self.calls += 1
        self.prompts.append(prompt)
        return self.answers.pop(0)


class TestIntegration(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.NOTSET)  # the job log needs INFO records
        obe_logger = logging.getLogger("obe")
        previous_level = obe_logger.level
        obe_logger.setLevel(logging.INFO)
        self.addCleanup(obe_logger.setLevel, previous_level)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp_dir = Path(tmp.name)
        self.db_path = self.tmp_dir / "integration.db"
        self.out_dir = self.tmp_dir / "output"
        db.init_db(self.db_path)

    # ------------------------------------------------------------- helpers
    def make_client(self, llm):
        app = create_app(db_path=self.db_path, output_dir=self.out_dir, live_generate_fn=llm)
        return app.test_client()

    @staticmethod
    def wait_for_job(client, job_id, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = client.get(f"/api/jobs/{job_id}").get_json()
            if data["status"] != "running":
                return data
            time.sleep(0.05)
        raise AssertionError("the job did not finish in time")

    def generate_and_wait(self, client):
        response = client.post("/generate", data=FORM)
        self.assertEqual(response.status_code, 302)
        job_id = response.headers["Location"].rstrip("/").rsplit("/", 1)[1]
        return self.wait_for_job(client, job_id)

    # ------------------------------------------------ the 13-step demo flow
    def test_full_demo_flow_through_the_web_ui(self):
        llm = ScriptedLLM(["{this is malformed json", VALID_JSON])
        client = self.make_client(llm)

        # Steps 3-5: submit the course; Qwen's first answer is rejected, the retry passes
        job = self.generate_and_wait(client)
        self.assertEqual(job["status"], "done")
        log_text = "\n".join(job["log"])
        self.assertIn("json_error", log_text)          # attempt 1 rejected
        self.assertIn("passed validation", log_text)   # attempt 2 accepted
        self.assertEqual(llm.calls, 2)
        self.assertIn("REJECTED", llm.prompts[1])      # the retry prompt carried feedback

        # Steps 6-7: the validated course is stored in SQLite
        courses = db.list_courses(self.db_path)
        self.assertEqual(len(courses), 1)
        cid = courses[0]["id"]
        self.assertEqual((courses[0]["clo_count"], courses[0]["week_count"]), (5, 18))
        schedule = db.get_weekly_schedule(cid, self.db_path)
        self.assertEqual(sum(len(w["lesson_outcomes"]) for w in schedule), 54)

        # Step 8: open the generated outcomes
        clo = db.get_course_outcomes(cid, self.db_path)[0]
        page = client.get(job["course_url"]).get_data(as_text=True)
        self.assertIn(esc(clo["text"]), page)

        # Steps 9-10: manually edit one CLO and save it
        client.post(f"/courses/{cid}/clo/{clo['id']}", data={"text": EDITED_CLO, "action": "save"})
        saved = db.get_course_outcomes(cid, self.db_path)[0]
        self.assertEqual(saved["text"], EDITED_CLO)
        self.assertEqual(saved["original_text"], clo["original_text"])

        # Steps 11-13: export with Jinja2, open the HTML, the edit is in the final syllabus
        export = client.post(f"/courses/{cid}/export", data={})
        self.assertEqual(export.status_code, 302)
        final = client.get("/output/generated_syllabus.html")
        html = final.get_data(as_text=True)
        self.assertEqual(final.status_code, 200)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn(EDITED_CLO, html)
        self.assertNotIn(esc(clo["original_text"]), html)
        for week in schedule:
            self.assertIn(esc(week["topic"]), html)
        self.assertEqual(html, (self.out_dir / "generated_syllabus.html").read_text(encoding="utf-8"))

    # --------------------------------------------- database design (spec 4)
    def test_database_is_normalized_not_a_json_blob(self):
        client = self.make_client(ScriptedLLM([VALID_JSON]))
        self.assertEqual(self.generate_and_wait(client)["status"], "done")

        expected_tables = {"courses", "course_outcomes", "weekly_schedules", "lesson_outcomes"}
        with closing(sqlite3.connect(self.db_path)) as conn:
            tables = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            self.assertEqual(tables, expected_tables)

            for table in expected_tables:
                for row in conn.execute(f"SELECT * FROM {table}"):
                    for value in row:
                        if isinstance(value, str):
                            self.assertFalse(
                                value.lstrip().startswith(("{", "[")),
                                f"{table} appears to store JSON text",
                            )

            fks = {
                table: [(r[2], r[6]) for r in conn.execute(f"PRAGMA foreign_key_list({table})")]
                for table in expected_tables
            }
        self.assertEqual(fks["courses"], [])
        self.assertEqual(fks["course_outcomes"], [("courses", "CASCADE")])
        self.assertEqual(fks["weekly_schedules"], [("courses", "CASCADE")])
        self.assertEqual(fks["lesson_outcomes"], [("weekly_schedules", "CASCADE")])

    # ----------------------------------------------------- both sample data
    def test_all_sample_courses_roundtrip_through_db_and_export(self):
        for name in ("sample_validated_output.json", "sample_dsa_validated_output.json"):
            with self.subTest(sample=name):
                syllabus = SyllabusSchema.model_validate(
                    json.loads((ROOT / name).read_text(encoding="utf-8"))
                )
                cid = db.create_course(syllabus, db_path=self.db_path)
                self.assertEqual(len(db.get_weekly_schedule(cid, self.db_path)), 18)
                path = export_syllabus(cid, self.tmp_dir / f"sample_{cid}.html", db_path=self.db_path)
                html = path.read_text(encoding="utf-8")
                self.assertIn(esc(syllabus.course.course_title), html)
                self.assertIn(esc(syllabus.weekly_schedule[17].topic), html)

    # -------------------------------------------- error handling (spec 11)
    @patch("llm_engine.requests.get", side_effect=requests.exceptions.ConnectionError())
    def test_ollama_down_gives_friendly_message_in_ui(self, _get):
        client = create_app(db_path=self.db_path, output_dir=self.out_dir).test_client()  # real engine path

        home = client.get("/")
        self.assertIn("pill-bad", home.get_data(as_text=True))

        response = client.post("/generate", data=FORM)
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Cannot reach Ollama", html)
        self.assertNotIn("Traceback", html)
        self.assertEqual(db.list_courses(self.db_path), [])


if __name__ == "__main__":
    unittest.main()