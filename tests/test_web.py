"""Tests for the Flask web interface (main.create_app) using the Flask test client."""
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

import db_manager as db
from main import create_app

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = json.loads((ROOT / "sample_validated_output.json").read_text(encoding="utf-8"))
VALID_JSON = json.dumps(
    {"course_outcomes": SAMPLE["course_outcomes"], "weekly_schedule": SAMPLE["weekly_schedule"]}
)
EDITED_CLO = "Analyze fundamental concepts and applications of artificial intelligence."

FORM = {
    "course_code": "WEB-101",
    "course_title": "Web Interface Test Course",
    "description": "A course used only by the automated tests of the web interface.",
    "units": "3",
    "prerequisite": "None",
    "mode": "live",
}


def fake_llm(prompt, system):
    return VALID_JSON


def garbage_llm(prompt, system):
    return "this is not json"


class BlockingLLM:
    """Waits until released, so a job stays 'running' for the one-at-a-time test."""

    def __init__(self):
        self.release = threading.Event()

    def __call__(self, prompt, system):
        self.release.wait(timeout=20)
        return VALID_JSON


class WebTests(unittest.TestCase):
    def make_client(self, llm=fake_llm):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db_path = Path(tmp.name) / "web.db"
        self.out_dir = Path(tmp.name) / "output"
        app = create_app(db_path=self.db_path, output_dir=self.out_dir, live_generate_fn=llm)
        return app.test_client()

    @staticmethod
    def job_id(response):
        return response.headers["Location"].rstrip("/").rsplit("/", 1)[1]

    @staticmethod
    def wait_for_job(client, job_id, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = client.get(f"/api/jobs/{job_id}").get_json()
            if data["status"] != "running":
                return data
            time.sleep(0.05)
        raise AssertionError("the job did not finish in time")

    def generate_and_wait(self, client, **overrides):
        response = client.post("/generate", data={**FORM, **overrides})
        self.assertEqual(response.status_code, 302)
        return self.wait_for_job(client, self.job_id(response))

    # ---------------------------------------------------------------- pages
    def test_index_page_loads(self):
        client = self.make_client()
        response = client.get("/")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Generate syllabus", html)
        self.assertIn("Saved courses", html)
        self.assertIn("Artificial Intelligence", html)  # replay option

    def test_live_generation_end_to_end(self):
        client = self.make_client()
        job = self.generate_and_wait(client)
        self.assertEqual(job["status"], "done")
        self.assertTrue(job["course_url"])

        page = client.get(job["course_url"])
        html = page.get_data(as_text=True)
        self.assertEqual(page.status_code, 200)
        self.assertIn("Web Interface Test Course", html)
        self.assertIn(SAMPLE["course_outcomes"][0]["text"], html)
        self.assertIn("Week 18", html)
        self.assertTrue((self.out_dir / "last_live_generation.json").exists())

    def test_invalid_form_rejected_without_starting_job(self):
        client = self.make_client()
        response = client.post("/generate", data={**FORM, "description": "short"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("at least 20 characters", response.get_data(as_text=True))
        self.assertEqual(db.list_courses(self.db_path), [])

    def test_duplicate_course_rejected_then_replace_allowed(self):
        client = self.make_client()
        self.generate_and_wait(client)

        again = client.post("/generate", data=FORM)
        self.assertEqual(again.status_code, 400)
        self.assertIn("already exists", again.get_data(as_text=True))

        self.generate_and_wait(client, replace="on")
        self.assertEqual(len(db.list_courses(self.db_path)), 1)

    def test_replay_mode_saves_course(self):
        client = self.make_client()
        job = self.generate_and_wait(client, mode="replay", replay_key="sample_ai")
        self.assertEqual(job["status"], "done")
        courses = db.list_courses(self.db_path)
        self.assertEqual(courses[0]["course_title"], SAMPLE["course"]["course_title"])

    def test_failed_generation_reports_error_and_saves_nothing(self):
        client = self.make_client(llm=garbage_llm)
        job = self.generate_and_wait(client)
        self.assertEqual(job["status"], "error")
        self.assertIn("Nothing was saved", job["error"])
        self.assertEqual(db.list_courses(self.db_path), [])

    def test_only_one_generation_at_a_time(self):
        blocking = BlockingLLM()
        client = self.make_client(llm=blocking)
        first = client.post("/generate", data=FORM)
        self.assertEqual(first.status_code, 302)

        second = client.post("/generate", data={**FORM, "course_code": "WEB-102"})
        self.assertEqual(second.status_code, 400)
        self.assertIn("already running", second.get_data(as_text=True))

        blocking.release.set()
        job = self.wait_for_job(client, self.job_id(first))
        self.assertEqual(job["status"], "done")

    # -------------------------------------------------------------- editing
    def test_edit_save_export_shows_edited_clo(self):
        client = self.make_client()
        job = self.generate_and_wait(client)
        cid = db.list_courses(self.db_path)[0]["id"]
        clo = db.get_course_outcomes(cid, self.db_path)[0]

        edit = client.post(f"/courses/{cid}/clo/{clo['id']}", data={"text": EDITED_CLO, "action": "save"})
        self.assertEqual(edit.status_code, 302)
        self.assertIn(f"/courses/{cid}", edit.headers["Location"])
        self.assertIn(EDITED_CLO, client.get(job["course_url"]).get_data(as_text=True))

        export = client.post(f"/courses/{cid}/export", data={})
        self.assertEqual(export.status_code, 302)
        self.assertTrue(export.headers["Location"].endswith("/output/generated_syllabus.html"))

        final = client.get("/output/generated_syllabus.html")
        html = final.get_data(as_text=True)
        self.assertEqual(final.status_code, 200)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn(EDITED_CLO, html)               # the human edit is in the final syllabus
        self.assertNotIn(clo["original_text"], html)  # the AI wording is gone

    def test_rejected_edit_shows_message_and_changes_nothing(self):
        client = self.make_client()
        self.generate_and_wait(client)
        cid = db.list_courses(self.db_path)[0]["id"]
        clo = db.get_course_outcomes(cid, self.db_path)[0]

        response = client.post(
            f"/courses/{cid}/clo/{clo['id']}",
            data={"text": "Understand the basics of artificial intelligence.", "action": "save"},
            follow_redirects=True,
        )
        self.assertIn("vague verb", response.get_data(as_text=True))
        self.assertEqual(db.get_course_outcomes(cid, self.db_path)[0]["text"], clo["text"])

    def test_reset_restores_ai_wording(self):
        client = self.make_client()
        self.generate_and_wait(client)
        cid = db.list_courses(self.db_path)[0]["id"]
        clo = db.get_course_outcomes(cid, self.db_path)[0]

        client.post(f"/courses/{cid}/clo/{clo['id']}", data={"text": EDITED_CLO, "action": "save"})
        self.assertTrue(db.get_course_outcomes(cid, self.db_path)[0]["is_edited"])
        client.post(f"/courses/{cid}/clo/{clo['id']}", data={"action": "reset"})
        after = db.get_course_outcomes(cid, self.db_path)[0]
        self.assertFalse(after["is_edited"])
        self.assertEqual(after["text"], clo["text"])

    def test_delete_course(self):
        client = self.make_client()
        self.generate_and_wait(client)
        cid = db.list_courses(self.db_path)[0]["id"]
        response = client.post(f"/courses/{cid}/delete")
        self.assertEqual(response.status_code, 302)
        self.assertEqual(db.list_courses(self.db_path), [])

    # ------------------------------------------------------- error handling
    def test_unknown_course_shows_friendly_404(self):
        client = self.make_client()
        response = client.get("/courses/999")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 404)
        self.assertIn("No course with id 999", html)
        self.assertNotIn("Traceback", html)

    def test_output_route_blocks_path_traversal(self):
        client = self.make_client()
        self.assertEqual(client.get("/output/..%2Fmain.py").status_code, 404)
        self.assertEqual(client.get("/output/missing.html").status_code, 404)


if __name__ == "__main__":
    unittest.main()