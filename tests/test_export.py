"""Tests for export_engine.py (spec TEST 8 and TEST 9, plus error handling)."""
import json
import logging
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

from markupsafe import escape

import db_manager as db
from export_engine import (
    ExportError,
    ExportExtras,
    export_syllabus,
    load_extras,
    render_syllabus,
)
from obe_schemas import SyllabusSchema

ROOT = Path(__file__).resolve().parent.parent
SYLLABUS = SyllabusSchema.model_validate(
    json.loads((ROOT / "sample_validated_output.json").read_text(encoding="utf-8"))
)
EDITED_CLO = "Analyze fundamental concepts and applications of artificial intelligence."


def esc(text: str) -> str:
    """What Jinja's autoescape turns a string into."""
    return str(escape(text))


class TagBalance(HTMLParser):
    """Tiny well-formedness check: every opened tag must be closed in order."""

    VOID = {"meta", "br", "hr", "img", "link", "input"}

    def __init__(self):
        super().__init__()
        self.stack, self.errors = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> (open: {self.stack[-3:]})")
        else:
            self.stack.pop()


class TestExport(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "export.db"
        db.init_db(self.db_path)
        self.cid = db.create_course(SYLLABUS, db_path=self.db_path)

    def tearDown(self):
        self._tmp.cleanup()
        logging.disable(logging.NOTSET)

    def render(self, **kw):
        return render_syllabus(self.cid, db_path=self.db_path, **kw)

    # ---- TEST 8: Jinja2 generates valid HTML
    def test_T8_html_is_well_formed(self):
        html = self.render()
        self.assertTrue(html.lstrip().startswith("<!DOCTYPE html>"))
        checker = TagBalance()
        checker.feed(html)
        checker.close()
        self.assertEqual(checker.errors, [])
        self.assertEqual(checker.stack, [])
        self.assertIn("<title>", html)

    def test_T8b_all_database_content_is_rendered(self):
        html = self.render()
        data = db.get_full_syllabus(self.cid, self.db_path)
        self.assertIn(esc(data["course"]["course_title"]), html)
        self.assertIn(esc(data["course"]["description"]), html)
        for clo in data["course_outcomes"]:
            self.assertIn(esc(clo["text"]), html)
        self.assertEqual(len(data["weekly_schedule"]), 18)
        for week in data["weekly_schedule"]:
            self.assertIn(esc(week["topic"]), html)
            for lo in week["lesson_outcomes"]:
                self.assertIn(esc(lo["text"]), html)
        self.assertIn("not the official", html)  # draft banner is shown

    # ---- TEST 9: the edited CLO appears in the final HTML
    def test_T9_edited_clo_appears_in_html(self):
        clo = db.get_course_outcomes(self.cid, self.db_path)[0]
        lesson = db.get_weekly_schedule(self.cid, self.db_path)[0]["lesson_outcomes"][1]
        edited_lesson = "Implement a breadth-first search in Python for a maze problem."
        db.update_course_outcome(clo["id"], EDITED_CLO, self.db_path)
        db.update_lesson_outcome(lesson["id"], edited_lesson, self.db_path)

        html = self.render()
        self.assertIn(EDITED_CLO, html)
        self.assertIn(edited_lesson, html)
        self.assertNotIn(esc(clo["original_text"]), html)   # AI wording is gone
        self.assertNotIn(esc(lesson["original_text"]), html)

    def test_T9b_edit_markers_are_optional(self):
        clo = db.get_course_outcomes(self.cid, self.db_path)[0]
        db.update_course_outcome(clo["id"], EDITED_CLO, self.db_path)
        self.assertNotIn('<span class="edited-flag">', self.render())
        self.assertIn('<span class="edited-flag">', self.render(show_edit_markers=True))

    def test_export_writes_html_file(self):
        out = Path(self._tmp.name) / "nested" / "syllabus.html"
        path = export_syllabus(self.cid, out, db_path=self.db_path)
        self.assertEqual(path, out)
        self.assertTrue(out.read_text(encoding="utf-8").lstrip().startswith("<!DOCTYPE html>"))

    def test_user_text_is_html_escaped(self):
        clo = db.get_course_outcomes(self.cid, self.db_path)[0]
        db.update_course_outcome(
            clo["id"], "Describe <b>bold</b> claims about artificial intelligence.", self.db_path
        )
        html = self.render()
        self.assertNotIn("<b>bold</b>", html)
        self.assertIn("&lt;b&gt;bold&lt;/b&gt;", html)

    def test_pvm_and_grading_shown_only_when_supplied(self):
        plain = self.render()
        self.assertNotIn('id="pvm"', plain)
        self.assertNotIn('id="grading"', plain)

        extras = load_extras(ROOT / "sample_extras.json")
        full = self.render(extras=extras)
        self.assertIn('id="pvm"', full)
        self.assertIn('id="grading"', full)
        self.assertIn("Midterm Examination", full)

    def test_invalid_extras_rejected(self):
        bad = Path(self._tmp.name) / "bad_extras.json"
        bad.write_text(json.dumps({
            "grading_matrix": {"columns": ["A", "B"], "rows": [["only one cell"]]}
        }), encoding="utf-8")
        with self.assertRaises(ExportError):
            load_extras(bad)
        with self.assertRaises(ExportError):          # typo'd key is rejected too
            self.render(extras={"grading": {}})
        with self.assertRaises(ExportError):
            load_extras(Path(self._tmp.name) / "missing.json")

    def test_missing_template_error(self):
        with self.assertRaises(ExportError):
            self.render(template_name="does_not_exist.html")

    def test_undefined_template_variable_friendly_error(self):
        tpl_dir = Path(self._tmp.name) / "tpl"
        tpl_dir.mkdir()
        (tpl_dir / "bad.html").write_text("<p>{{ nonexistent_value }}</p>", encoding="utf-8")
        with self.assertRaises(ExportError):
            self.render(template_name="bad.html", template_dir=tpl_dir)

    def test_invalid_course_id(self):
        with self.assertRaises(db.CourseNotFoundError):
            render_syllabus(999, db_path=self.db_path)

    def test_unwritable_output_path(self):
        blocker = Path(self._tmp.name) / "blocker"
        blocker.write_text("I am a file, not a folder", encoding="utf-8")
        with self.assertRaises(ExportError):
            export_syllabus(self.cid, blocker / "out.html", db_path=self.db_path)


if __name__ == "__main__":
    unittest.main()