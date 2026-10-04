"""Tests for llm_engine.py (spec TEST 3 and TEST 4, plus error handling).
No real Ollama is needed: the LLM is faked or requests.post is mocked."""
import json
import logging
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests
from pydantic import ValidationError

import llm_engine
from llm_engine import (
    GenerationFailedError,
    ModelNotFoundError,
    OllamaTimeoutError,
    OllamaUnavailableError,
    generate_content,
    generate_syllabus,
)
from obe_schemas import CourseMetadataSchema

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = json.loads((ROOT / "sample_validated_output.json").read_text(encoding="utf-8"))
COURSE_INFO = SAMPLE["course"]
VALID_JSON = json.dumps(
    {"course_outcomes": SAMPLE["course_outcomes"], "weekly_schedule": SAMPLE["weekly_schedule"]}
)


def bad_domain_json():
    data = json.loads(VALID_JSON)
    data["course_outcomes"][0]["domain"] = "X"
    return json.dumps(data)


def make_response(body, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body
    r.text = json.dumps(body)
    return r


def ok_body(text=VALID_JSON, reason="stop"):
    return {"response": text, "done": True, "done_reason": reason}


class ScriptedLLM:
    """Fake generate_fn: returns scripted answers and records the prompts."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt, system):
        self.prompts.append(prompt)
        return self.answers.pop(0)


class TestLLMEngine(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)
        self.course = CourseMetadataSchema.model_validate(COURSE_INFO)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    # ---- TEST 3: malformed JSON triggers retry handling
    def test_T3_malformed_json_triggers_retry(self):
        llm = ScriptedLLM(["{this is not json", VALID_JSON])
        result = generate_content(self.course, generate_fn=llm)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.history[0].status, "json_error")
        self.assertEqual(result.history[1].status, "ok")
        self.assertIn("REJECTED", llm.prompts[1])
        self.assertIn("Invalid JSON", llm.prompts[1])

    def test_T3b_validation_error_triggers_retry_with_feedback(self):
        llm = ScriptedLLM([bad_domain_json(), VALID_JSON])
        result = generate_content(self.course, generate_fn=llm)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.history[0].status, "validation_error")
        self.assertIn("course_outcomes.0.domain", llm.prompts[1])

    def test_gives_up_after_three_attempts(self):
        llm = ScriptedLLM(["bad", "still bad", "{nope"])
        with self.assertRaises(GenerationFailedError) as ctx:
            generate_content(self.course, generate_fn=llm)
        self.assertEqual(len(llm.prompts), 3)
        self.assertEqual(len(ctx.exception.history), 3)

    # ---- TEST 4: valid JSON coming back from (mocked) Ollama is accepted
    @patch("llm_engine.requests.post")
    def test_T4_ollama_valid_json_accepted(self, mock_post):
        mock_post.return_value = make_response(ok_body())
        result = generate_content(self.course)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(len(result.content.weekly_schedule), 18)
        sent = mock_post.call_args.kwargs["json"]
        self.assertEqual(sent["format"], "json")
        self.assertFalse(sent["stream"])
        self.assertEqual(sent["model"], llm_engine.MODEL_NAME)

    @patch("llm_engine.requests.post")
    def test_truncated_output_is_retried(self, mock_post):
        mock_post.side_effect = [
            make_response(ok_body(text='{"course_outcomes": [', reason="length")),
            make_response(ok_body()),
        ]
        result = generate_content(self.course)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.history[0].status, "output_error")

    # ---- error handling
    @patch("llm_engine.requests.post", side_effect=requests.exceptions.ConnectionError())
    def test_ollama_unavailable(self, _):
        with self.assertRaises(OllamaUnavailableError):
            generate_content(self.course)

    @patch("llm_engine.requests.post")
    def test_model_not_found(self, mock_post):
        mock_post.return_value = make_response({"error": "model not found"}, status=404)
        with self.assertRaises(ModelNotFoundError):
            generate_content(self.course)

    @patch("llm_engine.requests.post", side_effect=requests.exceptions.ReadTimeout())
    def test_timeout_is_not_retried(self, mock_post):
        with self.assertRaises(OllamaTimeoutError):
            generate_content(self.course)
        self.assertEqual(mock_post.call_count, 1)

    # ---- full pipeline entry point
    def test_invalid_course_info_rejected_before_llm_call(self):
        llm = ScriptedLLM([VALID_JSON])
        with self.assertRaises(ValidationError):
            generate_syllabus({"course_code": "X", "course_title": "A"}, generate_fn=llm)
        self.assertEqual(llm.prompts, [])

    def test_generate_syllabus_merges_user_metadata(self):
        llm = ScriptedLLM([VALID_JSON])
        syllabus, result = generate_syllabus(COURSE_INFO, generate_fn=llm)
        self.assertEqual(syllabus.course.course_code, COURSE_INFO["course_code"])
        self.assertEqual(len(syllabus.weekly_schedule), 18)
        self.assertEqual(result.attempts, 1)


if __name__ == "__main__":
    unittest.main()