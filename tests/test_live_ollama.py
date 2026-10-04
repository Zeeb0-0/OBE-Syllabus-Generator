"""OPTIONAL real-Ollama test (TEST 4 with a live model). Skipped unless OBE_LIVE_TEST=1.
It can take several minutes (up to ~3 attempts on a CPU)."""
import os
import tempfile
import unittest
from pathlib import Path

import db_manager as db
import llm_engine
from llm_engine import LLMEngineError
from main import run_pipeline


@unittest.skipUnless(
    os.getenv("OBE_LIVE_TEST") == "1",
    "set OBE_LIVE_TEST=1 to run the real-Ollama test (takes minutes)",
)
class TestLiveOllama(unittest.TestCase):
    def test_T4_real_qwen_output_is_validated_and_saved(self):
        try:
            llm_engine.check_ollama()
        except LLMEngineError as exc:
            self.fail(str(exc))

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "live.db"
            result = run_pipeline(llm_engine.SAMPLE_COURSE_INFO, db_path=db_path)  # real Qwen call

            seconds = sum(r.seconds for r in result.generation.history)
            print(f"\n[live] attempts={result.generation.attempts}, total={seconds:.0f}s, model={llm_engine.MODEL_NAME}")

            self.assertLessEqual(result.generation.attempts, 3)
            schedule = db.get_weekly_schedule(result.course_id, db_path)
            self.assertEqual(len(schedule), 18)
            for week in schedule:
                domains = {lo["domain"] for lo in week["lesson_outcomes"]}
                self.assertTrue({"K", "S", "A"} <= domains, f"week {week['week']} lacks K/S/A")


if __name__ == "__main__":
    unittest.main()