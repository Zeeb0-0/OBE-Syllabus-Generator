"""
verify_requirements.py
Runs the tests behind each requirement in the assignment spec and prints a
PASS/FAIL report you can paste into your documentation.

Usage:  python verify_requirements.py
"""
import logging
import re
import sys
import unittest
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
logging.getLogger("obe").addHandler(logging.NullHandler())  # keep the report clean

# (label, description, [unittest ids that must all pass])
REQUIRED_TESTS = [
    ("TEST 1", "Valid JSON passes Pydantic",
     ["tests.test_schemas.TestSchemas.test_T1_valid_samples_pass"]),
    ("TEST 2", "Invalid domain 'X' fails validation",
     ["tests.test_schemas.TestSchemas.test_T2_invalid_course_domain_fails",
      "tests.test_schemas.TestSchemas.test_T2b_invalid_lesson_domain_fails"]),
    ("TEST 3", "Malformed JSON triggers retry handling",
     ["tests.test_llm_engine.TestLLMEngine.test_T3_malformed_json_triggers_retry",
      "tests.test_llm_engine.TestLLMEngine.test_T3b_validation_error_triggers_retry_with_feedback",
      "tests.test_integration.TestIntegration.test_full_demo_flow_through_the_web_ui"]),
    ("TEST 4", "Ollama-generated valid JSON is accepted (mocked HTTP)",
     ["tests.test_llm_engine.TestLLMEngine.test_T4_ollama_valid_json_accepted"]),
    ("TEST 5", "A course can be inserted into SQLite",
     ["tests.test_db.TestDatabase.test_T5_course_inserted"]),
    ("TEST 6", "Course outcomes can be edited",
     ["tests.test_editing.TestEditing.test_T6_edit_clo_saved_and_original_preserved"]),
    ("TEST 7", "Weekly schedules are persisted correctly",
     ["tests.test_db.TestDatabase.test_T7_weekly_schedule_persisted_in_order"]),
    ("TEST 8", "Jinja2 generates valid HTML",
     ["tests.test_export.TestExport.test_T8_html_is_well_formed",
      "tests.test_export.TestExport.test_T8b_all_database_content_is_rendered"]),
    ("TEST 9", "The edited CLO appears in the final HTML",
     ["tests.test_export.TestExport.test_T9_edited_clo_appears_in_html",
      "tests.test_web.WebTests.test_edit_save_export_shows_edited_clo"]),
    ("TEST 10", "Deleting a course handles related records",
     ["tests.test_db.TestDatabase.test_T10_delete_cascades_to_all_children",
      "tests.test_db.TestDatabase.test_T10b_delete_leaves_other_courses_untouched"]),
]

ERROR_CASES = [
    ("Ollama unavailable",
     ["tests.test_llm_engine.TestLLMEngine.test_ollama_unavailable",
      "tests.test_integration.TestIntegration.test_ollama_down_gives_friendly_message_in_ui"]),
    ("Model unavailable", ["tests.test_llm_engine.TestLLMEngine.test_model_not_found"]),
    ("Connection timeout", ["tests.test_llm_engine.TestLLMEngine.test_timeout_is_not_retried"]),
    ("Malformed JSON", ["tests.test_llm_engine.TestLLMEngine.test_T3_malformed_json_triggers_retry"]),
    ("Pydantic ValidationError",
     ["tests.test_llm_engine.TestLLMEngine.test_T3b_validation_error_triggers_retry_with_feedback"]),
    ("Database errors",
     ["tests.test_db.TestDatabase.test_db_check_constraint_rejects_bad_domain",
      "tests.test_db.TestDatabase.test_foreign_keys_enforced",
      "tests.test_db.TestDatabase.test_duplicate_course_code_rejected"]),
    ("Missing template", ["tests.test_export.TestExport.test_missing_template_error"]),
    ("Invalid course ID",
     ["tests.test_db.TestDatabase.test_invalid_course_id",
      "tests.test_web.WebTests.test_unknown_course_shows_friendly_404"]),
    ("Export failure", ["tests.test_export.TestExport.test_unwritable_output_path"]),
]

REQUIRED_FILES = [
    "main.py", "llm_engine.py", "obe_schemas.py", "db_manager.py", "export_engine.py",
    "schema.sql", "requirements.txt", "sample_validated_output.json",
    "templates/index.html", "templates/edit_course.html", "templates/uphsd_ccs_template.html",
    "static/style.css",
]


def run_tests(test_ids):
    suite = unittest.TestSuite()
    for test_id in test_ids:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName(test_id))
    result = unittest.TestResult()
    suite.run(result)
    ok = result.wasSuccessful() and result.testsRun == len(test_ids)
    detail = ""
    if not ok:
        problems = result.failures + result.errors
        detail = problems[0][1].strip().splitlines()[-1] if problems else "test not found"
    return ok, detail


# ------------------------------------------------------- project checks
def check_required_files():
    missing = [f for f in REQUIRED_FILES if not (BASE_DIR / f).exists()]
    return not missing, ("missing: " + ", ".join(missing)) if missing else ""


def check_requirements_file():
    path = BASE_DIR / "requirements.txt"
    if not path.exists():
        return False, "requirements.txt not found"
    names = {re.split(r"[<>=!~\[ ]", line.strip())[0].lower()
             for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    missing = {"pydantic", "jinja2", "requests"} - names
    return not missing, ("missing: " + ", ".join(sorted(missing))) if missing else ""


def check_no_cloud_ai():
    import_pattern = re.compile(
        r"^\s*(?:import|from)\s+(?:openai|anthropic|google\.generativeai|google\.genai|cohere|groq|mistralai)\b",
        re.MULTILINE)
    hosts = ("api.openai.com", "api." + "anthropic.com", "generativelanguage." + "googleapis.com")
    files = [p for p in [*BASE_DIR.glob("*.py"), *(BASE_DIR / "tests").glob("*.py")]
             if p.name != Path(__file__).name]
    offenders = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        if import_pattern.search(text) or any(h in text for h in hosts):
            offenders.append(path.name)
    return not offenders, ("found in: " + ", ".join(offenders)) if offenders else ""


def check_local_ollama():
    import llm_engine
    host = urlparse(llm_engine.OLLAMA_BASE_URL).hostname
    ok = host in {"localhost", "127.0.0.1"} and llm_engine.MODEL_NAME.startswith("qwen2.5")
    return ok, f"host={host}, model={llm_engine.MODEL_NAME}"


def check_json_mode_and_retries():
    import llm_engine
    source = (BASE_DIR / "llm_engine.py").read_text(encoding="utf-8")
    ok = '"format": "json"' in source and llm_engine.MAX_ATTEMPTS == 3
    return ok, f"max attempts = {llm_engine.MAX_ATTEMPTS}"


PROJECT_CHECKS = [
    ("All required project files exist", check_required_files),
    ("requirements.txt lists pydantic, jinja2, requests", check_requirements_file),
    ("No cloud AI libraries or API hosts in the code", check_no_cloud_ai),
    ("Ollama is local and the model is Qwen 2.5", check_local_ollama),
    ("Ollama JSON mode on, retries capped at 3", check_json_mode_and_retries),
]


def main() -> int:
    failures = 0
    print("OBE Syllabus Generator - requirements verification")
    print("=" * 60)

    print("\nA. Required tests (assignment section 12)")
    for label, title, ids in REQUIRED_TESTS:
        ok, detail = run_tests(ids)
        failures += 0 if ok else 1
        print(f"  {label:<8}{title:<58}{'PASS' if ok else 'FAIL'}")
        if detail:
            print(f"          -> {detail}")

    print("\nB. Error handling (assignment section 11)")
    for title, ids in ERROR_CASES:
        ok, detail = run_tests(ids)
        failures += 0 if ok else 1
        print(f"  {title:<66}{'PASS' if ok else 'FAIL'}")
        if detail:
            print(f"          -> {detail}")

    print("\nC. Project checks")
    for title, check in PROJECT_CHECKS:
        ok, detail = check()
        failures += 0 if ok else 1
        extra = f"  ({detail})" if detail else ""
        print(f"  {title:<58}{'PASS' if ok else 'FAIL'}{extra}")
    readme = (BASE_DIR / "README.md").exists()
    print(f"  {'README.md present':<58}{'PASS' if readme else 'PENDING (written in Phase 10)'}")

    print("\n" + "=" * 60)
    if failures:
        print(f"RESULT: {failures} check(s) FAILED")
        return 1
    print(f"RESULT: ALL CHECKS PASSED ({len(REQUIRED_TESTS)} required tests, "
          f"{len(ERROR_CASES)} error-handling cases, {len(PROJECT_CHECKS)} project checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())