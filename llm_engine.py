"""
llm_engine.py
Talks to a LOCAL Ollama server (no cloud APIs), asks Qwen for OBE content as
JSON, validates it with Pydantic, and retries with feedback when it fails.

Pipeline stage:  prompt -> Ollama -> JSON parse -> Pydantic -> (retry) -> result
"""
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, List, Tuple

import requests
from pydantic import ValidationError

from obe_schemas import (
    EXPECTED_WEEKS,
    CourseMetadataSchema,
    GeneratedContentSchema,
    SyllabusSchema,
)

log = logging.getLogger("obe.llm")

# ----------------------------------------------------------------- config
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
MODEL_NAME = os.getenv("OBE_MODEL", "qwen2.5:3b")

MAX_ATTEMPTS = 3
CONNECT_TIMEOUT_S = 5
READ_TIMEOUT_S = int(os.getenv("OBE_TIMEOUT", "900"))  # CPU generation is slow
TEMPERATURE = 0.3      # low = more consistent JSON
NUM_PREDICT = 4096     # max tokens to generate
NUM_CTX = 8192         # context window (prompt + retry feedback + answer)
MAX_PREVIOUS_CHARS = 4000


# ------------------------------------------------------------- exceptions
class LLMEngineError(Exception):
    """Base class. Messages are written for a student developer."""


class OllamaUnavailableError(LLMEngineError):
    pass


class ModelNotFoundError(LLMEngineError):
    pass


class OllamaTimeoutError(LLMEngineError):
    pass


class GenerationFailedError(LLMEngineError):
    """All attempts failed validation. `history` explains each failure."""

    def __init__(self, message: str, history: list):
        super().__init__(message)
        self.history = history


class StructuredOutputError(Exception):
    """Recoverable: empty or truncated model output. Triggers a retry."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.raw = raw


# ----------------------------------------------------------------- results
@dataclass
class AttemptRecord:
    attempt: int
    status: str      # ok | json_error | validation_error | output_error
    detail: str
    seconds: float


@dataclass
class GenerationResult:
    content: GeneratedContentSchema
    attempts: int
    history: List[AttemptRecord]
    raw_response: str


# ------------------------------------------------------------------ prompts
SYSTEM_PROMPT = f"""You are an Outcome-Based Education (OBE) curriculum designer for a university computing program.

Return ONLY one valid JSON object. No Markdown, no code fences, no comments, and no text before or after the JSON.

The JSON object must have exactly two top-level keys: "course_outcomes" and "weekly_schedule".
Do NOT include course codes, course titles, units, prerequisites, institution names, or any other metadata.

Required structure:
{{
  "course_outcomes": [
    {{"number": 1, "domain": "K", "text": "<measurable outcome starting with an action verb>"}}
  ],
  "weekly_schedule": [
    {{
      "week": 1,
      "topic": "<short topic title>",
      "description": "<one or two sentences describing the week>",
      "lesson_outcomes": [
        {{"domain": "K", "text": "<measurable outcome starting with an action verb>"}},
        {{"domain": "S", "text": "<measurable outcome starting with an action verb>"}},
        {{"domain": "A", "text": "<measurable outcome starting with an action verb>"}}
      ]
    }}
  ]
}}

Rules:
1. "course_outcomes" must contain 4 to 6 items, numbered 1, 2, 3, ... in order.
2. "weekly_schedule" must contain EXACTLY {EXPECTED_WEEKS} items, with "week" numbered 1 to {EXPECTED_WEEKS} in order. Never fewer, never more.
3. Every week must have at least one lesson outcome for EACH domain: K, S and A.
4. Domains: K = Knowledge (remembering and explaining concepts), S = Skills (applying, building, analyzing), A = Attitudes (values, responsibility, collaboration, professionalism).
5. Every outcome "text" must start with an active Bloom's Taxonomy verb, for example: define, describe, explain, compare, apply, implement, analyze, evaluate, design, construct, justify, demonstrate, collaborate.
6. NEVER start an outcome with a vague verb: understand, know, learn, be familiar with.
7. Every outcome must be measurable and observable, and at least 4 words long.
8. Keep weekly descriptions short (one or two sentences) and outcomes concise (one sentence).
9. Do not invent institutional information.
"""


def build_user_prompt(course: CourseMetadataSchema, notes: str = "") -> str:
    lines = [
        f"Create the course learning outcomes and the {EXPECTED_WEEKS}-week schedule for this course.",
        f"Course title: {course.course_title}",
        f"Course description: {course.description}",
        f"Units: {course.units}",
        f"Prerequisite: {course.prerequisite}",
    ]
    if notes.strip():
        lines.append(f"Additional instructor notes: {notes.strip()}")
    lines.append("Return the JSON object only.")
    return "\n".join(lines)


def build_retry_prompt(user_prompt: str, problems: str, previous: str) -> str:
    return (
        f"{user_prompt}\n\n"
        "Your previous response was REJECTED by the validator.\n"
        f"Problems found:\n{problems}\n\n"
        f"Previous response (may be shortened):\n{previous[:MAX_PREVIOUS_CHARS]}\n\n"
        "Fix every problem listed above. Return the COMPLETE corrected JSON object "
        f"(both keys, all {EXPECTED_WEEKS} weeks). JSON only."
    )


# --------------------------------------------------------------- Ollama I/O
def call_ollama(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Send one request to the local Ollama server and return the raw text."""
    payload = {
        "model": MODEL_NAME,
        "system": system,
        "prompt": prompt,
        "format": "json",   # Ollama JSON mode
        "stream": False,    # wait for the full answer
        "keep_alive": "10m",
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": NUM_PREDICT,
            "num_ctx": NUM_CTX,
        },
    }
    try:
        resp = requests.post(
            GENERATE_URL, json=payload, timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S)
        )
    except requests.exceptions.ConnectionError as exc:
        raise OllamaUnavailableError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Make sure the Ollama app is "
            "running (or run 'ollama serve' in a terminal)."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaTimeoutError(
            f"Ollama did not answer within {READ_TIMEOUT_S} seconds. The model may be "
            "too large for this computer. Try a smaller model or raise OBE_TIMEOUT."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise LLMEngineError(f"Unexpected network error talking to Ollama: {exc}") from exc

    if resp.status_code == 404:
        raise ModelNotFoundError(
            f"Model '{MODEL_NAME}' was not found in Ollama. Install it with: "
            f"ollama pull {MODEL_NAME}"
        )
    if resp.status_code >= 400:
        raise LLMEngineError(
            f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]} "
            "(If this mentions memory, use a smaller model.)"
        )

    try:
        data = resp.json()
    except ValueError as exc:
        raise LLMEngineError("Ollama returned a response that was not valid JSON.") from exc

    text = data.get("response", "") or ""
    if data.get("done_reason") == "length":
        raise StructuredOutputError(
            "The output was cut off because it reached the token limit. "
            "Be more concise and keep descriptions short.",
            raw=text,
        )
    if not text.strip():
        raise StructuredOutputError("The model returned an empty response.", raw=text)
    return text


def check_ollama() -> List[str]:
    """Verify Ollama is running and MODEL_NAME is installed. Returns model names."""
    try:
        resp = requests.get(TAGS_URL, timeout=CONNECT_TIMEOUT_S)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise OllamaUnavailableError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Start the Ollama app first."
        ) from exc

    names = [m.get("name", "") for m in resp.json().get("models", [])]
    installed = MODEL_NAME in names or (
        ":" not in MODEL_NAME and f"{MODEL_NAME}:latest" in names
    )
    if not installed:
        raise ModelNotFoundError(
            f"Model '{MODEL_NAME}' is not installed. Run: ollama pull {MODEL_NAME}\n"
            f"Installed models: {', '.join(names) or '(none)'}"
        )
    return names


# ---------------------------------------------------- parse + validate
def parse_and_validate(raw: str) -> GeneratedContentSchema:
    """JSON text -> dict -> Pydantic. Raises JSONDecodeError or ValidationError."""
    data = json.loads(raw)
    return GeneratedContentSchema.model_validate(data)


def format_validation_error(exc: ValidationError, limit: int = 15) -> str:
    """Turn a Pydantic error into short lines the LLM (and a human) can act on."""
    lines = []
    for err in exc.errors(include_url=False, include_input=False)[:limit]:
        where = ".".join(str(p) for p in err["loc"]) or "(whole document)"
        lines.append(f"- {where}: {err['msg']}")
    total = exc.error_count()
    if total > limit:
        lines.append(f"- ... and {total - limit} more problems")
    return "\n".join(lines)


# ---------------------------------------------------------- retry loop
def generate_content(
    course: CourseMetadataSchema,
    notes: str = "",
    *,
    generate_fn: Callable[[str, str], str] = call_ollama,
    max_attempts: int = MAX_ATTEMPTS,
) -> GenerationResult:
    """Ask the LLM for CLOs + schedule. Retry (with feedback) on bad output."""
    user_prompt = build_user_prompt(course, notes)
    history: List[AttemptRecord] = []
    problems = None
    previous = ""

    for attempt in range(1, max_attempts + 1):
        prompt = user_prompt if problems is None else build_retry_prompt(
            user_prompt, problems, previous
        )
        log.info("Attempt %d/%d -> sending prompt to %s", attempt, max_attempts, MODEL_NAME)
        started = time.perf_counter()
        raw = ""
        try:
            raw = generate_fn(prompt, SYSTEM_PROMPT)
            log.debug("Raw response (attempt %d): %s", attempt, raw[:2000])
            content = parse_and_validate(raw)
        except json.JSONDecodeError as exc:
            status = "json_error"
            problems = f"Invalid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})."
        except ValidationError as exc:
            status = "validation_error"
            problems = format_validation_error(exc)
        except StructuredOutputError as exc:
            status = "output_error"
            problems = str(exc)
            raw = exc.raw
        else:
            seconds = time.perf_counter() - started
            history.append(AttemptRecord(attempt, "ok", "Passed Pydantic validation", seconds))
            log.info("Attempt %d passed validation (%.1fs)", attempt, seconds)
            return GenerationResult(content, attempt, history, raw)

        seconds = time.perf_counter() - started
        history.append(AttemptRecord(attempt, status, problems, seconds))
        log.warning("Attempt %d failed (%s, %.1fs): %s", attempt, status, seconds, problems)
        previous = raw

    raise GenerationFailedError(
        f"The model did not produce valid output after {max_attempts} attempts. "
        f"Last problem:\n{problems}",
        history,
    )


def generate_syllabus(
    course_info: dict, **kwargs
) -> Tuple[SyllabusSchema, GenerationResult]:
    """
    Full pipeline entry point.
    course_info = form data: course_code, course_title, description, units,
    prerequisite, and optional 'notes'.
    Raises pydantic.ValidationError (bad form data) or LLMEngineError subclasses.
    """
    course = CourseMetadataSchema.model_validate(course_info)  # fail fast, before the LLM
    notes = str(course_info.get("notes") or "")[:500]
    result = generate_content(course, notes, **kwargs)
    syllabus = SyllabusSchema.model_validate(
        {"course": course.model_dump(), **result.content.model_dump()}
    )
    return syllabus, result


# ------------------------------------------------------------------- CLI
SAMPLE_COURSE_INFO = {
    "course_code": "SAMPLE-CS301",
    "course_title": "Artificial Intelligence (SAMPLE DATA)",
    "description": "An introduction to the principles of artificial intelligence, covering "
                   "intelligent agents, search, knowledge representation, reasoning under "
                   "uncertainty, machine learning, and the ethical use of AI systems.",
    "units": 3,
    "prerequisite": "Data Structures and Algorithms",
}


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="OBE LLM engine tools")
    parser.add_argument("--check", action="store_true", help="check Ollama and the model")
    parser.add_argument("--demo", action="store_true", help="generate the sample AI course")
    parser.add_argument("--verbose", action="store_true", help="show debug logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.check or not args.demo:
            names = check_ollama()
            print(f"OK: Ollama is running. Installed models: {', '.join(names)}")
            print(f"OK: model '{MODEL_NAME}' is available.")

        if args.demo:
            syllabus, result = generate_syllabus(SAMPLE_COURSE_INFO)
            print(f"\nSUCCESS after {result.attempts} attempt(s)")
            for rec in result.history:
                print(f"  attempt {rec.attempt}: {rec.status} - {rec.detail.splitlines()[0]} ({rec.seconds:.1f}s)")
            print(f"CLOs: {len(syllabus.course_outcomes)}, weeks: {len(syllabus.weekly_schedule)}")
            os.makedirs("output", exist_ok=True)
            out_path = os.path.join("output", "llm_demo_output.json")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(syllabus.model_dump_json(indent=2))
            print(f"Saved: {out_path}")
    except GenerationFailedError as exc:
        print(f"\nFAILED: {exc}")
        for rec in exc.history:
            print(f"  attempt {rec.attempt}: {rec.status} ({rec.seconds:.1f}s)")
        return 1
    except LLMEngineError as exc:
        print(f"\nERROR: {exc}")
        return 1
    except ValidationError as exc:
        print(f"\nINVALID COURSE INPUT:\n{format_validation_error(exc)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())