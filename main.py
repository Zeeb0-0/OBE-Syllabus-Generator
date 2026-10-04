"""
main.py - the controller and the web interface.

Pipeline for one course:

  form data -> validate -> (duplicate check) -> LLM -> JSON -> Pydantic
            -> retry on failure -> SQLite -> human edits -> Jinja2 -> HTML

Two ways to use it (both share run_pipeline):
  python main.py            -> starts the web interface (Phase 8)
  python main.py --sample   -> command-line pipeline (Phase 5)
"""
import argparse
import json
import logging
import os
import sys
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, Dict, List, Optional

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from pydantic import ValidationError
from werkzeug.exceptions import HTTPException

import db_manager
import llm_engine
from db_manager import (
    CourseNotFoundError,
    DBError,
    DuplicateCourseError,
    InvalidEditError,
    OutcomeNotFoundError,
)
from export_engine import ExportError, export_syllabus, load_extras
from llm_engine import (
    GenerationFailedError,
    GenerationResult,
    LLMEngineError,
    format_validation_error,
)
from obe_schemas import CourseMetadataSchema, SyllabusSchema

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
EXPORT_NAME = "generated_syllabus.html"
LAST_LIVE_NAME = "last_live_generation.json"
SAMPLE_EXTRAS = BASE_DIR / "sample_extras.json"
log = logging.getLogger("obe.main")

STAGES = ["init", "generate", "validated", "saved"]


# ---------------------------------------------------------------- logging
def setup_logging(verbose: bool = False) -> None:
    """Console: INFO (DEBUG with --verbose). File logs/obe.log: always DEBUG."""
    LOG_DIR.mkdir(exist_ok=True)
    root = logging.getLogger("obe")
    root.setLevel(logging.DEBUG)
    root.propagate = False
    root.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    logfile = RotatingFileHandler(
        LOG_DIR / "obe.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8"
    )
    logfile.setLevel(logging.DEBUG)
    logfile.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(logfile)


# --------------------------------------------------------------- pipeline
@dataclass
class PipelineResult:
    course_id: int
    syllabus: SyllabusSchema
    generation: GenerationResult


def make_replay_fn(path) -> Callable[[str, str], str]:
    """A stand-in for the LLM call that returns a previously saved JSON file.
    Everything after the LLM (parse, Pydantic, SQLite) still runs for real."""
    text = Path(path).read_text(encoding="utf-8")

    def replay(prompt: str, system: str) -> str:
        return text

    return replay


def run_pipeline(
    course_info: dict,
    *,
    replace: bool = False,
    generate_fn: Optional[Callable[[str, str], str]] = None,
    db_path=None,
    on_stage: Optional[Callable[[str, str], None]] = None,
) -> PipelineResult:
    """
    Raises: pydantic.ValidationError (bad form data), DuplicateCourseError,
    LLMEngineError subclasses (Ollama problems / output never valid), DBError.
    """
    say = on_stage or (lambda stage, message: None)

    # 1. Fail fast on bad form data (milliseconds, not minutes)
    course = CourseMetadataSchema.model_validate(course_info)

    say("init", "Preparing the database")
    db_manager.init_db(db_path)

    # 2. Fail fast on duplicates, BEFORE the slow LLM call
    if not replace:
        for existing in db_manager.list_courses(db_path):
            if existing["course_code"] == course.course_code:
                raise DuplicateCourseError(
                    f"A course with code '{course.course_code}' already exists "
                    f"(id {existing['id']}). Choose 'replace' to overwrite it, or delete it first."
                )

    # 3. LLM -> JSON -> Pydantic, with automatic retries
    if generate_fn is None:
        say("generate", f"Asking {llm_engine.MODEL_NAME} (validation and retries are automatic)")
        kwargs = {}
    else:
        say("generate", "Using a supplied generator (replay or test), the live LLM is bypassed")
        kwargs = {"generate_fn": generate_fn}
    syllabus, generation = llm_engine.generate_syllabus(course_info, **kwargs)
    say("validated", f"Pydantic accepted the output after {generation.attempts} attempt(s)")

    # 4. Only validated data reaches SQLite
    course_id = db_manager.create_course(syllabus, replace=replace, db_path=db_path)
    say("saved", f"Saved to SQLite as course id {course_id}")

    log.info("Pipeline finished: course id %s, %d attempt(s)", course_id, generation.attempts)
    return PipelineResult(course_id, syllabus, generation)


def describe_error(exc: Exception) -> Optional[str]:
    """Plain-language message for EXPECTED errors; None for anything unexpected."""
    if isinstance(exc, ValidationError):
        return "The course information is not valid:\n" + format_validation_error(exc)
    if isinstance(exc, GenerationFailedError):
        attempts = "; ".join(f"attempt {r.attempt}: {r.status}" for r in exc.history)
        return f"{exc} ({attempts}). Nothing was saved."
    if isinstance(exc, (LLMEngineError, DBError)):
        return str(exc)
    return None


# ================================================================ WEB (Phase 8)
@dataclass
class Job:
    """One background generation. Read by the progress page through /api/jobs/<id>."""

    id: str
    course_code: str
    mode: str                      # "live" or "replay"
    status: str = "running"        # running | done | error
    stage: str = "init"
    course_id: Optional[int] = None
    error: Optional[str] = None
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    log: List[str] = field(default_factory=list)

    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    def add(self, text: str) -> None:
        self.log.append(f"{self.elapsed():7.1f}s  {text}")

    def on_stage(self, stage: str, message: str) -> None:
        self.stage = stage
        self.add(f"[{STAGES.index(stage) + 1}/{len(STAGES)}] {message}")


# thread id -> job, so log lines written by a worker thread land in its job
THREAD_JOBS: Dict[int, Job] = {}


class JobLogHandler(logging.Handler):
    """Copies the engine's log lines (attempts, failures, timings) into the running job."""

    def emit(self, record: logging.LogRecord) -> None:
        job = THREAD_JOBS.get(record.thread)
        if job is not None and record.levelno >= logging.INFO:
            job.add(f"{record.levelname:<7} {record.getMessage()}")


def install_job_log_handler() -> None:
    root = logging.getLogger("obe")
    if not any(isinstance(h, JobLogHandler) for h in root.handlers):
        root.addHandler(JobLogHandler())


def replay_files(output_dir: Path) -> Dict[str, tuple]:
    """Saved results the UI may replay. Keys (not paths) come from the form."""
    candidates = {
        "sample_ai": ("Sample: Artificial Intelligence (hand-written SAMPLE data)",
                      BASE_DIR / "sample_validated_output.json"),
        "sample_dsa": ("Sample: Data Structures and Algorithms (hand-written SAMPLE data)",
                       BASE_DIR / "sample_dsa_validated_output.json"),
        "last_live": ("Last live generation (real Qwen output)", output_dir / LAST_LIVE_NAME),
        "phase3": ("Earlier test run (real Qwen output)", output_dir / "llm_demo_output.json"),
    }
    return {key: value for key, value in candidates.items() if value[1].exists()}


def ollama_status(skip: bool) -> dict:
    if skip:
        return {"ok": True, "text": "not checked (test mode)"}
    try:
        llm_engine.check_ollama()
        return {"ok": True, "text": f"running, model {llm_engine.MODEL_NAME} is installed"}
    except LLMEngineError as exc:
        return {"ok": False, "text": str(exc).splitlines()[0]}


def blank_form() -> dict:
    return {"course_code": "", "course_title": "", "description": "", "units": "3",
            "prerequisite": "", "notes": "", "mode": "live", "replay_key": "", "replace": False}


FORM_FIELDS = ("course_code", "course_title", "description", "units", "prerequisite", "notes")


def create_app(db_path=None, output_dir=None, live_generate_fn=None) -> Flask:
    """App factory. Tests pass a temp db_path/output_dir and a fake live_generate_fn."""
    app = Flask(__name__)
    app.secret_key = os.getenv("OBE_SECRET_KEY", "local-demo-only-not-a-secret")
    out_dir = Path(output_dir) if output_dir else BASE_DIR / "output"
    app.config["LIVE_GENERATE_FN"] = live_generate_fn

    jobs: Dict[str, Job] = {}
    jobs_lock = threading.Lock()

    db_manager.init_db(db_path)
    install_job_log_handler()

    # ------------------------------------------------------------ job runner
    def save_last_live(result: PipelineResult) -> None:
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / LAST_LIVE_NAME).write_text(
                result.syllabus.model_dump_json(indent=2), encoding="utf-8"
            )
        except OSError as exc:
            log.warning("Could not save %s: %s", LAST_LIVE_NAME, exc)

    def run_job(job: Job, course_info: dict, replace: bool, generate_fn) -> None:
        THREAD_JOBS[threading.get_ident()] = job
        try:
            result = run_pipeline(
                course_info, replace=replace, generate_fn=generate_fn,
                db_path=db_path, on_stage=job.on_stage,
            )
            if job.mode == "live":
                save_last_live(result)  # becomes a real replay source for demos
            job.course_id = result.course_id
            job.status = "done"
        except Exception as exc:  # noqa: BLE001 - report every failure to the UI
            message = describe_error(exc)
            if message is None:
                log.exception("Unexpected error in generation job")
                message = f"Unexpected error: {exc}. See logs/obe.log for details."
            job.error = message
            job.status = "error"
        finally:
            job.finished = time.time()
            THREAD_JOBS.pop(threading.get_ident(), None)

    def start_job(course_info: dict, replace: bool, generate_fn, mode: str) -> Optional[Job]:
        with jobs_lock:
            if any(j.status == "running" for j in jobs.values()):
                return None  # one generation at a time (RAM)
            job = Job(id=uuid.uuid4().hex[:8], course_code=str(course_info.get("course_code", "")), mode=mode)
            jobs[job.id] = job
        threading.Thread(
            target=run_job, args=(job, course_info, replace, generate_fn), daemon=True
        ).start()
        return job

    def running_job() -> Optional[Job]:
        return next((j for j in jobs.values() if j.status == "running"), None)

    # ---------------------------------------------------------------- pages
    def index_context(form=None) -> dict:
        files = replay_files(out_dir)
        return {
            "form": form or blank_form(),
            "courses": db_manager.list_courses(db_path),
            "replays": [(key, item[0]) for key, item in files.items()],
            "sample_course": llm_engine.SAMPLE_COURSE_INFO,
            "ollama": ollama_status(skip=live_generate_fn is not None),
            "running_job": running_job(),
        }

    @app.get("/")
    def index():
        return render_template("index.html", **index_context())

    @app.post("/generate")
    def generate():
        form = blank_form()
        for key in FORM_FIELDS + ("replay_key",):
            form[key] = request.form.get(key, "").strip()
        form["mode"] = "replay" if request.form.get("mode") == "replay" else "live"
        form["replace"] = request.form.get("replace") == "on"

        def fail(message: str):
            flash(message, "error")
            return render_template("index.html", **index_context(form)), 400

        try:
            if form["mode"] == "replay":
                files = replay_files(out_dir)
                if form["replay_key"] not in files:
                    return fail("Please choose a saved result to replay.")
                path = files[form["replay_key"]][1]
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("the saved file must contain a JSON object")
                course_info = dict(data.get("course", {}))
                generate_fn = make_replay_fn(path)
            else:
                course_info = {k: form[k] for k in FORM_FIELDS if form[k]}
                generate_fn = app.config["LIVE_GENERATE_FN"]

            course = CourseMetadataSchema.model_validate(course_info)
            if not form["replace"]:
                for existing in db_manager.list_courses(db_path):
                    if existing["course_code"] == course.course_code:
                        return fail(
                            f"A course with code '{course.course_code}' already exists "
                            f"(id {existing['id']}). Tick 'Replace existing course' to "
                            "overwrite it, or delete it first."
                        )
            if form["mode"] == "live" and generate_fn is None:
                llm_engine.check_ollama()  # friendly error now, not after a long wait
        except Exception as exc:  # noqa: BLE001
            message = describe_error(exc)
            if message is None and isinstance(exc, (OSError, ValueError)):
                message = f"Could not read the saved result: {exc}"
            if message is None:
                raise
            return fail(message)

        job = start_job(course_info, form["replace"], generate_fn, form["mode"])
        if job is None:
            return fail("A generation is already running. Wait for it to finish first.")
        return redirect(url_for("job_page", job_id=job.id))

    @app.get("/jobs/<job_id>")
    def job_page(job_id):
        job = jobs.get(job_id)
        if job is None:
            return render_template("error.html", title="Unknown job",
                                   message="That generation job does not exist (the server may have restarted)."), 404
        return render_template("job.html", job=job)

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id):
        job = jobs.get(job_id)
        if job is None:
            return jsonify(error="Unknown job"), 404
        return jsonify(
            status=job.status,
            stage=job.stage,
            elapsed=round(job.elapsed(), 1),
            log=list(job.log),
            error=job.error,
            course_url=url_for("edit_course", course_id=job.course_id) if job.course_id else None,
        )

    @app.get("/courses/<int:course_id>")
    def edit_course(course_id):
        data = db_manager.get_full_syllabus(course_id, db_path)  # CourseNotFoundError -> 404 page
        edited = sum(1 for o in data["course_outcomes"] if o["is_edited"]) + sum(
            1 for w in data["weekly_schedule"] for lo in w["lesson_outcomes"] if lo["is_edited"]
        )
        return render_template(
            "edit_course.html",
            course=data["course"], outcomes=data["course_outcomes"], weeks=data["weekly_schedule"],
            edited_count=edited, has_sample_extras=SAMPLE_EXTRAS.exists(),
        )

    # ------------------------------------------------------------- editing
    def apply_edit(course_id: int, row_id: int, kind: str):
        action = request.form.get("action", "save")
        try:
            if kind == "clo":
                if action == "reset":
                    db_manager.reset_course_outcome(row_id, db_path)
                else:
                    db_manager.update_course_outcome(row_id, request.form.get("text", ""), db_path)
            else:
                if action == "reset":
                    db_manager.reset_lesson_outcome(row_id, db_path)
                else:
                    db_manager.update_lesson_outcome(row_id, request.form.get("text", ""), db_path)
            flash("Restored the AI-generated wording." if action == "reset"
                  else "Saved. The exported syllabus will use this wording.", "ok")
        except (InvalidEditError, OutcomeNotFoundError) as exc:
            flash(str(exc), "error")
        return redirect(url_for("edit_course", course_id=course_id, _anchor=f"{kind}-{row_id}"))

    @app.post("/courses/<int:course_id>/clo/<int:outcome_id>")
    def edit_clo(course_id, outcome_id):
        return apply_edit(course_id, outcome_id, "clo")

    @app.post("/courses/<int:course_id>/lesson/<int:lesson_id>")
    def edit_lesson(course_id, lesson_id):
        return apply_edit(course_id, lesson_id, "lesson")

    # ------------------------------------------------------ export / delete
    @app.post("/courses/<int:course_id>/export")
    def export_course(course_id):
        extras = None
        if request.form.get("sample_extras") == "on" and SAMPLE_EXTRAS.exists():
            extras = load_extras(SAMPLE_EXTRAS)
        export_syllabus(
            course_id, out_dir / EXPORT_NAME, db_path=db_path, extras=extras,
            show_edit_markers=request.form.get("mark_edits") == "on",
        )
        return redirect(url_for("exported_file", filename=EXPORT_NAME))

    @app.get("/output/<path:filename>")
    def exported_file(filename):
        response = send_from_directory(out_dir, filename)  # only serves files inside output/
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.post("/courses/<int:course_id>/delete")
    def remove_course(course_id):
        db_manager.delete_course(course_id, db_path)
        flash("Course deleted (including its outcomes and weekly schedule).", "ok")
        return redirect(url_for("index"))

    # ------------------------------------------------------- error handling
    @app.errorhandler(CourseNotFoundError)
    def on_course_not_found(exc):
        return render_template("error.html", title="Course not found", message=str(exc)), 404

    @app.errorhandler(404)
    def on_404(exc):
        return render_template("error.html", title="Page not found",
                               message="That page does not exist."), 404

    @app.errorhandler(DBError)
    @app.errorhandler(ExportError)
    def on_known_error(exc):
        log.error("Request %s failed: %s", request.path, exc)
        return render_template("error.html", title="Something went wrong", message=str(exc)), 500

    @app.errorhandler(Exception)
    def on_unexpected(exc):
        if isinstance(exc, HTTPException):
            return exc
        log.exception("Unexpected error while handling %s", request.path)
        return render_template(
            "error.html", title="Something went wrong",
            message="An unexpected error occurred. Details were written to logs/obe.log.",
        ), 500

    return app


def serve_web(port: int, open_browser: bool) -> int:
    app = create_app()
    url = f"http://127.0.0.1:{port}/"
    print(f"OBE Syllabus Generator running at {url}  (press Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
    except OSError as exc:
        print(f"ERROR: could not start the server on port {port}: {exc}\n"
              "Try another port: python main.py --serve --port 5001")
        return 1
    return 0


# -------------------------------------------------------------------- CLI
def print_stage(stage: str, message: str) -> None:
    print(f"[{STAGES.index(stage) + 1}/{len(STAGES)}] {message}", flush=True)


def print_summary(result: PipelineResult) -> None:
    s = result.syllabus
    lessons = sum(len(w.lesson_outcomes) for w in s.weekly_schedule)
    first = s.course_outcomes[0]
    print("\nDONE")
    print(f"  Course   : {s.course.course_code} - {s.course.course_title}")
    print(f"  Saved    : course id {result.course_id} "
          f"({len(s.course_outcomes)} CLOs, {len(s.weekly_schedule)} weeks, {lessons} lesson outcomes)")
    print(f"  Attempts : {result.generation.attempts}")
    print(f"  First CLO: [{first.domain}] {first.text}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="OBE Syllabus Generator (no options = start the web interface)")
    p.add_argument("--serve", action="store_true", help="start the web interface")
    p.add_argument("--port", type=int, default=5000, help="web interface port (default 5000)")
    p.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--sample", action="store_true",
                     help="CLI: use the built-in sample AI course with the REAL LLM")
    src.add_argument("--replay", metavar="FILE",
                     help="CLI: skip the LLM; replay a saved JSON (e.g. output/llm_demo_output.json)")
    p.add_argument("--code", help="course code")
    p.add_argument("--title", help="course title")
    p.add_argument("--description", help="course description (at least 20 characters)")
    p.add_argument("--units", type=int, help="units (1-6)")
    p.add_argument("--prerequisite", help="prerequisite course(s)")
    p.add_argument("--notes", help="optional extra instructions for the LLM")
    p.add_argument("--replace", action="store_true", help="overwrite an existing course with the same code")
    p.add_argument("--verbose", action="store_true", help="show debug logs on the console")
    return p.parse_args(argv)


def build_course_info(args) -> dict:
    base = {}
    if args.replay:
        with open(args.replay, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("The replay file must contain a JSON object.")
        base = dict(data.get("course", {}))
    elif args.sample:
        base = dict(llm_engine.SAMPLE_COURSE_INFO)

    overrides = {
        "course_code": args.code, "course_title": args.title,
        "description": args.description, "units": args.units,
        "prerequisite": args.prerequisite, "notes": args.notes,
    }
    base.update({k: v for k, v in overrides.items() if v is not None})
    return base


def main(argv=None) -> int:
    argv_list = sys.argv[1:] if argv is None else list(argv)
    args = parse_args(argv_list)
    setup_logging(args.verbose)

    if args.serve or not argv_list:
        return serve_web(args.port, open_browser=not args.no_browser)

    try:
        course_info = build_course_info(args)
        generate_fn = make_replay_fn(args.replay) if args.replay else None
        result = run_pipeline(course_info, replace=args.replace,
                              generate_fn=generate_fn, on_stage=print_stage)
        print_summary(result)
        return 0
    except KeyboardInterrupt:
        print("\nCancelled. Nothing was saved.")
        return 130
    except ValidationError as exc:
        print("\nThe course information is not valid:\n" + format_validation_error(exc))
    except DuplicateCourseError as exc:
        print(f"\nERROR: {exc}\nTip: add --replace to overwrite it.")
    except GenerationFailedError as exc:
        print(f"\nFAILED: {exc}")
        for rec in exc.history:
            print(f"  attempt {rec.attempt}: {rec.status} ({rec.seconds:.1f}s)")
        print("Nothing was saved. See logs/obe.log for the raw model output.")
    except (LLMEngineError, DBError) as exc:
        print(f"\nERROR: {exc}")
    except (OSError, ValueError) as exc:  # missing/unreadable replay file, bad JSON
        print(f"\nERROR: could not read the input file: {exc}")
    log.debug("Pipeline ended with an error", exc_info=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())