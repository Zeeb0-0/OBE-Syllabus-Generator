"""
db_manager.py
SQLite persistence for validated OBE syllabi. Pure parameterized SQL.

Pipeline stage:  validated SyllabusSchema -> SQLite -> human edits -> dicts for export
"""
import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import ValidationError

from obe_schemas import CourseOutcomeSchema, LessonOutcomeSchema, SyllabusSchema

log = logging.getLogger("obe.db")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = Path(os.getenv("OBE_DB_PATH", BASE_DIR / "data" / "obe_syllabus.db"))
SCHEMA_PATH = BASE_DIR / "schema.sql"


# ------------------------------------------------------------- exceptions
class DBError(Exception):
    """Base class. Messages are written for a student developer."""


class CourseNotFoundError(DBError):
    pass


class DuplicateCourseError(DBError):
    pass


class OutcomeNotFoundError(DBError):
    pass


class InvalidEditError(DBError):
    """A human edit was rejected by the Pydantic rules (nothing was changed)."""


# -------------------------------------------------------------- connection
@contextmanager
def get_connection(db_path=None):
    """Open a connection with foreign keys ON. Commit on success, rollback on error."""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    conn = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")  # without this, CASCADE does nothing
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        if conn:
            conn.rollback()
        log.error("SQLite error: %s", exc)
        raise DBError(f"Database error: {exc}") from exc
    except Exception:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def init_db(db_path=None) -> None:
    """Create the tables from schema.sql (safe to run many times)."""
    if not SCHEMA_PATH.exists():
        raise DBError(f"schema.sql not found at {SCHEMA_PATH}. Keep it next to db_manager.py.")
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with get_connection(db_path) as conn:
        conn.executescript(sql)
    log.info("Database ready at %s", db_path or DEFAULT_DB_PATH)


# ------------------------------------------------------------------ helpers
def _require_course(conn, course_id) -> None:
    row = conn.execute("SELECT 1 FROM courses WHERE id = ?", (course_id,)).fetchone()
    if row is None:
        raise CourseNotFoundError(f"No course with id {course_id}.")


def _outcome_dict(row) -> Dict:
    d = dict(row)
    d["is_edited"] = d["text"] != d["original_text"]
    return d


def _friendly_validation_message(exc: ValidationError) -> str:
    messages = []
    for err in exc.errors(include_url=False, include_input=False):
        msg = err["msg"]
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        messages.append(msg)
    return "; ".join(messages)


# ------------------------------------------------------------------- CREATE
def create_course(syllabus: SyllabusSchema, replace: bool = False, db_path=None) -> int:
    """
    Store a validated syllabus (course + CLOs + 18 weeks + lesson outcomes)
    in ONE transaction. Returns the new course id.
    replace=True first deletes any existing course with the same code.
    """
    if not isinstance(syllabus, SyllabusSchema):
        raise DBError(
            "create_course() only accepts a validated SyllabusSchema. "
            "Run Pydantic validation before saving."
        )
    c = syllabus.course
    with get_connection(db_path) as conn:
        if replace:
            conn.execute("DELETE FROM courses WHERE course_code = ?", (c.course_code,))
        try:
            cur = conn.execute(
                "INSERT INTO courses (course_code, course_title, description, units, prerequisite) "
                "VALUES (?, ?, ?, ?, ?)",
                (c.course_code, c.course_title, c.description, c.units, c.prerequisite),
            )
        except sqlite3.IntegrityError as exc:
            if "courses.course_code" in str(exc):
                raise DuplicateCourseError(
                    f"A course with code '{c.course_code}' already exists. "
                    "Delete it first or save with replace=True."
                ) from exc
            raise
        course_id = cur.lastrowid

        conn.executemany(
            "INSERT INTO course_outcomes (course_id, number, domain, text, original_text) "
            "VALUES (?, ?, ?, ?, ?)",
            [(course_id, o.number, o.domain, o.text, o.text) for o in syllabus.course_outcomes],
        )
        for w in syllabus.weekly_schedule:
            cur = conn.execute(
                "INSERT INTO weekly_schedules (course_id, week, topic, description) "
                "VALUES (?, ?, ?, ?)",
                (course_id, w.week, w.topic, w.description),
            )
            schedule_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO lesson_outcomes (schedule_id, domain, text, original_text) "
                "VALUES (?, ?, ?, ?)",
                [(schedule_id, lo.domain, lo.text, lo.text) for lo in w.lesson_outcomes],
            )
    log.info("Saved course '%s' as id %d", c.course_code, course_id)
    return course_id


# --------------------------------------------------------------------- READ
def get_course(course_id: int, db_path=None) -> Dict:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM courses WHERE id = ?", (course_id,)).fetchone()
    if row is None:
        raise CourseNotFoundError(f"No course with id {course_id}.")
    return dict(row)


def get_course_outcomes(course_id: int, db_path=None) -> List[Dict]:
    with get_connection(db_path) as conn:
        _require_course(conn, course_id)
        rows = conn.execute(
            "SELECT id, number, domain, text, original_text FROM course_outcomes "
            "WHERE course_id = ? ORDER BY number",
            (course_id,),
        ).fetchall()
    return [_outcome_dict(r) for r in rows]


def get_weekly_schedule(course_id: int, db_path=None) -> List[Dict]:
    """Weeks in order, each with its lesson_outcomes list (K, S, A in saved order)."""
    with get_connection(db_path) as conn:
        _require_course(conn, course_id)
        weeks = conn.execute(
            "SELECT id, week, topic, description FROM weekly_schedules "
            "WHERE course_id = ? ORDER BY week",
            (course_id,),
        ).fetchall()
        lessons = conn.execute(
            "SELECT lo.id, lo.schedule_id, lo.domain, lo.text, lo.original_text "
            "FROM lesson_outcomes lo JOIN weekly_schedules ws ON ws.id = lo.schedule_id "
            "WHERE ws.course_id = ? ORDER BY lo.id",
            (course_id,),
        ).fetchall()

    schedule = []
    by_id = {}
    for w in weeks:
        entry = dict(w)
        entry["lesson_outcomes"] = []
        schedule.append(entry)
        by_id[entry["id"]] = entry
    for row in lessons:
        by_id[row["schedule_id"]]["lesson_outcomes"].append(_outcome_dict(row))
    return schedule


def get_full_syllabus(course_id: int, db_path=None) -> Dict:
    """Everything needed for export: course + outcomes + weekly schedule."""
    return {
        "course": get_course(course_id, db_path),
        "course_outcomes": get_course_outcomes(course_id, db_path),
        "weekly_schedule": get_weekly_schedule(course_id, db_path),
    }


def list_courses(db_path=None) -> List[Dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT c.id, c.course_code, c.course_title, c.units, c.created_at, "
            "  (SELECT COUNT(*) FROM course_outcomes WHERE course_id = c.id) AS clo_count, "
            "  (SELECT COUNT(*) FROM weekly_schedules WHERE course_id = c.id) AS week_count "
            "FROM courses c ORDER BY c.id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------- UPDATE
# Human-in-the-loop editing. Only `text` changes; `original_text` keeps the
# AI-generated wording. The new text must pass the SAME Pydantic rules.
def update_course_outcome(outcome_id: int, new_text: str, db_path=None) -> Dict:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT id, number, domain FROM course_outcomes WHERE id = ?", (outcome_id,)
        ).fetchone()
        if row is None:
            raise OutcomeNotFoundError(f"No course outcome with id {outcome_id}.")
        try:
            checked = CourseOutcomeSchema(
                number=row["number"], domain=row["domain"], text=new_text
            )
        except ValidationError as exc:
            raise InvalidEditError(f"Edit rejected: {_friendly_validation_message(exc)}") from exc
        conn.execute(
            "UPDATE course_outcomes SET text = ? WHERE id = ?", (checked.text, outcome_id)
        )
        updated = conn.execute(
            "SELECT id, number, domain, text, original_text FROM course_outcomes WHERE id = ?",
            (outcome_id,),
        ).fetchone()
    log.info("Course outcome %s edited", outcome_id)
    return _outcome_dict(updated)


def update_lesson_outcome(lesson_outcome_id: int, new_text: str, db_path=None) -> Dict:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT id, domain FROM lesson_outcomes WHERE id = ?", (lesson_outcome_id,)
        ).fetchone()
        if row is None:
            raise OutcomeNotFoundError(f"No lesson outcome with id {lesson_outcome_id}.")
        try:
            checked = LessonOutcomeSchema(domain=row["domain"], text=new_text)
        except ValidationError as exc:
            raise InvalidEditError(f"Edit rejected: {_friendly_validation_message(exc)}") from exc
        conn.execute(
            "UPDATE lesson_outcomes SET text = ? WHERE id = ?", (checked.text, lesson_outcome_id)
        )
        updated = conn.execute(
            "SELECT id, schedule_id, domain, text, original_text FROM lesson_outcomes WHERE id = ?",
            (lesson_outcome_id,),
        ).fetchone()
    log.info("Lesson outcome %s edited", lesson_outcome_id)
    return _outcome_dict(updated)


def reset_course_outcome(outcome_id: int, db_path=None) -> Dict:
    """Undo human edits: copy the AI-generated original back into `text`."""
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "UPDATE course_outcomes SET text = original_text WHERE id = ?", (outcome_id,)
        )
        if cur.rowcount == 0:
            raise OutcomeNotFoundError(f"No course outcome with id {outcome_id}.")
        row = conn.execute(
            "SELECT id, number, domain, text, original_text FROM course_outcomes WHERE id = ?",
            (outcome_id,),
        ).fetchone()
    return _outcome_dict(row)


def reset_lesson_outcome(lesson_outcome_id: int, db_path=None) -> Dict:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            "UPDATE lesson_outcomes SET text = original_text WHERE id = ?", (lesson_outcome_id,)
        )
        if cur.rowcount == 0:
            raise OutcomeNotFoundError(f"No lesson outcome with id {lesson_outcome_id}.")
        row = conn.execute(
            "SELECT id, schedule_id, domain, text, original_text FROM lesson_outcomes WHERE id = ?",
            (lesson_outcome_id,),
        ).fetchone()
    return _outcome_dict(row)


# ------------------------------------------------------------------- DELETE
def delete_course(course_id: int, db_path=None) -> None:
    """Delete a course; ON DELETE CASCADE removes all related rows."""
    with get_connection(db_path) as conn:
        cur = conn.execute("DELETE FROM courses WHERE id = ?", (course_id,))
        if cur.rowcount == 0:
            raise CourseNotFoundError(f"No course with id {course_id}.")
    log.info("Deleted course id %s (and its related records)", course_id)


# --------------------------------------------------------------------- CLI
def _print_outcome(label: str, o: Dict) -> None:
    flag = "  *EDITED*" if o["is_edited"] else ""
    print(f"  {label} (id {o['id']}) [{o['domain']}] {o['text']}{flag}")
    if o["is_edited"]:
        print(f"      AI original: {o['original_text']}")


def _cli() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="OBE database tools")
    parser.add_argument("--init", action="store_true", help="create the database tables")
    parser.add_argument("--load-sample", nargs="?", const="sample_validated_output.json",
                        metavar="FILE", help="validate a JSON file and save it (replaces same course code)")
    parser.add_argument("--list", action="store_true", help="list saved courses")
    parser.add_argument("--show", type=int, metavar="COURSE_ID", help="show a course's CLOs with ids")
    parser.add_argument("--week", type=int, metavar="N", help="with --show: also show week N's lesson outcomes")
    parser.add_argument("--edit-clo", nargs=2, metavar=("OUTCOME_ID", "TEXT"), help="edit a course outcome")
    parser.add_argument("--edit-lesson", nargs=2, metavar=("LESSON_ID", "TEXT"), help="edit a lesson outcome")
    parser.add_argument("--reset-clo", type=int, metavar="OUTCOME_ID", help="restore the AI wording of a CLO")
    parser.add_argument("--reset-lesson", type=int, metavar="LESSON_ID", help="restore the AI wording of a lesson outcome")
    parser.add_argument("--delete", type=int, metavar="ID", help="delete a course by id")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    acted = False
    try:
        if args.init:
            acted = True
            init_db()
            print(f"OK: database initialized at {DEFAULT_DB_PATH}")
        if args.load_sample:
            acted = True
            with open(args.load_sample, encoding="utf-8") as f:
                syllabus = SyllabusSchema.model_validate(json.load(f))
            course_id = create_course(syllabus, replace=True)
            print(f"OK: saved '{syllabus.course.course_title}' as course id {course_id}")
        if args.edit_clo:
            acted = True
            o = update_course_outcome(int(args.edit_clo[0]), args.edit_clo[1])
            print(f"OK: course outcome {o['id']} updated.")
            _print_outcome(f"CLO {o['number']}", o)
        if args.edit_lesson:
            acted = True
            o = update_lesson_outcome(int(args.edit_lesson[0]), args.edit_lesson[1])
            print(f"OK: lesson outcome {o['id']} updated.")
            _print_outcome("Lesson", o)
        if args.reset_clo is not None:
            acted = True
            o = reset_course_outcome(args.reset_clo)
            print(f"OK: course outcome {o['id']} restored to the AI wording.")
        if args.reset_lesson is not None:
            acted = True
            o = reset_lesson_outcome(args.reset_lesson)
            print(f"OK: lesson outcome {o['id']} restored to the AI wording.")
        if args.show is not None:
            acted = True
            course = get_course(args.show)
            print(f"Course {course['id']}: {course['course_code']} - {course['course_title']} "
                  f"({course['units']} units)")
            print("Course Learning Outcomes:")
            for o in get_course_outcomes(args.show):
                _print_outcome(f"CLO {o['number']}", o)
            if args.week is not None:
                weeks = [w for w in get_weekly_schedule(args.show) if w["week"] == args.week]
                if not weeks:
                    print(f"(no week {args.week} in this course)")
                for w in weeks:
                    print(f"Week {w['week']}: {w['topic']}")
                    for lo in w["lesson_outcomes"]:
                        _print_outcome("Lesson", lo)
        if args.delete is not None:
            acted = True
            delete_course(args.delete)
            print(f"OK: deleted course id {args.delete}")
        if args.list or not acted:
            courses = list_courses()
            if not courses:
                print("(no courses saved)")
            for c in courses:
                print(f"[{c['id']}] {c['course_code']} - {c['course_title']} "
                      f"({c['clo_count']} CLOs, {c['week_count']} weeks)")
    except DBError as exc:
        print(f"ERROR: {exc}")
        return 1
    except (OSError, ValueError) as exc:  # bad file, bad JSON, bad id, Pydantic ValidationError
        print(f"ERROR: could not process the input: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())