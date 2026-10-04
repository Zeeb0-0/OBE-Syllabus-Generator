"""
export_engine.py
Renders a syllabus stored in SQLite into HTML using Jinja2.

Pipeline stage:  SQLite (including human edits) -> Jinja2 template -> HTML file

The official CCS template has not been supplied yet, so the layout is a
clearly-marked draft. Everything institution-specific lives in DEFAULT_CONFIG
and in templates/uphsd_ccs_template.html, so both can be replaced later.
"""
import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from jinja2 import (
    Environment,
    FileSystemLoader,
    StrictUndefined,
    TemplateError,
    TemplateNotFound,
    select_autoescape,
)
from jinja2.exceptions import TemplateSyntaxError, UndefinedError
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

import db_manager

log = logging.getLogger("obe.export")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
DEFAULT_TEMPLATE = "uphsd_ccs_template.html"
DEFAULT_OUTPUT = BASE_DIR / "output" / "generated_syllabus.html"


class ExportError(Exception):
    """Raised for any export problem. Messages are written for a student developer."""


# ------------------------------------------------------------------ config
# Everything here is an ASSUMPTION until the official template is supplied.
DEFAULT_CONFIG = {
    "template_is_official": False,   # set True once the official template is in use
    "institution_name": "",          # left empty on purpose: not invented
    "department_name": "",
    "document_title": "Course Syllabus",
    "labels": {
        "course_info": "Course Information",
        "description": "Course Description",
        "pvm": "PVM",
        "outcomes": "Course Learning Outcomes",
        "schedule": "Weekly Schedule",
    },
    "domain_labels": {"K": "Knowledge", "S": "Skills", "A": "Attitudes"},
    "assumptions": [
        "Page layout, section order, and section headings are generic; the official CCS template has not been supplied.",
        "No institution name, logo, or department header is set (left empty on purpose).",
        "K, S, and A are expanded as Knowledge, Skills, and Attitudes (the standard OBE reading).",
        "PVM and grading matrix sections appear only when content is supplied; their layout is a generic assumption.",
    ],
}


def _merge_config(override: Optional[dict]) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if override:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg


# ------------------------------------------------------------ optional extras
class PVMBlock(BaseModel):
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)


class GradingMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = "Grading System"
    columns: List[str] = Field(min_length=1)
    rows: List[List[str]] = Field(min_length=1)

    @field_validator("rows", mode="before")
    @classmethod
    def stringify_cells(cls, value):
        # allow numbers in the JSON file (e.g. 30) by converting every cell to text
        if isinstance(value, list):
            return [[str(c) for c in row] if isinstance(row, list) else row for row in value]
        return value

    @model_validator(mode="after")
    def rows_match_columns(self):
        for i, row in enumerate(self.rows, start=1):
            if len(row) != len(self.columns):
                raise ValueError(
                    f"Row {i} has {len(row)} cells but there are {len(self.columns)} columns"
                )
        return self


class ExportExtras(BaseModel):
    """Optional, export-time content. Both parts are only rendered when supplied."""

    model_config = ConfigDict(extra="forbid")

    pvm: Optional[List[PVMBlock]] = None
    grading_matrix: Optional[GradingMatrix] = None


def _format_errors(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in e['loc']) or '(file)'}: {e['msg']}"
        for e in exc.errors(include_url=False, include_input=False)
    )


def load_extras(path) -> ExportExtras:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return ExportExtras.model_validate(data)
    except OSError as exc:
        raise ExportError(f"Could not read the extras file '{path}': {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ExportError(f"The extras file is not valid JSON: {exc}") from exc
    except ValidationError as exc:
        raise ExportError(f"The extras file has invalid content: {_format_errors(exc)}") from exc


# ----------------------------------------------------------------- rendering
def build_context(
    course_id: int, *, db_path=None, extras=None, config=None, show_edit_markers=False
) -> dict:
    data = db_manager.get_full_syllabus(course_id, db_path)  # raises CourseNotFoundError
    cfg = _merge_config(config)

    if isinstance(extras, ExportExtras):
        ex = extras
    else:
        try:
            ex = ExportExtras.model_validate(extras or {})
        except ValidationError as exc:
            raise ExportError(f"Invalid extras: {_format_errors(exc)}") from exc

    edited_count = sum(1 for o in data["course_outcomes"] if o["is_edited"]) + sum(
        1 for w in data["weekly_schedule"] for lo in w["lesson_outcomes"] if lo["is_edited"]
    )
    log.info("Exporting course %s (%d edited outcome(s))", course_id, edited_count)

    return {
        "course": data["course"],
        "course_outcomes": data["course_outcomes"],
        "weekly_schedule": data["weekly_schedule"],
        "config": cfg,
        "labels": cfg["labels"],
        "domain_labels": cfg["domain_labels"],
        "pvm": ex.pvm or None,
        "grading_matrix": ex.grading_matrix,
        "show_edit_markers": show_edit_markers,
        "edited_count": edited_count,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def render_syllabus(
    course_id: int,
    *,
    template_name: str = DEFAULT_TEMPLATE,
    template_dir=None,
    db_path=None,
    extras=None,
    config=None,
    show_edit_markers: bool = False,
) -> str:
    """Return the final HTML as a string."""
    context = build_context(
        course_id, db_path=db_path, extras=extras, config=config,
        show_edit_markers=show_edit_markers,
    )
    env = Environment(
        loader=FileSystemLoader(str(template_dir or TEMPLATE_DIR)),
        autoescape=select_autoescape(enabled_extensions=("html", "htm", "xml"), default_for_string=True),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    try:
        return env.get_template(template_name).render(**context)
    except TemplateNotFound as exc:
        raise ExportError(
            f"Template '{template_name}' was not found in {template_dir or TEMPLATE_DIR}."
        ) from exc
    except TemplateSyntaxError as exc:
        raise ExportError(
            f"Template syntax error in '{exc.name or template_name}' at line {exc.lineno}: {exc.message}"
        ) from exc
    except UndefinedError as exc:
        raise ExportError(
            f"The template uses a value that was not supplied: {exc.message}"
        ) from exc
    except TemplateError as exc:
        raise ExportError(f"Template error: {exc}") from exc


def export_syllabus(course_id: int, output_path=None, **render_kwargs) -> Path:
    """Render and write the HTML file. Returns the path written."""
    html = render_syllabus(course_id, **render_kwargs)
    path = Path(output_path) if output_path else DEFAULT_OUTPUT
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
    except OSError as exc:
        raise ExportError(
            f"Could not write the HTML file to '{path}': {exc}. Is it open in another program?"
        ) from exc
    log.info("Syllabus exported to %s", path)
    return path


# --------------------------------------------------------------------- CLI
def _cli() -> int:
    import argparse
    import webbrowser

    parser = argparse.ArgumentParser(description="Export a saved syllabus to HTML (Jinja2)")
    parser.add_argument("course_id", type=int, help="course id (see: python db_manager.py --list)")
    parser.add_argument("--out", help="output file (default: output/generated_syllabus.html)")
    parser.add_argument("--extras", metavar="FILE", help="JSON file with optional PVM and grading matrix")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE, help="template file name in templates/")
    parser.add_argument("--mark-edits", action="store_true", help="flag human-edited outcomes")
    parser.add_argument("--open", action="store_true", help="open the result in your browser")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        extras = load_extras(args.extras) if args.extras else None
        path = export_syllabus(
            args.course_id, args.out, template_name=args.template,
            extras=extras, show_edit_markers=args.mark_edits,
        )
    except (db_manager.DBError, ExportError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"OK: exported course {args.course_id} to {path.resolve()}")
    print(f"Open: {path.resolve().as_uri()}")
    if args.open:
        webbrowser.open(path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())