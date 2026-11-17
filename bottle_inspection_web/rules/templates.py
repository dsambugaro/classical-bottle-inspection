"""Templates of rule actions (REQ-IN7): Jinja2 SandboxedEnvironment, output limited to 64 KB."""

from __future__ import annotations

import json
from typing import Any

from jinja2 import StrictUndefined, TemplateError
from jinja2.sandbox import SandboxedEnvironment

MAX_OUTPUT: int = 64 * 1024
"""Rendered output limit (bytes)."""

VARIABLES: list[str] = [
    "analysis.predicted_class",
    "analysis.verdict",
    "analysis.confidence",
    "analysis.scores",
    "analysis.warnings",
    "analysis.timings_ms.total",
    "analysis.image.url",
    "analysis.ground_truth",
    "analysis.created_at",
    "inspection.id",
    "inspection.source",
    "inspection.original_name",
    "profile.id",
    "profile.name",
    "profile.algorithm",
    "profile.version",
    "rule.name",
    "rule.severity",
    "window_stats",
    "verdict",
    "app_url",
]
"""Variables offered by the editor's variable picker."""


class TemplateRenderError(Exception):
    """A template failed to render (syntax, undefined variable, sandbox, size)."""


def _tojson(value: Any) -> str:
    """JSON filter (also available as `tojson`)."""
    return json.dumps(value, ensure_ascii=False, default=str)


_env = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)
_env.filters["tojson"] = _tojson


def render(template: str, context: dict[str, Any]) -> str:
    """Renders a template in the sandbox; raises TemplateRenderError."""
    try:
        output = _env.from_string(template).render(**context)
    except TemplateError as error:
        raise TemplateRenderError(f"{type(error).__name__}: {error}") from error
    except Exception as error:  # noqa: BLE001 - sandbox violations raise various errors
        raise TemplateRenderError(f"{type(error).__name__}: {error}") from error
    if len(output.encode("utf-8")) > MAX_OUTPUT:
        raise TemplateRenderError(f"output larger than {MAX_OUTPUT} bytes")
    return output


def render_value(value: Any, context: dict[str, Any]) -> Any:
    """Renders strings inside dicts/lists recursively (for JSON bodies and headers)."""
    if isinstance(value, str):
        return render(value, context)
    if isinstance(value, dict):
        return {render(str(k), context): render_value(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [render_value(v, context) for v in value]
    return value


def sample_context(app_url: str = "http://127.0.0.1:8000") -> dict[str, Any]:
    """A sample context for previews when there is no analysis yet."""
    from ..classes import CLASS_NAMES

    scores = {name: 0.0 for name in CLASS_NAMES}
    scores["defect_no_cap"] = 0.92
    analysis = {
        "analysis_id": 0,
        "inspection_id": 0,
        "profile": {"id": 1, "name": "SVM (results)", "version": 1, "algorithm": "svm"},
        "predicted_class": "defect_no_cap",
        "verdict": "reject",
        "confidence": 0.83,
        "scores": scores,
        "warnings": [],
        "timings_ms": {"total": 33.8},
        "image": {"id": None, "url": None},
        "ground_truth": None,
        "created_at": "2026-11-09T14:03:22.120Z",
    }
    return {
        "analysis": analysis,
        "inspection": {"id": 0, "source": "upload", "original_name": "sample.jpg"},
        "profile": analysis["profile"],
        "rule": {"id": 0, "name": "Sample rule", "severity": "warning"},
        "window_stats": {name: 0 for name in CLASS_NAMES} | {"defect_no_cap": 4},
        "verdict": "reject",
        "app_url": app_url,
    }
