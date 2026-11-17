"""Rule conditions (REQ-IN5). Each profile's analysis is evaluated on its own (REQ-IN4), so
window conditions use the previous analyses of the same profile, ordered by time."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

from ..classes import CLASS_NAMES

CONDITION_TYPES: dict[str, dict[str, Any]] = {
    "class_is": {"classes": []},
    "verdict_is": {"verdict": "reject"},
    "any_defect": {},
    "repeated_class": {"class": "any", "n": 4, "m": 10},
    "consecutive_rejects": {"n": 3},
    "confidence_below": {"value": 0.5},
    "processing_time_above": {"ms": 80.0},
    "reject_rate_above": {"percent": 10.0, "window_minutes": 10, "min_samples": 20},
    "warning_present": {"codes": []},
}
"""v1 condition types and their default parameters (the CLI alert is n=4, m=10)."""

WARNING_CODES: list[str] = [
    "low_foreground",
    "resolution_mismatch",
    "modified_parameters",
    "low_confidence",
]


@dataclass(frozen=True)
class AnalysisFacts:
    """What conditions look at in one analysis."""

    id: int
    profile_id: int
    predicted_class: str
    verdict: str
    confidence: float
    total_ms: float
    warnings: tuple[str, ...]
    created_at: datetime


class History(Protocol):
    """Previous analyses of a profile, up to and including the current one."""

    def last(self, facts: AnalysisFacts, count: int) -> list[AnalysisFacts]:
        """The last `count` analyses of the profile (newest first, current included)."""
        ...

    def since(self, facts: AnalysisFacts, start: datetime) -> list[AnalysisFacts]:
        """Analyses of the profile from `start` up to the current one."""
        ...


@dataclass
class ConditionSpec:
    """One condition of a rule."""

    type: str
    params: dict[str, Any]
    negate: bool = False


@dataclass
class EvalState:
    """Hysteresis state of rate conditions, keyed by (rule, profile, condition position)."""

    disarmed: dict[tuple[int, int, int], datetime] = field(default_factory=dict)


class ConditionError(ValueError):
    """Invalid condition parameters; `code` is translated by the frontend."""

    def __init__(self, code: str, message: str) -> None:
        """Error with a translatable code."""
        super().__init__(message)
        self.code = code


def validate_condition(kind: str, params: dict[str, Any]) -> dict[str, Any]:
    """Fills defaults and checks the parameters of a condition."""
    if kind not in CONDITION_TYPES:
        raise ConditionError("rule.unknown_condition", f"unknown condition {kind}")
    merged = {**CONDITION_TYPES[kind], **params}
    unknown = set(merged) - set(CONDITION_TYPES[kind])
    if unknown:
        raise ConditionError("rule.invalid_condition", f"unknown parameters {sorted(unknown)}")
    if kind == "class_is":
        classes = merged["classes"]
        if not classes or any(c not in CLASS_NAMES for c in classes):
            raise ConditionError("rule.invalid_condition", "choose one or more of the 7 classes")
        merged["classes"] = [c for c in CLASS_NAMES if c in classes]
    if kind == "verdict_is" and merged["verdict"] not in ("pass", "reject", "review"):
        raise ConditionError("rule.invalid_condition", "invalid verdict")
    if kind == "repeated_class":
        if merged["class"] != "any" and merged["class"] not in CLASS_NAMES:
            raise ConditionError("rule.invalid_condition", "invalid class")
        n, m = int(merged["n"]), int(merged["m"])
        if not 1 <= n <= m <= 1000:
            raise ConditionError("rule.invalid_condition", "need 1 <= n <= m <= 1000")
        merged["n"], merged["m"] = n, m
    if kind == "consecutive_rejects" and not 1 <= int(merged["n"]) <= 1000:
        raise ConditionError("rule.invalid_condition", "n must be 1-1000")
    if kind == "confidence_below" and not 0 <= float(merged["value"]) <= 1:
        raise ConditionError("rule.invalid_condition", "value must be 0-1")
    if kind == "processing_time_above" and float(merged["ms"]) <= 0:
        raise ConditionError("rule.invalid_condition", "ms must be positive")
    if kind == "reject_rate_above":
        if not 0 < float(merged["percent"]) <= 100 or int(merged["window_minutes"]) < 1:
            raise ConditionError("rule.invalid_condition", "invalid rate parameters")
        merged["min_samples"] = max(1, int(merged["min_samples"]))
    if kind == "warning_present":
        codes = merged["codes"]
        if not codes or any(c not in WARNING_CODES for c in codes):
            raise ConditionError("rule.invalid_condition", "choose one or more warning codes")
    return merged


def is_reject_class(name: str) -> bool:
    """Every class except bottle_no_defect is a defect class."""
    return name != "bottle_no_defect"


def evaluate_one(
    spec: ConditionSpec,
    facts: AnalysisFacts,
    history: History,
    state: EvalState,
    key: tuple[int, int, int],
) -> tuple[bool, dict[str, Any]]:
    """(matches, details) of one condition, before `negate`."""
    p = spec.params
    kind = spec.type
    if kind == "class_is":
        return facts.predicted_class in p["classes"], {}
    if kind == "verdict_is":
        return facts.verdict == p["verdict"], {}
    if kind == "any_defect":
        return facts.verdict == "reject", {}
    if kind == "repeated_class":
        window = history.last(facts, int(p["m"]))
        counts = Counter(a.predicted_class for a in window if is_reject_class(a.predicted_class))
        if p["class"] == "any":
            hit = facts.predicted_class if is_reject_class(facts.predicted_class) else None
            count = counts.get(hit, 0) if hit else 0
        else:
            hit = p["class"]
            count = counts.get(hit, 0) if facts.predicted_class == hit else 0
        return count >= int(p["n"]), {"class": hit, "count": count, "window": len(window)}
    if kind == "consecutive_rejects":
        window = history.last(facts, int(p["n"]))
        ok = len(window) >= int(p["n"]) and all(a.verdict == "reject" for a in window)
        return ok, {"count": len(window)}
    if kind == "confidence_below":
        return facts.confidence < float(p["value"]), {}
    if kind == "processing_time_above":
        return facts.total_ms > float(p["ms"]), {}
    if kind == "reject_rate_above":
        window = history.since(
            facts, facts.created_at - timedelta(minutes=int(p["window_minutes"]))
        )
        rate = 100.0 * sum(a.verdict == "reject" for a in window) / len(window) if window else 0.0
        above = len(window) >= int(p["min_samples"]) and rate > float(p["percent"])
        details = {"rate": rate, "samples": len(window)}
        if key in state.disarmed:
            # Hysteresis: re-arm only when the rate falls back below the threshold.
            if rate <= float(p["percent"]):
                del state.disarmed[key]
            return False, details
        if above:
            state.disarmed[key] = facts.created_at
        return above, details
    if kind == "warning_present":
        return any(code in facts.warnings for code in p["codes"]), {}
    raise ConditionError("rule.unknown_condition", kind)


def evaluate(
    rule_id: int,
    logic: str,
    specs: list[ConditionSpec],
    facts: AnalysisFacts,
    history: History,
    state: EvalState,
) -> tuple[bool, list[dict[str, Any]]]:
    """Evaluates every condition (all/any); returns (fired, per-condition details)."""
    results = []
    for position, spec in enumerate(specs):
        matched, details = evaluate_one(
            spec, facts, history, state, (rule_id, facts.profile_id, position)
        )
        if spec.negate:
            matched = not matched
        results.append({"type": spec.type, "matched": matched, **details})
    if not results:
        return False, results
    values = [r["matched"] for r in results]
    fired = all(values) if logic == "all" else any(values)
    return fired, results
