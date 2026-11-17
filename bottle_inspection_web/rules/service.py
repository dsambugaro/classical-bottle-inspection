"""Rules engine and integrations API (§12): targets, rules, evaluation after each analysis,
cooldown/dedupe, the action queue and workers, dry run, test, template preview, the
execution log and the daily retention job."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, FastAPI
from pydantic import BaseModel, Field
from sqlalchemy import and_, delete, func, or_, select, update

from ..classes import CLASS_NAMES, ordered_counts
from ..context import AppContext, get_ctx
from ..db.models import (
    ActionExecution,
    Analysis,
    Image,
    Inspection,
    Rule,
    RuleAction,
    RuleCondition,
    Target,
    utcnow,
)
from ..errors import ApiError
from ..inspections import AnalysisContext
from ..notifications import notify
from ..profiles import iso
from . import actions as act
from .conditions import (
    CONDITION_TYPES,
    WARNING_CODES,
    AnalysisFacts,
    ConditionError,
    ConditionSpec,
    EvalState,
    evaluate,
    validate_condition,
)
from .secrets import MASK, SecretStore, redact, secret_name
from .templates import VARIABLES, TemplateRenderError, render, sample_context

log = logging.getLogger(__name__)

QUEUE_SIZE: int = 1000
WORKERS: int = 4
RETENTION_INTERVAL_S: float = 24 * 3600


def naive(value: datetime) -> datetime:
    """UTC-naive datetime (as stored by SQLite)."""
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value


# ----------------------------------------------------------------- history from the DB


class DbHistory:
    """Previous analyses of a profile, read from the DB (newest first)."""

    def __init__(self, ctx: AppContext) -> None:
        """Uses the app database."""
        self.ctx = ctx

    @staticmethod
    def _facts(row: Analysis) -> AnalysisFacts:
        """AnalysisFacts of a row."""
        return AnalysisFacts(
            row.id,
            row.profile_id,
            row.predicted_class,
            row.verdict,
            row.confidence,
            row.total_ms,
            tuple(row.warnings_json or []),
            row.created_at,
        )

    def get(self, analysis_id: int) -> AnalysisFacts | None:
        """Facts of a stored analysis, with the exact stored timestamp."""
        with self.ctx.db.session() as session:
            row = session.get(Analysis, analysis_id)
            return self._facts(row) if row is not None else None

    def _until(self, facts: AnalysisFacts) -> Any:
        """Rows of the same profile up to and including the current analysis."""
        at = naive(facts.created_at)
        return and_(
            Analysis.profile_id == facts.profile_id,
            or_(Analysis.created_at < at, and_(Analysis.created_at == at, Analysis.id <= facts.id)),
        )

    def last(self, facts: AnalysisFacts, count: int) -> list[AnalysisFacts]:
        """The last `count` analyses of the profile, current included."""
        with self.ctx.db.session() as session:
            rows = session.scalars(
                select(Analysis)
                .where(self._until(facts))
                .order_by(Analysis.created_at.desc(), Analysis.id.desc())
                .limit(count)
            ).all()
            return [self._facts(r) for r in rows]

    def since(self, facts: AnalysisFacts, start: datetime) -> list[AnalysisFacts]:
        """Analyses of the profile from `start` up to the current one."""
        with self.ctx.db.session() as session:
            rows = session.scalars(
                select(Analysis)
                .where(self._until(facts), Analysis.created_at >= naive(start))
                .order_by(Analysis.created_at.desc())
            ).all()
            return [self._facts(r) for r in rows]


def facts_of(result: dict[str, Any]) -> AnalysisFacts:
    """AnalysisFacts of an AnalysisResult JSON."""
    created = datetime.fromisoformat(str(result["created_at"]).replace("Z", "+00:00"))
    return AnalysisFacts(
        int(result["analysis_id"]),
        int(result["profile"]["id"]),
        str(result["predicted_class"]),
        str(result["verdict"]),
        float(result["confidence"]),
        float(result["timings_ms"]["total"]),
        tuple(result.get("warnings") or []),
        naive(created),
    )


# ----------------------------------------------------------------- rule snapshots


@dataclass
class ActionSnapshot:
    """An action as loaded from the DB."""

    id: int
    type: str
    target_id: int | None
    params: dict[str, Any]
    enabled: bool
    timeout_s: float | None
    max_retries: int


@dataclass
class RuleSnapshot:
    """A rule with its conditions and actions, as loaded from the DB."""

    id: int
    name: str
    severity: str
    enabled: bool
    scope: dict[str, Any]
    trigger: str
    logic: str
    cooldown_s: int
    dedupe_key: str
    run_mode: str
    conditions: list[ConditionSpec]
    actions: list[ActionSnapshot]

    def in_scope(self, profile_id: int) -> bool:
        """True when the rule applies to the profile."""
        profiles = self.scope.get("profiles", "all")
        return profiles == "all" or profile_id in profiles

    def dedupe_value(self, facts: AnalysisFacts) -> str:
        """The dedupe key value of a firing."""
        if self.dedupe_key == "rule":
            return f"rule:{self.id}"
        if self.dedupe_key == "rule+profile":
            return f"rule:{self.id}:profile:{facts.profile_id}"
        return f"rule:{self.id}:class:{facts.predicted_class}"


@dataclass
class RuleRun:
    """One firing of a rule waiting in the queue."""

    rule: RuleSnapshot
    executions: list[tuple[int, ActionSnapshot]]
    context: dict[str, Any]
    is_test: bool = False
    image_path: Path | None = None
    image_bytes: bytes | None = None
    done: asyncio.Future[list[dict[str, Any]]] | None = None


@dataclass
class Suppression:
    """Open suppression window of a dedupe key (one log row with a counter)."""

    last_fired: float
    execution_id: int | None = None


class RulesService:
    """Evaluates rules after every analysis and runs their actions (ctx.extras["rules"])."""

    def __init__(self, ctx: AppContext) -> None:
        """Wires the hook; workers start with the app."""
        self.ctx = ctx
        self.secrets = SecretStore(ctx.config.paths.secrets)
        self.history = DbHistory(ctx)
        self.state = EvalState()
        self._rules: list[RuleSnapshot] | None = None
        self._rules_lock = threading.Lock()
        self._fired: dict[str, Suppression] = {}
        self._fired_lock = threading.Lock()
        self._running: set[int] = set()
        self._rule_locks: dict[int, asyncio.Lock] = {}
        self.queue: asyncio.Queue[RuleRun] | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.tasks: list[asyncio.Task[None]] = []
        self.deps = act.ActionDeps(
            notify=lambda level, title, message: notify(ctx.db, ctx.events, level, title, message),
            publish_event=ctx.events.publish,
            data_root=ctx.config.paths.root,
        )
        ctx.inspections.hooks.append(self.on_analysis)

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Starts the action workers and the daily retention job."""
        self.loop = asyncio.get_running_loop()
        self.queue = asyncio.Queue(QUEUE_SIZE)
        if not self.ctx.config.start_workers:
            return
        self.tasks = [asyncio.create_task(self._worker(i)) for i in range(WORKERS)]
        self.tasks.append(asyncio.create_task(self._retention_loop()))

    async def stop(self) -> None:
        """Stops the workers."""
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self.tasks = []

    def queue_size(self) -> int:
        """Runs waiting in the queue."""
        return self.queue.qsize() if self.queue is not None else 0

    # ------------------------------------------------------------- rules cache

    def invalidate(self) -> None:
        """Reloads the rules on the next analysis."""
        with self._rules_lock:
            self._rules = None

    def rules(self) -> list[RuleSnapshot]:
        """Every rule with its conditions and actions (cached)."""
        with self._rules_lock:
            if self._rules is None:
                self._rules = self._load_rules()
            return self._rules

    def _load_rules(self, only_id: int | None = None) -> list[RuleSnapshot]:
        """Reads rules from the DB."""
        with self.ctx.db.session() as session:
            query = select(Rule).order_by(Rule.id)
            if only_id is not None:
                query = query.where(Rule.id == only_id)
            result = []
            for rule in session.scalars(query):
                conditions = session.scalars(
                    select(RuleCondition)
                    .where(RuleCondition.rule_id == rule.id)
                    .order_by(RuleCondition.position)
                ).all()
                actions = session.scalars(
                    select(RuleAction)
                    .where(RuleAction.rule_id == rule.id)
                    .order_by(RuleAction.position)
                ).all()
                result.append(
                    RuleSnapshot(
                        rule.id,
                        rule.name,
                        rule.severity,
                        rule.enabled,
                        dict(rule.scope_json or {}),
                        rule.trigger,
                        rule.condition_logic,
                        rule.cooldown_s,
                        rule.dedupe_key,
                        rule.run_mode,
                        [
                            ConditionSpec(c.type, dict(c.params_json or {}), c.negate)
                            for c in conditions
                        ],
                        [
                            ActionSnapshot(
                                a.id,
                                a.type,
                                a.target_id,
                                dict(a.params_json or {}),
                                a.enabled,
                                a.timeout_s,
                                a.max_retries,
                            )
                            for a in actions
                        ],
                    )
                )
            return result

    # ------------------------------------------------------------- evaluation

    def on_analysis(self, analysis: AnalysisContext) -> None:
        """Analysis hook (thread pool): evaluates every rule in scope."""
        rules = [r for r in self.rules() if r.enabled and r.trigger == "analysis_completed"]
        if not rules:
            return
        # The JSON timestamp is rounded to milliseconds; windows need the stored one.
        facts = self.history.get(analysis.analysis_id) or facts_of(analysis.result)
        for rule in rules:
            if not rule.in_scope(facts.profile_id):
                continue
            try:
                fired, details = evaluate(
                    rule.id, rule.logic, rule.conditions, facts, self.history, self.state
                )
            except ConditionError:
                log.exception("rule %s has an invalid condition", rule.id)
                continue
            if fired:
                self.fire(rule, facts, analysis, details)

    def window_stats(self, facts: AnalysisFacts, rule: RuleSnapshot) -> dict[str, int]:
        """Counts per class (all 7) in the rule's window (repeated_class m, default 10)."""
        size = next(
            (int(c.params.get("m", 10)) for c in rule.conditions if c.type == "repeated_class"), 10
        )
        return ordered_counts(
            dict(_count(a.predicted_class for a in self.history.last(facts, size)))
        )

    def context_for(
        self, rule: RuleSnapshot, result: dict[str, Any], facts: AnalysisFacts | None
    ) -> dict[str, Any]:
        """Template context (REQ-IN7)."""
        with self.ctx.db.session() as session:
            inspection = (
                session.get(Inspection, result["inspection_id"])
                if result.get("inspection_id")
                else None
            )
            inspection_json = (
                {
                    "id": inspection.id,
                    "source": inspection.source,
                    "original_name": inspection.original_name,
                    "session_id": inspection.session_id,
                    "camera": inspection.camera_name,
                }
                if inspection
                else {"id": result.get("inspection_id")}
            )
        return {
            "analysis": result,
            "inspection": inspection_json,
            "profile": result.get("profile", {}),
            "rule": {"id": rule.id, "name": rule.name, "severity": rule.severity},
            "window_stats": (
                self.window_stats(facts, rule) if facts else {c: 0 for c in CLASS_NAMES}
            ),
            "verdict": result.get("verdict"),
            "app_url": f"http://{self.ctx.config.host}:{self.ctx.config.port}",
        }

    def fire(
        self,
        rule: RuleSnapshot,
        facts: AnalysisFacts,
        analysis: AnalysisContext,
        details: list[dict[str, Any]],
    ) -> None:
        """Cooldown/dedupe, then queues the rule's actions (or logs the suppression)."""
        key = rule.dedupe_value(facts)
        now = time.monotonic()
        with self._fired_lock:
            window = self._fired.get(key)
            if window is not None and now - window.last_fired < rule.cooldown_s:
                window.execution_id = self._log_suppressed(rule, analysis, key, window.execution_id)
                return
            if rule.run_mode == "single" and rule.id in self._running:
                log.info("rule %s is still running; firing ignored (run_mode single)", rule.id)
                return
            self._fired[key] = Suppression(now)
        context = self.context_for(rule, analysis.result, facts)
        context["conditions"] = details
        run = RuleRun(rule, [], context)
        image = analysis.result.get("image") or {}
        if image.get("id"):
            with self.ctx.db.session() as session:
                row = session.get(Image, image["id"])
                run.image_path = Path(row.path) if row else None
        else:
            run.image_bytes = self.ctx.inspections.frame(analysis.inspection_id)
        run.executions = self._create_executions(
            rule, analysis.analysis_id, analysis.inspection_id, key
        )
        self.ctx.events.publish(
            "rule.fired",
            {
                "rule_id": rule.id,
                "rule": rule.name,
                "severity": rule.severity,
                "profile": facts.profile_id,
                "predicted_class": facts.predicted_class,
                "analysis_id": facts.id,
            },
        )
        if rule.severity == "critical":
            notify(
                self.ctx.db,
                self.ctx.events,
                "critical",
                f"{rule.name}",
                f"{facts.predicted_class}",
                code="rule.fired",
                params={"rule": rule.name},
            )
        self._enqueue(run)

    def _create_executions(
        self,
        rule: RuleSnapshot,
        analysis_id: int | None,
        inspection_id: int | None,
        key: str,
        is_test: bool = False,
    ) -> list[tuple[int, ActionSnapshot]]:
        """One `queued` log row per enabled action (each with its own event UUID)."""
        created = []
        with self.ctx.db.session() as session:
            for action in rule.actions:
                if not action.enabled:
                    continue
                row = ActionExecution(
                    rule_id=rule.id,
                    action_id=action.id,
                    target_id=action.target_id,
                    analysis_id=analysis_id,
                    inspection_id=inspection_id,
                    event_uuid=str(uuid.uuid4()),
                    status="queued",
                    attempt=0,
                    is_test=is_test,
                    action_type=action.type,
                    dedupe_value=key,
                    created_at=utcnow(),
                )
                session.add(row)
                session.flush()
                created.append((row.id, action))
        return created

    def _log_suppressed(
        self, rule: RuleSnapshot, analysis: AnalysisContext, key: str, execution_id: int | None
    ) -> int:
        """REQ-IN10: one `skipped_cooldown` row per suppression window, with a counter."""
        with self.ctx.db.session() as session:
            row = session.get(ActionExecution, execution_id) if execution_id else None
            if row is None:
                row = ActionExecution(
                    rule_id=rule.id,
                    analysis_id=analysis.analysis_id,
                    inspection_id=analysis.inspection_id,
                    event_uuid=str(uuid.uuid4()),
                    status="skipped_cooldown",
                    attempt=0,
                    suppressed_count=0,
                    dedupe_value=key,
                    created_at=utcnow(),
                )
                session.add(row)
            row.suppressed_count += 1
            row.finished_at = utcnow()
            session.flush()
            return row.id

    def _enqueue(self, run: RuleRun) -> None:
        """Puts a run in the queue without ever blocking the inspection (thread-safe)."""
        if self.loop is None or self.queue is None:
            self._mark(
                [e for e, _ in run.executions], "dropped_queue_full", "action queue not running"
            )
            return
        queue = self.queue

        def put() -> None:
            """Runs on the event loop."""
            try:
                queue.put_nowait(run)
            except asyncio.QueueFull:
                self._mark(
                    [e for e, _ in run.executions], "dropped_queue_full", "action queue full"
                )
                notify(
                    self.ctx.db,
                    self.ctx.events,
                    "warning",
                    "Action queue full",
                    run.rule.name,
                    code="action.dropped",
                )
                if run.done is not None and not run.done.done():
                    run.done.set_result([])

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self.loop:
            put()
        else:
            self.loop.call_soon_threadsafe(put)

    def _mark(self, ids: list[int], status: str, error: str | None = None) -> None:
        """Sets the status of several log rows."""
        if not ids:
            return
        with self.ctx.db.session() as session:
            session.execute(
                update(ActionExecution)
                .where(ActionExecution.id.in_(ids))
                .values(status=status, error=error, finished_at=utcnow())
            )

    # ------------------------------------------------------------- workers

    async def _worker(self, index: int) -> None:
        """Consumes runs from the queue (run_mode: single, queued, parallel)."""
        assert self.queue is not None
        while True:
            run = await self.queue.get()
            try:
                lock = None
                if run.rule.run_mode == "queued":
                    lock = self._rule_locks.setdefault(run.rule.id, asyncio.Lock())
                if lock is not None:
                    async with lock:
                        results = await self._execute_run(run)
                else:
                    results = await self._execute_run(run)
                if run.done is not None and not run.done.done():
                    run.done.set_result(results)
            except Exception as error:  # noqa: BLE001 - a failing action never stops the worker
                log.exception("action worker %s failed", index)
                if run.done is not None and not run.done.done():
                    run.done.set_exception(error)
            finally:
                self.queue.task_done()

    def _target(self, target_id: int | None) -> tuple[str | None, dict[str, Any], dict[str, str]]:
        """(type, config, secrets) of a target."""
        if target_id is None:
            return None, {}, {}
        with self.ctx.db.session() as session:
            row = session.get(Target, target_id)
            if row is None:
                return None, {}, {}
            secrets = {
                name: value
                for name in act.TARGET_SECRETS.get(row.type, [])
                if (value := self.secrets.get(secret_name(row.id, name))) is not None
            }
            if not row.enabled:
                raise act.PermanentError(f"target '{row.name}' is disabled")
            return row.type, dict(row.config_json or {}), secrets

    async def _execute_run(self, run: RuleRun) -> list[dict[str, Any]]:
        """Executes the actions of a firing in order."""
        self._running.add(run.rule.id)
        results = []
        try:
            for execution_id, action in run.executions:
                results.append(await self._execute_action(run, execution_id, action))
        finally:
            self._running.discard(run.rule.id)
        return results

    async def _execute_action(
        self, run: RuleRun, execution_id: int, action: ActionSnapshot
    ) -> dict[str, Any]:
        """One action with retries; updates its log row and publishes `action.finished`."""
        with self.ctx.db.session() as session:
            row = session.get(ActionExecution, execution_id)
            event_uuid = row.event_uuid if row else str(uuid.uuid4())
            if row is not None:
                row.status = "running"
        secrets: dict[str, str] = {}
        try:
            target_type, config, secrets = self._target(action.target_id)
        except act.PermanentError as error:
            outcome = act.Outcome("failed", 1, "", None, str(error), "", 0.0)
        else:
            expected = act.ACTION_TARGET.get(action.type)
            if expected and target_type != expected:
                outcome = act.Outcome(
                    "failed", 1, "", None, f"this action needs a {expected} target", "", 0.0
                )
            else:
                request = act.ActionRequest(
                    execution_id,
                    event_uuid,
                    action.type,
                    action.params,
                    target_type,
                    config,
                    secrets,
                    run.context,
                    action.timeout_s,
                    action.max_retries,
                    run.is_test,
                    run.image_path,
                    run.image_bytes,
                )

                def on_attempt(attempt: int) -> None:
                    """Records the attempt number."""
                    with self.ctx.db.session() as session:
                        session.execute(
                            update(ActionExecution)
                            .where(ActionExecution.id == execution_id)
                            .values(attempt=attempt)
                        )

                outcome = await act.execute(request, self.deps, on_attempt)
        secret_values = list(secrets.values())
        with self.ctx.db.session() as session:
            row = session.get(ActionExecution, execution_id)
            if row is not None:
                row.status = outcome.status
                row.attempt = outcome.attempt
                row.http_status = outcome.http_status
                row.error = redact(outcome.error, secret_values) if outcome.error else None
                row.rendered_payload = redact(outcome.rendered, secret_values)[: act.MAX_PAYLOAD]
                row.response_snippet = outcome.response[: act.MAX_RESPONSE]
                row.duration_ms = outcome.duration_ms
                row.finished_at = utcnow()
            if outcome.disable_action:
                action_row = session.get(RuleAction, action.id)
                if action_row is not None:
                    action_row.enabled = False
        if outcome.disable_action:
            self.invalidate()
            notify(
                self.ctx.db,
                self.ctx.events,
                "error",
                f"{run.rule.name}: action disabled (410 Gone)",
                outcome.error or "",
                code="action.disabled_410",
                params={"rule": run.rule.name},
            )
        elif outcome.status == "failed" and not run.is_test:
            notify(
                self.ctx.db,
                self.ctx.events,
                "error",
                f"{run.rule.name}: {action.type} failed",
                redact(outcome.error or "", secret_values),
                code="action.failed",
                params={"rule": run.rule.name, "action": action.type},
            )
        result = {
            "execution_id": execution_id,
            "action_id": action.id,
            "type": action.type,
            "status": outcome.status,
            "attempt": outcome.attempt,
            "http_status": outcome.http_status,
            "error": outcome.error,
            "duration_ms": outcome.duration_ms,
        }
        self.ctx.events.publish(
            "action.finished", {**result, "rule_id": run.rule.id, "is_test": run.is_test}
        )
        return result

    # ------------------------------------------------------------- dry run / test

    def dry_run(self, rule_id: int, limit: int) -> dict[str, Any]:
        """Evaluates a rule against the last N analyses in time order (nothing is sent)."""
        rule = self._load_rules(rule_id)
        if not rule:
            raise ApiError(404, "rule.not_found", "rule not found")
        snapshot = rule[0]
        with self.ctx.db.session() as session:
            rows = session.scalars(
                select(Analysis)
                .order_by(Analysis.created_at.desc(), Analysis.id.desc())
                .limit(limit)
            ).all()
            facts = [DbHistory._facts(r) for r in reversed(rows)]
        state = EvalState()
        last_fire: dict[str, datetime] = {}
        fires, suppressed, evaluated = [], 0, 0
        for f in facts:
            if not snapshot.in_scope(f.profile_id):
                continue
            evaluated += 1
            fired, details = evaluate(
                snapshot.id, snapshot.logic, snapshot.conditions, f, self.history, state
            )
            if not fired:
                continue
            key = snapshot.dedupe_value(f)
            previous = last_fire.get(key)
            if (
                previous is not None
                and (f.created_at - previous).total_seconds() < snapshot.cooldown_s
            ):
                suppressed += 1
                continue
            last_fire[key] = f.created_at
            fires.append(
                {
                    "analysis_id": f.id,
                    "profile_id": f.profile_id,
                    "predicted_class": f.predicted_class,
                    "verdict": f.verdict,
                    "created_at": f.created_at.isoformat() + "Z",
                    "conditions": details,
                }
            )
        return {
            "evaluated": evaluated,
            "would_fire": len(fires),
            "suppressed_by_cooldown": suppressed,
            "fires": fires,
        }

    async def test_rule(self, rule_id: int) -> list[dict[str, Any]]:
        """Runs the rule's actions once with is_test = true (last analysis or a sample)."""
        rules = self._load_rules(rule_id)
        if not rules:
            raise ApiError(404, "rule.not_found", "rule not found")
        rule = rules[0]
        with self.ctx.db.session() as session:
            last = session.scalars(
                select(Analysis.id).order_by(Analysis.id.desc()).limit(1)
            ).first()
        if last is not None:
            result = self.ctx.inspections.result(last)
            context = self.context_for(rule, result, self.history.get(last) or facts_of(result))
        else:
            context = sample_context(f"http://{self.ctx.config.host}:{self.ctx.config.port}")
            context["rule"] = {"id": rule.id, "name": rule.name, "severity": rule.severity}
        run = RuleRun(
            rule,
            self._create_executions(rule, last, None, f"test:{rule.id}", is_test=True),
            context,
            is_test=True,
        )
        return await self._run_now(run)

    async def _run_now(self, run: RuleRun) -> list[dict[str, Any]]:
        """Executes a run through the queue and waits for it (or directly without workers)."""
        if not self.tasks:
            return await self._execute_run(run)
        run.done = asyncio.get_running_loop().create_future()
        self._enqueue(run)
        return await asyncio.wait_for(run.done, timeout=300)

    async def test_target(self, target_id: int, to: list[str] | None) -> dict[str, Any]:
        """Sends a test message through a target and stores the result."""
        with self.ctx.db.session() as session:
            row = session.get(Target, target_id)
            if row is None:
                raise ApiError(404, "target.not_found", "target not found")
            kind, config = row.type, dict(row.config_json or {})
        secrets = {
            n: v
            for n in act.TARGET_SECRETS.get(kind, [])
            if (v := self.secrets.get(secret_name(target_id, n))) is not None
        }
        context = sample_context(f"http://{self.ctx.config.host}:{self.ctx.config.port}")
        context["rule"] = {"id": 0, "name": "Test message", "severity": "info"}
        action_type = {
            "smtp": "email",
            "webhook": "webhook",
            "mqtt": "mqtt_publish",
            "file": "write_file",
        }[kind]
        params = dict(act.ACTION_TYPES[action_type])
        if kind == "smtp":
            params["to"] = to or [config.get("from_address") or config.get("username")]
            params["subject"] = "Bottle inspection: test message"
            params["body"] = "This is a test message from the bottle inspection app."
        if kind == "webhook":
            params["body"] = (
                '{"type": "test", "sent_at": "' + datetime.now(timezone.utc).isoformat() + '"}'
            )
        if kind == "mqtt":
            params["topic"] = "bottle_inspection/test"
            params["payload"] = '{"type": "test"}'
        request = act.ActionRequest(
            0,
            str(uuid.uuid4()),
            action_type,
            params,
            kind,
            config,
            secrets,
            context,
            max_retries=0,
            is_test=True,
        )
        outcome = await act.execute(request, self.deps)
        detail = redact(
            outcome.error or (f"HTTP {outcome.http_status}" if outcome.http_status else "ok"),
            list(secrets.values()),
        )
        with self.ctx.db.session() as session:
            row = session.get(Target, target_id)
            if row is not None:
                row.last_test_at = utcnow()
                row.last_test_ok = outcome.status == "success"
                row.last_test_detail = detail
        return {
            "ok": outcome.status == "success",
            "detail": detail,
            "http_status": outcome.http_status,
            "duration_ms": outcome.duration_ms,
        }

    # ------------------------------------------------------------- retention

    async def _retention_loop(self) -> None:
        """Daily retention job (§13, REQ-IN11)."""
        while True:
            try:
                await asyncio.get_running_loop().run_in_executor(self.ctx.executor, self.retention)
            except Exception:  # noqa: BLE001
                log.exception("retention job failed")
            await asyncio.sleep(RETENTION_INTERVAL_S)

    def retention(self) -> dict[str, int]:
        """Deletes executions and images older than the configured retention."""
        settings = self.ctx.settings.all()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        exec_limit = now - timedelta(days=int(settings["execution_retention_days"]))
        image_limit = now - timedelta(days=int(settings["image_retention_days"]))
        deleted_images = 0
        with self.ctx.db.session() as session:
            executions = session.execute(
                delete(ActionExecution).where(ActionExecution.created_at < exec_limit)
            ).rowcount  # type: ignore[attr-defined]
            old = session.scalars(select(Image).where(Image.created_at < image_limit)).all()
            ids = [row.id for row in old]
            if ids:
                session.execute(
                    update(Inspection).where(Inspection.image_id.in_(ids)).values(image_id=None)
                )
            for row in old:
                path = Path(row.path)
                if path.is_file() and path.resolve().is_relative_to(
                    self.ctx.config.paths.images.resolve()
                ):
                    path.unlink()
                thumb = self.ctx.config.paths.thumbs / f"{row.id}.jpg"
                if thumb.is_file():
                    thumb.unlink()
                session.delete(row)
                deleted_images += 1
        return {"executions": int(executions or 0), "images": deleted_images}


def _count(values: Any) -> dict[str, int]:
    """Counter as a plain dict."""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


# ----------------------------------------------------------------- API

router = APIRouter(prefix="/api", tags=["integrations"])


def rules_service(ctx: AppContext) -> RulesService:
    """The rules service."""
    return ctx.extras["rules"]


class TargetIn(BaseModel):
    """Body of POST /api/targets (secrets are write-only)."""

    name: str = Field(min_length=1, max_length=60)
    type: Literal["smtp", "webhook", "mqtt", "file"]
    config: dict[str, Any] = Field(default_factory=dict)
    secrets: dict[str, str | None] = Field(default_factory=dict)
    enabled: bool = True


class TargetPatch(BaseModel):
    """Body of PATCH /api/targets/{id}; a secret set to null is removed."""

    name: str | None = Field(default=None, min_length=1, max_length=60)
    config: dict[str, Any] | None = None
    secrets: dict[str, str | None] | None = None
    enabled: bool | None = None


def clean_target_config(kind: str, config: dict[str, Any]) -> dict[str, Any]:
    """Known non-secret settings of a target type, with defaults."""
    defaults = act.TARGET_TYPES[kind]
    unknown = set(config) - set(defaults)
    if unknown & set(act.TARGET_SECRETS[kind]):
        raise ApiError(422, "target.secret_in_config", "send secrets in `secrets`, not in `config`")
    if unknown:
        raise ApiError(422, "target.invalid_config", "unknown settings", {"keys": sorted(unknown)})
    merged = {**defaults, **config}
    if kind == "smtp" and merged["security"] not in ("starttls", "ssl", "none"):
        raise ApiError(422, "target.invalid_config", "security must be starttls, ssl or none")
    if (
        kind == "webhook"
        and merged["base_url"]
        and not str(merged["base_url"]).lower().startswith(("http://", "https://"))
    ):
        raise ApiError(422, "target.invalid_config", "base_url must start with http:// or https://")
    if kind == "file" and merged["format"] not in ("csv", "jsonl"):
        raise ApiError(422, "target.invalid_config", "format must be csv or jsonl")
    return merged


def target_json(row: Target, store: SecretStore) -> dict[str, Any]:
    """JSON form of a target; secrets are shown masked (REQ-IN2)."""
    return {
        "id": row.id,
        "name": row.name,
        "type": row.type,
        "config": row.config_json,
        "secrets": {
            name: (MASK if store.has(secret_name(row.id, name)) else None)
            for name in act.TARGET_SECRETS[row.type]
        },
        "enabled": row.enabled,
        "last_test_at": iso(row.last_test_at),
        "last_test_ok": row.last_test_ok,
        "last_test_detail": row.last_test_detail,
    }


def save_secrets(
    store: SecretStore, target_id: int, kind: str, secrets: dict[str, str | None]
) -> None:
    """Stores or removes secret fields of a target."""
    for name, value in secrets.items():
        if name not in act.TARGET_SECRETS[kind]:
            raise ApiError(422, "target.invalid_secret", "unknown secret field", {"field": name})
        if value is None or value == "":
            store.delete(secret_name(target_id, name))
        elif value != MASK:
            store.set(secret_name(target_id, name), value)


@router.get("/integrations/meta")
def meta() -> dict[str, Any]:
    """Condition, action and target types with defaults, warning codes and template variables."""
    return {
        "conditions": CONDITION_TYPES,
        "actions": act.ACTION_TYPES,
        "action_targets": act.ACTION_TARGET,
        "targets": act.TARGET_TYPES,
        "target_secrets": act.TARGET_SECRETS,
        "warning_codes": WARNING_CODES,
        "variables": VARIABLES,
        "classes": CLASS_NAMES,
    }


@router.get("/targets")
def list_targets(ctx: AppContext = Depends(get_ctx)) -> list[dict[str, Any]]:
    """Every target."""
    store = rules_service(ctx).secrets
    with ctx.db.session() as session:
        return [target_json(r, store) for r in session.scalars(select(Target).order_by(Target.id))]


@router.post("/targets", status_code=201)
def create_target(body: TargetIn, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Creates a target; secrets go to secrets.toml."""
    store = rules_service(ctx).secrets
    config = clean_target_config(body.type, body.config)
    with ctx.db.session() as session:
        if session.scalars(select(Target).where(Target.name == body.name)).first():
            raise ApiError(409, "target.name_taken", "a target with this name exists")
        row = Target(
            name=body.name,
            type=body.type,
            config_json=config,
            enabled=body.enabled,
            secret_ref=None,
        )
        session.add(row)
        session.flush()
        row.secret_ref = f"target_{row.id}_"
        save_secrets(store, row.id, body.type, body.secrets)
        return target_json(row, store)


@router.patch("/targets/{target_id}")
def patch_target(
    target_id: int, body: TargetPatch, ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Edits a target (config merged, secrets replaced or removed)."""
    store = rules_service(ctx).secrets
    with ctx.db.session() as session:
        row = session.get(Target, target_id)
        if row is None:
            raise ApiError(404, "target.not_found", "target not found")
        if body.name is not None:
            row.name = body.name
        if body.config is not None:
            row.config_json = clean_target_config(
                row.type, {**(row.config_json or {}), **body.config}
            )
        if body.enabled is not None:
            row.enabled = body.enabled
        if body.secrets:
            save_secrets(store, row.id, row.type, body.secrets)
        return target_json(row, store)


@router.delete("/targets/{target_id}")
def delete_target(target_id: int, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Deletes a target and its secrets (actions using it lose their target)."""
    service = rules_service(ctx)
    with ctx.db.session() as session:
        row = session.get(Target, target_id)
        if row is None:
            raise ApiError(404, "target.not_found", "target not found")
        session.delete(row)
    service.secrets.delete_prefix(f"target_{target_id}_")
    service.invalidate()
    return {"id": target_id, "deleted": True}


class TestTargetIn(BaseModel):
    """Optional recipients of an SMTP test."""

    to: list[str] | None = None


@router.post("/targets/{target_id}/test")
async def test_target(
    target_id: int, body: TestTargetIn | None = None, ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Sends a test message and stores the result."""
    return await rules_service(ctx).test_target(target_id, body.to if body else None)


class ConditionIn(BaseModel):
    """One condition of a rule."""

    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    negate: bool = False


class ActionIn(BaseModel):
    """One action of a rule."""

    type: str
    target_id: int | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    timeout_s: float | None = Field(default=None, gt=0, le=300)
    max_retries: int = Field(default=3, ge=0, le=3)


class RuleIn(BaseModel):
    """Body of POST /api/rules."""

    name: str = Field(min_length=1, max_length=80)
    description: str = ""
    enabled: bool = True
    severity: Literal["info", "warning", "critical"] = "warning"
    scope: dict[str, Any] = Field(default_factory=lambda: {"profiles": "all"})
    trigger: Literal["analysis_completed"] = "analysis_completed"
    condition_logic: Literal["all", "any"] = "all"
    cooldown_s: int = Field(default=60, ge=0, le=86400)
    dedupe_key: Literal["rule", "rule+class", "rule+profile"] = "rule+class"
    run_mode: Literal["single", "queued", "parallel"] = "single"
    conditions: list[ConditionIn] = Field(default_factory=list)
    actions: list[ActionIn] = Field(default_factory=list)


class RulePatch(BaseModel):
    """Body of PATCH /api/rules/{id}; conditions/actions, when given, replace the lists."""

    name: str | None = Field(default=None, min_length=1, max_length=80)
    description: str | None = None
    enabled: bool | None = None
    severity: Literal["info", "warning", "critical"] | None = None
    scope: dict[str, Any] | None = None
    condition_logic: Literal["all", "any"] | None = None
    cooldown_s: int | None = Field(default=None, ge=0, le=86400)
    dedupe_key: Literal["rule", "rule+class", "rule+profile"] | None = None
    run_mode: Literal["single", "queued", "parallel"] | None = None
    conditions: list[ConditionIn] | None = None
    actions: list[ActionIn] | None = None


def check_scope(scope: dict[str, Any]) -> dict[str, Any]:
    """{"profiles": "all"} or {"profiles": [ids]}."""
    profiles = scope.get("profiles", "all")
    if profiles != "all" and not (
        isinstance(profiles, list) and all(isinstance(p, int) for p in profiles)
    ):
        raise ApiError(422, "rule.invalid_scope", "scope.profiles must be 'all' or a list of ids")
    return {"profiles": profiles}


def check_actions(session: Any, items: list[ActionIn]) -> list[ActionIn]:
    """Action types, parameters and target types."""
    for item in items:
        if item.type not in act.ACTION_TYPES:
            raise ApiError(422, "rule.unknown_action", "unknown action", {"type": item.type})
        unknown = set(item.params) - set(act.ACTION_TYPES[item.type])
        if unknown:
            raise ApiError(
                422, "rule.invalid_action", "unknown parameters", {"keys": sorted(unknown)}
            )
        needed = act.ACTION_TARGET.get(item.type)
        if needed:
            target = session.get(Target, item.target_id) if item.target_id else None
            if target is None or target.type != needed:
                raise ApiError(
                    422,
                    "rule.target_required",
                    f"this action needs a {needed} target",
                    {"type": item.type},
                )
        item.params = {**act.ACTION_TYPES[item.type], **item.params}
    return items


def write_children(
    session: Any, rule_id: int, conditions: list[ConditionIn] | None, actions: list[ActionIn] | None
) -> None:
    """Replaces the conditions and/or actions of a rule."""
    if conditions is not None:
        validated = []
        for c in conditions:
            try:
                validated.append((c, validate_condition(c.type, c.params)))
            except ConditionError as error:
                raise ApiError(422, error.code, str(error), {"type": c.type}) from error
        session.execute(delete(RuleCondition).where(RuleCondition.rule_id == rule_id))
        for position, (c, params) in enumerate(validated):
            session.add(
                RuleCondition(
                    rule_id=rule_id,
                    position=position,
                    type=c.type,
                    params_json=params,
                    negate=c.negate,
                )
            )
    if actions is not None:
        checked = check_actions(session, actions)
        session.execute(
            update(ActionExecution).where(ActionExecution.rule_id == rule_id).values(action_id=None)
        )
        session.execute(delete(RuleAction).where(RuleAction.rule_id == rule_id))
        for position, a in enumerate(checked):
            session.add(
                RuleAction(
                    rule_id=rule_id,
                    position=position,
                    type=a.type,
                    target_id=a.target_id,
                    params_json=a.params,
                    enabled=a.enabled,
                    timeout_s=a.timeout_s,
                    max_retries=a.max_retries,
                )
            )


def rule_json(session: Any, row: Rule) -> dict[str, Any]:
    """JSON form of a rule with its conditions, actions and firing statistics."""
    conditions = session.scalars(
        select(RuleCondition)
        .where(RuleCondition.rule_id == row.id)
        .order_by(RuleCondition.position)
    ).all()
    actions = session.scalars(
        select(RuleAction).where(RuleAction.rule_id == row.id).order_by(RuleAction.position)
    ).all()
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=24)
    fired_24h = session.scalar(
        select(func.count()).select_from(
            select(ActionExecution.analysis_id)
            .where(
                ActionExecution.rule_id == row.id,
                ActionExecution.created_at >= since,
                ActionExecution.is_test.is_(False),
                ActionExecution.status != "skipped_cooldown",
            )
            .distinct()
            .subquery()
        )
    )
    last = session.scalar(
        select(func.max(ActionExecution.created_at)).where(
            ActionExecution.rule_id == row.id,
            ActionExecution.is_test.is_(False),
            ActionExecution.status != "skipped_cooldown",
        )
    )
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "enabled": row.enabled,
        "severity": row.severity,
        "scope": row.scope_json,
        "trigger": row.trigger,
        "condition_logic": row.condition_logic,
        "cooldown_s": row.cooldown_s,
        "dedupe_key": row.dedupe_key,
        "run_mode": row.run_mode,
        "conditions": [
            {"id": c.id, "type": c.type, "params": c.params_json, "negate": c.negate}
            for c in conditions
        ],
        "actions": [
            {
                "id": a.id,
                "type": a.type,
                "target_id": a.target_id,
                "params": a.params_json,
                "enabled": a.enabled,
                "timeout_s": a.timeout_s,
                "max_retries": a.max_retries,
            }
            for a in actions
        ],
        "fired_24h": int(fired_24h or 0),
        "last_fired_at": iso(last),
        "created_at": iso(row.created_at),
        "updated_at": iso(row.updated_at),
    }


@router.get("/rules")
def list_rules(ctx: AppContext = Depends(get_ctx)) -> list[dict[str, Any]]:
    """Every rule."""
    with ctx.db.session() as session:
        return [rule_json(session, r) for r in session.scalars(select(Rule).order_by(Rule.id))]


@router.post("/rules", status_code=201)
def create_rule(body: RuleIn, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Creates a rule with its conditions and actions."""
    with ctx.db.session() as session:
        row = Rule(
            name=body.name,
            description=body.description,
            enabled=body.enabled,
            severity=body.severity,
            scope_json=check_scope(body.scope),
            trigger=body.trigger,
            condition_logic=body.condition_logic,
            cooldown_s=body.cooldown_s,
            dedupe_key=body.dedupe_key,
            run_mode=body.run_mode,
            created_at=utcnow(),
            updated_at=utcnow(),
        )
        session.add(row)
        session.flush()
        write_children(session, row.id, body.conditions, body.actions)
        session.flush()
        result = rule_json(session, row)
    rules_service(ctx).invalidate()
    return result


@router.get("/rules/{rule_id}")
def read_rule(rule_id: int, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """One rule."""
    with ctx.db.session() as session:
        row = session.get(Rule, rule_id)
        if row is None:
            raise ApiError(404, "rule.not_found", "rule not found")
        return rule_json(session, row)


@router.patch("/rules/{rule_id}")
def patch_rule(rule_id: int, body: RulePatch, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Edits a rule."""
    with ctx.db.session() as session:
        row = session.get(Rule, rule_id)
        if row is None:
            raise ApiError(404, "rule.not_found", "rule not found")
        values = body.model_dump(exclude_unset=True, exclude={"conditions", "actions", "scope"})
        for key, value in values.items():
            setattr(row, key, value)
        if body.scope is not None:
            row.scope_json = check_scope(body.scope)
        write_children(session, row.id, body.conditions, body.actions)
        row.updated_at = utcnow()
        session.flush()
        result = rule_json(session, row)
    rules_service(ctx).invalidate()
    return result


@router.delete("/rules/{rule_id}")
def delete_rule(rule_id: int, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Deletes a rule (its log rows stay, without the rule link)."""
    with ctx.db.session() as session:
        row = session.get(Rule, rule_id)
        if row is None:
            raise ApiError(404, "rule.not_found", "rule not found")
        session.execute(
            update(ActionExecution)
            .where(ActionExecution.rule_id == rule_id)
            .values(rule_id=None, action_id=None)
        )
        session.delete(row)
    rules_service(ctx).invalidate()
    return {"id": rule_id, "deleted": True}


class DryRunIn(BaseModel):
    """Body of POST /api/rules/{id}/dry-run."""

    limit: int = Field(default=100, ge=1, le=5000)


@router.post("/rules/{rule_id}/dry-run")
async def dry_run(
    rule_id: int, body: DryRunIn | None = None, ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    """Evaluates the rule against the last N analyses (nothing is sent)."""
    service = rules_service(ctx)
    limit = body.limit if body else 100
    return await asyncio.get_running_loop().run_in_executor(
        ctx.executor, service.dry_run, rule_id, limit
    )


@router.post("/rules/{rule_id}/test")
async def test_rule(rule_id: int, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Runs the rule's actions once (is_test = true)."""
    return {"results": await rules_service(ctx).test_rule(rule_id)}


class PreviewIn(BaseModel):
    """Body of POST /api/templates/preview."""

    template: str = Field(max_length=65536)
    analysis_id: int | None = None
    rule_id: int | None = None


@router.post("/templates/preview")
def template_preview(body: PreviewIn, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Renders a template against the last analysis (or a sample payload)."""
    service = rules_service(ctx)
    with ctx.db.session() as session:
        analysis_id = (
            body.analysis_id
            or session.scalars(select(Analysis.id).order_by(Analysis.id.desc()).limit(1)).first()
        )
    rules = service._load_rules(body.rule_id) if body.rule_id else []
    rule = (
        rules[0]
        if rules
        else RuleSnapshot(
            0,
            "Preview",
            "warning",
            True,
            {"profiles": "all"},
            "analysis_completed",
            "all",
            60,
            "rule+class",
            "single",
            [],
            [],
        )
    )
    if analysis_id is not None:
        result = ctx.inspections.result(analysis_id)
        context = service.context_for(rule, result, facts_of(result))
        source = "analysis"
    else:
        context = sample_context(f"http://{ctx.config.host}:{ctx.config.port}")
        source = "sample"
    try:
        return {
            "ok": True,
            "output": render(body.template, context),
            "source": source,
            "analysis_id": analysis_id,
        }
    except TemplateRenderError as error:
        return {"ok": False, "error": str(error), "source": source, "analysis_id": analysis_id}


@router.get("/executions")
def executions(
    rule: int | None = None,
    target: int | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """The action log, newest first, filtered by rule, target and status."""
    where = []
    if rule is not None:
        where.append(ActionExecution.rule_id == rule)
    if target is not None:
        where.append(ActionExecution.target_id == target)
    if status:
        where.append(ActionExecution.status.in_(status.split(",")))
    with ctx.db.session() as session:
        total = session.scalar(select(func.count(ActionExecution.id)).where(*where)) or 0
        rows = session.scalars(
            select(ActionExecution)
            .where(*where)
            .order_by(ActionExecution.id.desc())
            .limit(min(limit, 500))
            .offset(offset)
        ).all()
        names = {r.id: r.name for r in session.scalars(select(Rule))}
        targets = {t.id: t.name for t in session.scalars(select(Target))}
        items = [
            {
                "id": r.id,
                "rule_id": r.rule_id,
                "rule": names.get(r.rule_id or -1),
                "action_id": r.action_id,
                "action_type": r.action_type,
                "target_id": r.target_id,
                "target": targets.get(r.target_id or -1),
                "analysis_id": r.analysis_id,
                "event_uuid": r.event_uuid,
                "status": r.status,
                "attempt": r.attempt,
                "http_status": r.http_status,
                "error": r.error,
                "rendered_payload": r.rendered_payload,
                "response_snippet": r.response_snippet,
                "duration_ms": r.duration_ms,
                "is_test": r.is_test,
                "suppressed_count": r.suppressed_count,
                "created_at": iso(r.created_at),
                "finished_at": iso(r.finished_at),
            }
            for r in rows
        ]
    return {"total": int(total), "items": items}


def register(app: FastAPI, ctx: AppContext) -> None:
    """Adds the rules service and the integrations API."""
    ctx.extras["rules"] = RulesService(ctx)
    app.include_router(router)
