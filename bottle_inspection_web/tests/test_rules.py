"""Integrations: conditions, cooldown/dedupe, templates sandbox, webhook signing, secrets,
and an end-to-end rule sending a real e-mail (local SMTP) and a signed webhook (local HTTP)."""

from __future__ import annotations

import json
import os
import stat
import threading
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert

from ..app import create_app
from ..db.models import Analysis, Inspection
from ..inspections import AnalysisContext
from ..rules import actions as act
from ..rules.conditions import AnalysisFacts, ConditionSpec, EvalState, evaluate, validate_condition
from ..rules.secrets import MASK, SecretStore, redact
from ..rules.templates import TemplateRenderError, render
from .conftest import make_config

# ----------------------------------------------------------------- unit: conditions


class ListHistory:
    """In-memory history (newest last in `items`)."""

    def __init__(self, items: list[AnalysisFacts]) -> None:
        """All analyses so far."""
        self.items = items

    def last(self, facts: AnalysisFacts, count: int) -> list[AnalysisFacts]:
        """Last `count` up to `facts`, newest first."""
        upto = [a for a in self.items if a.id <= facts.id and a.profile_id == facts.profile_id]
        return list(reversed(upto))[:count]

    def since(self, facts: AnalysisFacts, start: datetime) -> list[AnalysisFacts]:
        """From `start` up to `facts`."""
        return [
            a
            for a in self.items
            if a.id <= facts.id and a.created_at >= start and a.profile_id == facts.profile_id
        ]


T0 = datetime(2026, 11, 9, 12, 0, 0)


def make_facts(
    classes: list[str], profile: int = 1, seconds: float = 1.0, **extra: Any
) -> list[AnalysisFacts]:
    """A sequence of analyses, one per class, `seconds` apart."""
    return [
        AnalysisFacts(
            i + 1,
            profile,
            c,
            "pass" if c == "bottle_no_defect" else "reject",
            extra.get("confidence", 0.9),
            extra.get("total_ms", 30.0),
            tuple(extra.get("warnings", ())),
            T0 + timedelta(seconds=i * seconds),
        )
        for i, c in enumerate(classes)
    ]


def fires(spec: dict[str, Any], items: list[AnalysisFacts], negate: bool = False) -> list[bool]:
    """Whether a single-condition rule fires on each analysis."""
    condition = ConditionSpec(
        spec["type"], validate_condition(spec["type"], spec.get("params", {})), negate
    )
    history, state = ListHistory(items), EvalState()
    return [evaluate(1, "all", [condition], f, history, state)[0] for f in items]


OK, NC, LF = "bottle_no_defect", "defect_no_cap", "defect_low_fill"


def test_simple_conditions() -> None:
    """class_is, verdict_is, any_defect, confidence, time, warnings, negate."""
    items = make_facts([OK, NC, LF])
    assert fires({"type": "class_is", "params": {"classes": [NC]}}, items) == [False, True, False]
    assert fires({"type": "verdict_is", "params": {"verdict": "pass"}}, items) == [
        True,
        False,
        False,
    ]
    assert fires({"type": "any_defect"}, items) == [False, True, True]
    assert fires({"type": "any_defect"}, items, negate=True) == [True, False, False]
    low = make_facts([OK], confidence=0.2)
    assert fires({"type": "confidence_below", "params": {"value": 0.5}}, low) == [True]
    slow = make_facts([OK], total_ms=120.0)
    assert fires({"type": "processing_time_above", "params": {"ms": 80}}, slow) == [True]
    warned = make_facts([OK], warnings=("low_foreground",))
    assert fires({"type": "warning_present", "params": {"codes": ["low_foreground"]}}, warned) == [
        True
    ]


def test_repeated_class_matches_the_cli_alert() -> None:
    """n=4 in the last m=10 (the CLI's simulated alert)."""
    sequence = [NC, OK, NC, OK, NC, OK, OK, NC, OK, OK, OK, OK, OK, OK, OK, NC]
    result = fires(
        {"type": "repeated_class", "params": {"class": NC, "n": 4, "m": 10}}, make_facts(sequence)
    )
    assert result.index(True) == 7  # the 4th no_cap within 10 bottles
    assert result[-1] is False  # only 2 no_cap in the last 10
    any_reject = fires(
        {"type": "repeated_class", "params": {"class": "any", "n": 2, "m": 3}},
        make_facts([NC, LF, NC]),
    )
    assert any_reject == [False, False, True]


def test_window_is_per_profile() -> None:
    """Analyses of other profiles do not count (each profile evaluated on its own)."""
    items = make_facts([NC, NC], profile=1) + [
        AnalysisFacts(10 + i, 2, NC, "reject", 0.9, 30.0, (), T0) for i in range(5)
    ]
    items.sort(key=lambda a: a.id)
    spec = {"type": "consecutive_rejects", "params": {"n": 3}}
    assert fires(spec, items)[:2] == [False, False]


def test_reject_rate_hysteresis() -> None:
    """Fires once above the threshold, re-arms only after falling back below it."""
    sequence = [OK] * 5 + [NC] * 5 + [NC] * 3 + [OK] * 12 + [NC] * 10
    result = fires(
        {
            "type": "reject_rate_above",
            "params": {"percent": 40, "window_minutes": 10, "min_samples": 5},
        },
        make_facts(sequence),
    )
    assert result.count(True) == 2
    first = result.index(True)
    assert result[first + 1 : first + 3] == [False, False]


# ----------------------------------------------------------------- unit: templates, signing, secrets


def test_template_sandbox_and_limits() -> None:
    """Unsafe attribute access is blocked; output over 64 KB fails; undefined fails."""
    assert render("{{ rule.name }}: {{ n * 2 }}", {"rule": {"name": "R"}, "n": 3}) == "R: 6"
    with pytest.raises(TemplateRenderError):
        render("{{ ''.__class__.__mro__[1].__subclasses__() }}", {})
    with pytest.raises(TemplateRenderError):
        render("{{ 'x' * 70000 }}", {})
    with pytest.raises(TemplateRenderError):
        render("{{ missing.value }}", {})


def test_standard_webhooks_reference_vector() -> None:
    """Signature from the Standard Webhooks specification test vector."""
    signature = act.sign_webhook(
        "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw",
        "msg_p5jXN8AQM9LWM0D4loKWxJek",
        1614265330,
        b'{"test": 2432232314}',
    )
    assert signature == "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE="


def test_secret_store_and_redaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """0600 file, environment override, redaction of values and auth headers."""
    store = SecretStore(tmp_path / "secrets.toml")
    store.set("target_1_password", "hunter22")
    assert stat.S_IMODE(os.stat(tmp_path / "secrets.toml").st_mode) == 0o600
    assert store.get("target_1_password") == "hunter22"
    monkeypatch.setenv("BOTTLE_WEB_SECRET_TARGET_1_PASSWORD", "from-env")
    assert store.get("target_1_password") == "from-env"
    text = redact('password=hunter22 {"Authorization": "Bearer abc"} hunter22', ["hunter22"])
    assert "hunter22" not in text and "abc" not in text and MASK in text


# ----------------------------------------------------------------- end to end


class Recorder(BaseHTTPRequestHandler):
    """Local webhook receiver: records requests; status from the path (/s/<code>)."""

    received: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802
        """Records the request and answers."""
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length)
        Recorder.received.append({"path": self.path, "headers": dict(self.headers), "body": body})
        code = int(self.path.split("/s/")[1]) if "/s/" in self.path else 200
        self.send_response(code)
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Silent."""


@pytest.fixture()
def http_server() -> Iterator[str]:
    """A local HTTP server; yields its base URL."""
    Recorder.received = []
    server = HTTPServer(("127.0.0.1", 0), Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


@pytest.fixture()
def smtp_server() -> Iterator[tuple[int, list[Any]]]:
    """A local SMTP debug server (aiosmtpd); yields (port, received messages)."""
    from aiosmtpd.controller import Controller

    messages: list[Any] = []

    class Handler:
        """Keeps every message."""

        async def handle_DATA(
            self, _server: Any, _session: Any, envelope: Any
        ) -> str:  # noqa: N802
            """Stores the message."""
            messages.append(envelope)
            return "250 OK"

    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    controller = Controller(Handler(), hostname="127.0.0.1", port=port)
    controller.start()
    yield port, messages
    controller.stop()


@pytest.fixture()
def rules_client(
    tmp_path: Path, code_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """App client with fast retries."""
    monkeypatch.setattr(act, "RETRY_DELAYS", [0.05, 0.05, 0.05])
    app = create_app(make_config(tmp_path, code_dir))
    with TestClient(app) as client:
        yield client


def add_analysis(
    client: TestClient, profile_id: int, predicted: str, index: int
) -> AnalysisContext:
    """Inserts one analysis and returns the hook context the engine receives."""
    ctx = client.app.state.ctx  # type: ignore[attr-defined]
    with ctx.db.engine.begin() as connection:
        inspection_id = connection.execute(
            insert(Inspection).values(
                source="upload", created_at=datetime.now(timezone.utc).replace(tzinfo=None)
            )
        ).inserted_primary_key[0]
        analysis_id = connection.execute(
            insert(Analysis).values(
                inspection_id=inspection_id,
                profile_id=profile_id,
                profile_version=1,
                modified_params=False,
                predicted_class=predicted,
                verdict="pass" if predicted == OK else "reject",
                confidence=0.9,
                scores_json={},
                score_kind="rf_probability",
                timings_json={"total": 30.0},
                total_ms=30.0,
                warnings_json=[],
                source="upload",
                created_at=datetime.now(timezone.utc).replace(tzinfo=None)
                + timedelta(milliseconds=index),
            )
        ).inserted_primary_key[0]
    result = ctx.inspections.result(analysis_id)
    return AnalysisContext(analysis_id, inspection_id, profile_id, result)


def wait_log(client: TestClient, predicate: Any, timeout: float = 10) -> dict[str, Any]:
    """Polls the execution log until `predicate(items)` holds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        log = client.get("/api/executions").json()
        if predicate(log["items"]):
            return log
        time.sleep(0.05)
    raise AssertionError(client.get("/api/executions").json())


def test_rule_end_to_end_email_webhook_cooldown(
    rules_client: TestClient, http_server: str, smtp_server: tuple[int, list[Any]]
) -> None:
    """Acceptance 7: "defect_no_cap 4 times in the last 10" sends an e-mail and a signed
    webhook; both appear in the log; cooldown suppresses repeats; secrets never leak."""
    port, messages = smtp_server
    client = rules_client
    smtp = client.post(
        "/api/targets",
        json={
            "name": "Local SMTP",
            "type": "smtp",
            "config": {
                "host": "127.0.0.1",
                "port": port,
                "security": "none",
                "from_address": "qa@line.local",
            },
        },
    ).json()
    hook = client.post(
        "/api/targets",
        json={
            "name": "MES",
            "type": "webhook",
            "config": {"base_url": http_server},
            "secrets": {"signing_secret": "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"},
        },
    ).json()
    assert hook["secrets"] == {"signing_secret": MASK}
    assert "whsec" not in json.dumps(client.get("/api/targets").json())
    assert client.post(f"/api/targets/{smtp['id']}/test", json={"to": ["qa@line.local"]}).json()[
        "ok"
    ]
    assert len(messages) == 1

    profile_id = client.get("/api/profiles").json()[0]["id"]
    rule = client.post(
        "/api/rules",
        json={
            "name": "No cap repeated",
            "severity": "warning",
            "cooldown_s": 3600,
            "conditions": [
                {"type": "repeated_class", "params": {"class": "defect_no_cap", "n": 4, "m": 10}}
            ],
            "actions": [
                {"type": "email", "target_id": smtp["id"], "params": {"to": ["qa@line.local"]}},
                {"type": "webhook", "target_id": hook["id"], "params": {"path": "/events"}},
            ],
        },
    )
    assert rule.status_code == 201, rule.text
    rule_id = rule.json()["id"]
    service = client.app.state.ctx.extras["rules"]  # type: ignore[attr-defined]
    sequence = [NC, OK, NC, OK, NC, NC, NC, NC]
    for index, name in enumerate(sequence):
        service.on_analysis(add_analysis(client, profile_id, name, index))

    log = wait_log(client, lambda items: sum(i["status"] == "success" for i in items) >= 2)
    statuses = {i["action_type"]: i for i in log["items"] if i["status"] == "success"}
    first_fire = sorted(sequence[:6]).count(NC)  # the 4th no_cap is the 6th analysis
    assert first_fire == 4 and {i["analysis_id"] for i in statuses.values()} == {6}
    assert set(statuses) == {"email", "webhook"}
    assert len(messages) == 2 and b"defect_no_cap" in messages[1].content
    request = Recorder.received[-1]
    assert request["path"] == "/events"
    assert request["headers"]["webhook-id"] == statuses["webhook"]["event_uuid"]
    expected = act.sign_webhook(
        "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw",
        request["headers"]["webhook-id"],
        int(request["headers"]["webhook-timestamp"]),
        request["body"],
    )
    assert request["headers"]["webhook-signature"] == expected
    assert json.loads(request["body"])["predicted_class"] == "defect_no_cap"
    suppressed = [i for i in log["items"] if i["status"] == "skipped_cooldown"]
    assert len(suppressed) == 1 and suppressed[0]["suppressed_count"] >= 2
    assert "whsec" not in json.dumps(log)
    rules = client.get("/api/rules").json()
    assert rules[0]["fired_24h"] == 1 and rules[0]["last_fired_at"]

    dry = client.post(f"/api/rules/{rule_id}/dry-run", json={"limit": 100}).json()
    assert dry["would_fire"] == 1 and dry["suppressed_by_cooldown"] >= 2


def test_webhook_retries_and_410(rules_client: TestClient, http_server: str) -> None:
    """5xx is retried (3 more attempts); 410 disables the action and notifies."""
    client = rules_client
    hook = client.post(
        "/api/targets", json={"name": "W", "type": "webhook", "config": {"base_url": http_server}}
    ).json()
    rule = client.post(
        "/api/rules",
        json={
            "name": "Any defect",
            "cooldown_s": 0,
            "dedupe_key": "rule",
            "run_mode": "queued",
            "conditions": [{"type": "any_defect"}],
            "actions": [
                {"type": "webhook", "target_id": hook["id"], "params": {"path": "/s/503"}},
                {"type": "webhook", "target_id": hook["id"], "params": {"path": "/s/410"}},
            ],
        },
    ).json()
    result = client.post(f"/api/rules/{rule['id']}/test").json()["results"]
    assert (
        result[0]["status"] == "failed"
        and result[0]["attempt"] == 4
        and result[0]["http_status"] == 503
    )
    assert (
        result[1]["status"] == "failed"
        and result[1]["attempt"] == 1
        and result[1]["http_status"] == 410
    )
    assert client.get(f"/api/rules/{rule['id']}").json()["actions"][1]["enabled"] is False
    codes = [n["code"] for n in client.get("/api/notifications").json()["items"]]
    assert "action.disabled_410" in codes


def test_file_and_notify_actions_and_template_preview(rules_client: TestClient) -> None:
    """write_file appends JSON lines; notify_ui creates a notification; preview renders."""
    client = rules_client
    target = client.post(
        "/api/targets",
        json={"name": "Log", "type": "file", "config": {"directory": "exports/rules"}},
    ).json()
    rule = client.post(
        "/api/rules",
        json={
            "name": "Everything",
            "cooldown_s": 0,
            "conditions": [
                {"type": "verdict_is", "params": {"verdict": "pass"}},
                {"type": "any_defect"},
            ],
            "condition_logic": "any",
            "actions": [
                {"type": "write_file", "target_id": target["id"]},
                {
                    "type": "notify_ui",
                    "params": {"title": "Hello {{ rule.name }}", "level": "info"},
                },
            ],
        },
    ).json()
    result = client.post(f"/api/rules/{rule['id']}/test").json()["results"]
    assert [r["status"] for r in result] == ["success", "success"]
    data_root = client.app.state.ctx.config.paths.root  # type: ignore[attr-defined]
    files = list((data_root / "exports" / "rules").glob("*.jsonl"))
    assert files and json.loads(files[0].read_text().splitlines()[0])["verdict"]
    assert any(
        n["title"] == "Hello Everything" for n in client.get("/api/notifications").json()["items"]
    )
    preview = client.post(
        "/api/templates/preview", json={"template": "{{ analysis.predicted_class }}"}
    ).json()
    assert preview["ok"] and preview["source"] == "sample" and preview["output"] == "defect_no_cap"
    bad = client.post("/api/templates/preview", json={"template": "{{ nope.x }}"}).json()
    assert bad["ok"] is False


def test_invalid_rules_are_rejected(rules_client: TestClient) -> None:
    """Unknown condition, bad params and missing targets give 422."""
    client = rules_client
    assert (
        client.post("/api/rules", json={"name": "x", "conditions": [{"type": "nope"}]}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/rules",
            json={
                "name": "x",
                "conditions": [{"type": "class_is", "params": {"classes": ["cat"]}}],
            },
        ).status_code
        == 422
    )
    assert (
        client.post("/api/rules", json={"name": "x", "actions": [{"type": "email"}]}).status_code
        == 422
    )
    meta = client.get("/api/integrations/meta").json()
    assert meta["classes"][0] == "bottle_no_defect" and "repeated_class" in meta["conditions"]
