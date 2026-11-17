"""Rule actions and targets (REQ-IN1, IN6, IN8, IN9): notify_ui, play_sound, email, webhook,
mqtt_publish and write_file, with retries on transient errors and Standard Webhooks signing."""

from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import hmac
import io
import ipaddress
import json
import random
import socket
import ssl
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiosmtplib
import httpx

from .templates import TemplateRenderError, render, render_value

ACTION_TYPES: dict[str, dict[str, Any]] = {
    "notify_ui": {
        "title": "{{ rule.name }}",
        "message": "{{ profile.name }}: {{ analysis.predicted_class }} ({{ verdict }})",
        "level": "warning",
    },
    "play_sound": {"sound": "chime"},
    "email": {
        "to": [],
        "subject": "[{{ rule.severity }}] {{ rule.name }}",
        "body": "{{ profile.name }} classified {{ analysis.predicted_class }} ({{ verdict }}).\n"
        "Window counts: {{ window_stats | tojson }}\n{{ app_url }}",
        "html": "",
        "attach_image": False,
    },
    "webhook": {"method": "POST", "path": "", "headers": {}, "body": ""},
    "mqtt_publish": {
        "topic": "bottle_inspection/{{ profile.name }}/{{ verdict }}",
        "payload": "",
        "qos": 1,
        "retain": False,
    },
    "write_file": {"fields": ["created_at", "profile", "predicted_class", "verdict", "confidence"]},
}
"""v1 action types and default parameters (templated fields are rendered with Jinja2)."""

TARGET_TYPES: dict[str, dict[str, Any]] = {
    "smtp": {
        "host": "",
        "port": 587,
        "security": "starttls",
        "username": "",
        "from_address": "",
        "timeout_s": 15,
    },
    "webhook": {
        "base_url": "",
        "method": "POST",
        "headers": {},
        "timeout_s": 10,
        "allow_private_ips": True,
    },
    "mqtt": {
        "host": "",
        "port": 1883,
        "tls": False,
        "ca_file": "",
        "username": "",
        "client_id": "",
        "qos": 1,
        "retain": False,
    },
    "file": {"directory": "exports", "format": "jsonl", "rotation": "daily", "size_mb": 10},
}
"""v1 target types and their non-secret settings."""

TARGET_SECRETS: dict[str, list[str]] = {
    "smtp": ["password"],
    "webhook": ["signing_secret"],
    "mqtt": ["password"],
    "file": [],
}
"""Secret fields of each target type (stored in secrets.toml, never in the DB)."""

ACTION_TARGET: dict[str, str] = {
    "email": "smtp",
    "webhook": "webhook",
    "mqtt_publish": "mqtt",
    "write_file": "file",
}

RETRY_DELAYS: list[float] = [1.0, 5.0, 30.0]
"""Delays before the 2nd, 3rd and 4th attempts (with jitter); tests may shorten them."""

MAX_PAYLOAD: int = 64 * 1024
MAX_RESPONSE: int = 2 * 1024


class TransientError(Exception):
    """Network error, HTTP 429 or 5xx: retried."""

    def __init__(
        self, message: str, http_status: int | None = None, retry_after: float | None = None
    ) -> None:
        """Error with optional HTTP status and Retry-After."""
        super().__init__(message)
        self.http_status = http_status
        self.retry_after = retry_after


class PermanentError(Exception):
    """Configuration error or other 4xx: not retried."""

    def __init__(self, message: str, http_status: int | None = None, disable: bool = False) -> None:
        """Error with optional HTTP status; `disable` for HTTP 410 Gone."""
        super().__init__(message)
        self.http_status = http_status
        self.disable = disable


@dataclass
class ActionRequest:
    """Everything one action execution needs."""

    execution_id: int
    event_uuid: str
    action_type: str
    params: dict[str, Any]
    target_type: str | None
    target_config: dict[str, Any]
    target_secrets: dict[str, str]
    context: dict[str, Any]
    timeout_s: float | None = None
    max_retries: int = 3
    is_test: bool = False
    image_path: Path | None = None
    image_bytes: bytes | None = None


@dataclass
class ActionResult:
    """What one attempt produced."""

    rendered: str = ""
    http_status: int | None = None
    response: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Outcome:
    """Final result of an execution (after retries)."""

    status: str
    attempt: int
    rendered: str
    http_status: int | None
    error: str | None
    response: str
    duration_ms: float
    disable_action: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionDeps:
    """Services the executors need (UI notifications, events, data paths)."""

    notify: Callable[[str, str, str], Any]
    publish_event: Callable[[str, dict[str, Any]], None]
    data_root: Path


# ----------------------------------------------------------------- Standard Webhooks


def sign_webhook(secret: str, message_id: str, timestamp: int, body: bytes) -> str:
    """`v1,<base64 HMAC-SHA256 of "{id}.{timestamp}.{body}">` (Standard Webhooks).

    The secret is `whsec_` + base64 key; a secret without the prefix is used as raw text.
    """
    key = (
        base64.b64decode(secret.removeprefix("whsec_"))
        if secret.startswith("whsec_")
        else secret.encode()
    )
    signed = f"{message_id}.{timestamp}.".encode() + body
    digest = hmac.new(key, signed, hashlib.sha256).digest()
    return "v1," + base64.b64encode(digest).decode()


# ----------------------------------------------------------------- executors


def _check_host(url: str, allow_private: bool) -> None:
    """Blocks private/loopback addresses when the target does not allow them."""
    if allow_private:
        return
    host = urlsplit(url).hostname or ""
    try:
        addresses = {str(info[4][0]) for info in socket.getaddrinfo(host, None)}
    except OSError as error:
        raise TransientError(f"cannot resolve {host}: {error}") from error
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise PermanentError(f"private address {address} is not allowed for this target")


def _retry_after(value: str | None) -> float | None:
    """Seconds from a Retry-After header (seconds or HTTP date)."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        from email.utils import parsedate_to_datetime

        try:
            when = parsedate_to_datetime(value)
            return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return None


async def run_webhook(req: ActionRequest, _deps: ActionDeps) -> ActionResult:
    """HTTP request to the target URL; signed when the target has a signing secret."""
    config = req.target_config
    params = req.params
    base = str(config.get("base_url", "")).rstrip("/")
    if not base.lower().startswith(("http://", "https://")):
        raise PermanentError("the webhook target needs an http(s) base URL")
    path = render(str(params.get("path") or ""), req.context)
    url = base + (("/" + path.lstrip("/")) if path else "")
    method = str(params.get("method") or config.get("method") or "POST").upper()
    headers = {
        **dict(config.get("headers") or {}),
        **render_value(dict(params.get("headers") or {}), req.context),
    }
    template = str(params.get("body") or "")
    if template.strip():
        rendered = render(template, req.context)
    else:
        rendered = json.dumps(req.context.get("analysis", {}), ensure_ascii=False, default=str)
    body = rendered.encode("utf-8")
    headers.setdefault("content-type", "application/json")
    secret = req.target_secrets.get("signing_secret")
    if secret:
        timestamp = int(time.time())
        headers["webhook-id"] = req.event_uuid
        headers["webhook-timestamp"] = str(timestamp)
        headers["webhook-signature"] = sign_webhook(secret, req.event_uuid, timestamp, body)
    _check_host(url, bool(config.get("allow_private_ips", True)))
    timeout = float(req.timeout_s or config.get("timeout_s") or 10)
    try:
        async with httpx.AsyncClient(follow_redirects=False, timeout=timeout) as client:
            response = await client.request(method, url, content=body, headers=headers)
    except httpx.HTTPError as error:
        raise TransientError(f"{type(error).__name__}: {error}") from error
    snippet = response.text[:MAX_RESPONSE]
    status = response.status_code
    if 200 <= status < 300:
        return ActionResult(rendered=rendered, http_status=status, response=snippet)
    if status == 410:
        raise PermanentError("410 Gone: the action was disabled", status, disable=True)
    if status == 429 or status >= 500:
        raise TransientError(
            f"HTTP {status}", status, _retry_after(response.headers.get("retry-after"))
        )
    if 300 <= status < 400:
        raise PermanentError(f"HTTP {status}: redirects are not followed", status)
    raise PermanentError(f"HTTP {status}: {snippet[:200]}", status)


async def run_email(req: ActionRequest, _deps: ActionDeps) -> ActionResult:
    """E-mail through the SMTP target (STARTTLS, SSL or plain)."""
    config = req.target_config
    params = req.params
    recipients = [str(r).strip() for r in params.get("to") or [] if str(r).strip()]
    if not recipients:
        raise PermanentError("the e-mail action has no recipients")
    if not config.get("host"):
        raise PermanentError("the SMTP target has no host")
    message = EmailMessage()
    message["From"] = str(
        config.get("from_address") or config.get("username") or "bottle-inspection@localhost"
    )
    message["To"] = ", ".join(recipients)
    message["Subject"] = render(str(params.get("subject") or ""), req.context).replace("\n", " ")[
        :250
    ]
    text = render(str(params.get("body") or ""), req.context)
    message.set_content(text)
    html = str(params.get("html") or "")
    if html.strip():
        message.add_alternative(render(html, req.context), subtype="html")
    if params.get("attach_image"):
        data = req.image_bytes or (
            req.image_path.read_bytes() if req.image_path and req.image_path.is_file() else None
        )
        if data:
            message.add_attachment(data, maintype="image", subtype="jpeg", filename="bottle.jpg")
    security = str(config.get("security") or "starttls")
    password = req.target_secrets.get("password")
    try:
        response = await aiosmtplib.send(
            message,
            hostname=str(config["host"]),
            port=int(config.get("port") or 587),
            username=str(config.get("username") or "") or None,
            password=password or None,
            use_tls=security == "ssl",
            start_tls=security == "starttls",
            timeout=float(req.timeout_s or config.get("timeout_s") or 15),
        )
    except aiosmtplib.SMTPAuthenticationError as error:
        raise PermanentError(f"SMTP authentication failed: {error}") from error
    except aiosmtplib.SMTPRecipientsRefused as error:
        raise PermanentError(f"recipients refused: {error}") from error
    except (aiosmtplib.SMTPException, OSError, asyncio.TimeoutError) as error:
        raise TransientError(f"{type(error).__name__}: {error}") from error
    return ActionResult(
        rendered=f"Subject: {message['Subject']}\nTo: {message['To']}\n\n{text}",
        response=str(response)[:MAX_RESPONSE],
    )


def _mqtt_publish_sync(
    config: dict[str, Any],
    secrets: dict[str, str],
    topic: str,
    payload: str,
    qos: int,
    retain: bool,
    timeout: float,
) -> None:
    """Connects, publishes one message and disconnects (paho-mqtt 2, blocking)."""
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion

    client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id=str(config.get("client_id") or ""))
    if config.get("username"):
        client.username_pw_set(str(config["username"]), secrets.get("password"))
    if config.get("tls"):
        client.tls_set(
            ca_certs=str(config.get("ca_file") or "") or None, tls_version=ssl.PROTOCOL_TLS_CLIENT
        )
    client.connect(str(config["host"]), int(config.get("port") or 1883), keepalive=30)
    client.loop_start()
    try:
        info = client.publish(topic, payload, qos=qos, retain=retain)
        info.wait_for_publish(timeout=timeout)
        if not info.is_published():
            raise TimeoutError("MQTT publish not acknowledged")
    finally:
        client.disconnect()
        client.loop_stop()


async def run_mqtt(req: ActionRequest, _deps: ActionDeps) -> ActionResult:
    """Publishes to the MQTT target."""
    config = req.target_config
    params = req.params
    if not config.get("host"):
        raise PermanentError("the MQTT target has no host")
    topic = render(str(params.get("topic") or ""), req.context)
    template = str(params.get("payload") or "")
    payload = (
        render(template, req.context)
        if template.strip()
        else json.dumps(req.context.get("analysis", {}), default=str)
    )
    qos = int(params.get("qos", config.get("qos", 1)))
    retain = bool(params.get("retain", config.get("retain", False)))
    try:
        await asyncio.get_running_loop().run_in_executor(
            None,
            _mqtt_publish_sync,
            config,
            req.target_secrets,
            topic,
            payload,
            qos,
            retain,
            float(req.timeout_s or 10),
        )
    except (OSError, TimeoutError, ValueError) as error:
        raise TransientError(f"{type(error).__name__}: {error}") from error
    return ActionResult(rendered=f"{topic}\n{payload}")


def _file_path(config: dict[str, Any], data_root: Path) -> Path:
    """Current output file of a `file` target (daily or size rotation)."""
    directory = Path(str(config.get("directory") or "exports"))
    if not directory.is_absolute():
        directory = data_root / directory
    directory.mkdir(parents=True, exist_ok=True)
    extension = "csv" if config.get("format") == "csv" else "jsonl"
    if config.get("rotation") == "size_mb":
        limit = float(config.get("size_mb") or 10) * 1024 * 1024
        index = 0
        while True:
            path = directory / f"bottle-inspection-{index:04d}.{extension}"
            if not path.exists() or path.stat().st_size < limit:
                return path
            index += 1
    return directory / f"bottle-inspection-{datetime.now():%Y-%m-%d}.{extension}"


def _field_value(context: dict[str, Any], name: str) -> Any:
    """A field of the write_file action (analysis fields, profile name, rule name...)."""
    analysis = context.get("analysis", {})
    if name == "profile":
        return context.get("profile", {}).get("name")
    if name == "rule":
        return context.get("rule", {}).get("name")
    if name.startswith("score_"):
        return analysis.get("scores", {}).get(name[6:])
    if name == "total_ms":
        return analysis.get("timings_ms", {}).get("total")
    value = analysis.get(name)
    return json.dumps(value) if isinstance(value, (dict, list)) else value


async def run_write_file(req: ActionRequest, deps: ActionDeps) -> ActionResult:
    """Appends one CSV or JSON line to the file target."""
    config = req.target_config
    fields = [str(f) for f in req.params.get("fields") or ACTION_TYPES["write_file"]["fields"]]
    row = {name: _field_value(req.context, name) for name in fields}
    path = _file_path(config, deps.data_root)
    if config.get("format") == "csv":
        new = not path.exists()
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        if new:
            writer.writerow(fields)
        writer.writerow([row[f] for f in fields])
        line = buffer.getvalue()
    else:
        line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
    return ActionResult(rendered=line.strip(), extra={"file": str(path)})


async def run_notify(req: ActionRequest, deps: ActionDeps) -> ActionResult:
    """A notification in the UI drawer (critical ones become persistent toasts)."""
    title = render(str(req.params.get("title") or ""), req.context)
    message = render(str(req.params.get("message") or ""), req.context)
    level = str(req.params.get("level") or "warning")
    deps.notify(level, title, message)
    return ActionResult(rendered=f"{title}\n{message}")


async def run_sound(req: ActionRequest, deps: ActionDeps) -> ActionResult:
    """Asks every open UI to play a sound."""
    sound = str(req.params.get("sound") or "chime")
    deps.publish_event(
        "rule.sound", {"sound": sound, "rule": req.context.get("rule", {}), "is_test": req.is_test}
    )
    return ActionResult(rendered=sound)


EXECUTORS: dict[str, Callable[[ActionRequest, ActionDeps], Awaitable[ActionResult]]] = {
    "notify_ui": run_notify,
    "play_sound": run_sound,
    "email": run_email,
    "webhook": run_webhook,
    "mqtt_publish": run_mqtt,
    "write_file": run_write_file,
}


async def execute(
    req: ActionRequest,
    deps: ActionDeps,
    on_attempt: Callable[[int], None] | None = None,
) -> Outcome:
    """Runs an action with retries: up to `max_retries` more attempts on transient errors."""
    start = time.perf_counter()
    executor = EXECUTORS.get(req.action_type)
    attempt = 0
    last_error: str | None = None
    last_status: int | None = None
    delays = RETRY_DELAYS[: max(0, req.max_retries)]
    while True:
        attempt += 1
        if on_attempt is not None:
            on_attempt(attempt)
        try:
            if executor is None:
                raise PermanentError(f"unknown action type {req.action_type}")
            result = await executor(req, deps)
            return Outcome(
                "success",
                attempt,
                result.rendered[:MAX_PAYLOAD],
                result.http_status,
                None,
                result.response[:MAX_RESPONSE],
                (time.perf_counter() - start) * 1000,
                extra=result.extra,
            )
        except TemplateRenderError as error:
            return Outcome(
                "failed",
                attempt,
                "",
                None,
                f"template: {error}",
                "",
                (time.perf_counter() - start) * 1000,
            )
        except PermanentError as error:
            return Outcome(
                "failed",
                attempt,
                "",
                error.http_status,
                str(error),
                "",
                (time.perf_counter() - start) * 1000,
                disable_action=error.disable,
            )
        except TransientError as error:
            last_error, last_status = str(error), error.http_status
            if attempt > len(delays):
                break
            delay = delays[attempt - 1] * random.uniform(0.8, 1.2)
            if error.retry_after is not None:
                delay = max(delay, min(error.retry_after, 300.0))
            await asyncio.sleep(delay)
    return Outcome(
        "failed", attempt, "", last_status, last_error, "", (time.perf_counter() - start) * 1000
    )
