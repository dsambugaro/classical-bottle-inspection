"""Secrets (REQ-IN2): never stored in app.db nor returned by the API.

They live in `webdata/secrets.toml` (mode 0600). The DB references them by name; an
environment variable `BOTTLE_WEB_SECRET_<NAME>` overrides the file.
"""

from __future__ import annotations

import os
import re
import threading
import tomllib
from pathlib import Path

import tomli_w

MASK: str = "••••"
ENV_PREFIX: str = "BOTTLE_WEB_SECRET_"


def secret_name(target_id: int, field: str) -> str:
    """Name of a target's secret field, e.g. `target_3_password`."""
    return f"target_{target_id}_{field}"


class SecretStore:
    """Reads and writes the secrets file (thread-safe)."""

    def __init__(self, path: Path) -> None:
        """The file is created on the first write."""
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict[str, str]:
        """Current file contents."""
        if not self.path.exists():
            return {}
        with self.path.open("rb") as handle:
            data = tomllib.load(handle)
        return {str(k): str(v) for k, v in data.get("secrets", {}).items()}

    def _write(self, data: dict[str, str]) -> None:
        """Writes the file with mode 0600."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as handle:
            tomli_w.dump({"secrets": data}, handle)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)

    def get(self, name: str) -> str | None:
        """Secret value (environment first), or None."""
        env = os.environ.get(ENV_PREFIX + name.upper())
        if env is not None:
            return env
        with self._lock:
            return self._read().get(name)

    def set(self, name: str, value: str) -> None:
        """Stores a secret."""
        with self._lock:
            data = self._read()
            data[name] = value
            self._write(data)

    def delete(self, name: str) -> None:
        """Removes a secret."""
        with self._lock:
            data = self._read()
            if data.pop(name, None) is not None:
                self._write(data)

    def delete_prefix(self, prefix: str) -> None:
        """Removes every secret whose name starts with `prefix`."""
        with self._lock:
            data = self._read()
            kept = {k: v for k, v in data.items() if not k.startswith(prefix)}
            if kept != data:
                self._write(kept)

    def has(self, name: str) -> bool:
        """True when the secret exists (file or environment)."""
        return self.get(name) is not None


def redact(text: str, secrets: list[str]) -> str:
    """Replaces every secret value (and Authorization-like headers) by the mask."""
    for value in secrets:
        if value and len(value) >= 3:
            text = text.replace(value, MASK)
    keys = r"(?:authorization|proxy-authorization|x-api-key|api[-_]?key|password|token|secret)"
    # Quoted values ("Authorization": "Bearer abc") are masked whole, spaces included.
    text = re.sub(
        r'("' + keys + r'"\s*:\s*")([^"]*)(")',
        lambda m: m.group(1) + MASK + m.group(3),
        text,
        flags=re.IGNORECASE,
    )
    # Unquoted forms (password=abc, Authorization: Bearer abc) up to the end of the value.
    return re.sub(
        r"(\b" + keys + r"\b\s*[:=]\s*)(?!\"|" + re.escape(MASK) + r")([^\n,;}&]+)",
        lambda m: m.group(1) + MASK,
        text,
        flags=re.IGNORECASE,
    )
