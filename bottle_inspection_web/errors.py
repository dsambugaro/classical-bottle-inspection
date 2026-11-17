"""API errors in the format {"error": {"code", "message", "details"}} (§14, REQ-L4)."""

from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """An error returned to the client; `code` is translated by the frontend."""

    def __init__(
        self, status: int, code: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        """Error with HTTP status, translatable code, English message and details."""
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}

    def body(self) -> dict[str, Any]:
        """JSON body."""
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}
