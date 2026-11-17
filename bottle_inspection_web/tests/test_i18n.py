"""The UI has a translation for every catalog parameter and for the 7 classes (REQ-L3)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from ..classes import CLASS_NAMES
from ..params_catalog import CATALOG
from .conftest import CODE_DIR

LOCALES = CODE_DIR / "frontend" / "src" / "locales"


def lookup(tree: dict[str, Any], key: str) -> Any:
    """Value of a dot-path key, or None."""
    node: Any = tree
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


@pytest.mark.parametrize("language", ["en", "pt-BR"])
def test_catalog_and_classes_are_translated(language: str) -> None:
    """Every label/help key of the catalog and every class description exists."""
    tree = json.loads((LOCALES / f"{language}.json").read_text())
    missing = [
        key
        for entry in CATALOG
        for key in (entry.label_key, entry.help_key)
        if not isinstance(lookup(tree, key), str)
    ]
    missing += [f"classes.{name}" for name in CLASS_NAMES if not lookup(tree, f"classes.{name}")]
    assert missing == []


def backend_codes(pattern: str) -> set[str]:
    """String literals matching `pattern` in the backend sources (not the tests)."""
    import re

    package = CODE_DIR / "bottle_inspection_web"
    codes: set[str] = set()
    for path in package.rglob("*.py"):
        if "tests" in path.parts:
            continue
        codes |= set(re.findall(pattern, path.read_text()))
    return codes


@pytest.mark.parametrize("language", ["en", "pt-BR"])
def test_backend_codes_are_translated(language: str) -> None:
    """REQ-L4: every error code raised by the API and every notification code has a text."""
    tree = json.loads((LOCALES / f"{language}.json").read_text())
    errors = backend_codes(r'ApiError\(\s*\d{3},\s*"([a-z_]+\.[a-z_0-9]+)"')
    errors |= backend_codes(
        r'(?:CameraError|ConditionError|SettingsError)\([^)]*?"([a-z_]+\.[a-z_0-9]+)"'
    )
    notes = backend_codes(r'code="([a-z_]+\.[a-z_0-9]+)"')
    missing = [
        f"errors.{c}" for c in sorted(errors) if not isinstance(lookup(tree, f"errors.{c}"), str)
    ]
    missing += [
        f"notifications.{c}"
        for c in sorted(notes)
        if not isinstance(lookup(tree, f"notifications.{c}"), str)
    ]
    assert len(errors) > 30
    assert missing == []
