"""GET /api/history: cursor pagination, every filter, 'disagree', 'only wrong' and full rows."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert

from bottle_inspection.config import CLASSES

from ..app import create_app
from ..db.models import Analysis, Inspection
from .conftest import make_config

GOOD, CAP, LABEL = "bottle_no_defect", "defect_no_cap", "defect_label_damage"
BASE = datetime(2030, 1, 1, 12, 0, 0)

# id -> (source, ground truth, name, [(profile index, predicted class, confidence), ...])
SPECS: dict[int, tuple[str, str | None, str, list[tuple[int, str, float]]]] = {
    1: ("upload", None, "alpha.jpg", [(0, GOOD, 0.9), (1, GOOD, 0.8)]),
    2: ("dataset", GOOD, "beta_001.jpg", [(0, GOOD, 0.7), (1, CAP, 0.6)]),
    3: ("dataset", CAP, "beta_002.jpg", [(0, GOOD, 0.5), (1, GOOD, 0.4)]),
    4: ("upload", None, "gamma.jpg", [(0, CAP, 0.95), (1, LABEL, 0.2)]),
    5: ("dataset", LABEL, "delta.jpg", [(0, LABEL, 0.85), (1, LABEL, 0.75)]),
    6: ("upload", None, "100%_sure.jpg", [(0, CAP, 0.3)]),
}


@pytest.fixture()
def history(tmp_path: Path, code_dir: Path) -> Iterator[tuple[TestClient, list[int]]]:
    """A client on a DB with the inspections of SPECS (one per minute) and the profile ids."""
    app = create_app(make_config(tmp_path, code_dir))
    with TestClient(app) as client:
        ctx = client.app.state.ctx  # type: ignore[attr-defined]
        profiles = [
            p["id"]
            for p in client.get("/api/profiles").json()
            if p["model_ref"].startswith("fx/rois_hog/")
        ][:2]
        assert len(profiles) == 2
        inspections: list[dict[str, Any]] = []
        analyses: list[dict[str, Any]] = []
        for i, (source, truth, name, rows) in SPECS.items():
            created = BASE + timedelta(minutes=i)
            inspections.append(
                {
                    "id": i,
                    "source": source,
                    "created_at": created,
                    "ground_truth_class": truth,
                    "original_name": name,
                }
            )
            for index, predicted, confidence in rows:
                analyses.append(
                    {
                        "inspection_id": i,
                        "profile_id": profiles[index],
                        "profile_version": 1,
                        "modified_params": False,
                        "predicted_class": predicted,
                        "verdict": "pass" if predicted == GOOD else "reject",
                        "confidence": confidence,
                        "scores_json": {c: 0.1 for c in CLASSES},
                        "score_kind": "rf_probability",
                        "timings_json": {"total": 10.0},
                        "total_ms": 10.0,
                        "warnings_json": [],
                        "source": source,
                        "ground_truth_class": truth,
                        "created_at": created,
                    }
                )
        with ctx.db.engine.begin() as connection:
            connection.execute(insert(Inspection), inspections)
            connection.execute(insert(Analysis), analyses)
        yield client, profiles


def ids_of(client: TestClient, query: str = "") -> list[int]:
    """Inspection ids of the first page for a query string (newest first)."""
    response = client.get(f"/api/history?limit=100{query}")
    assert response.status_code == 200, response.text
    return [item["inspection_id"] for item in response.json()["items"]]


def test_pagination_has_no_duplicates_or_gaps(history: tuple[TestClient, list[int]]) -> None:
    """Following next_cursor visits every inspection once, newest first."""
    client, _ = history
    seen: list[int] = []
    cursor: int | None = None
    while True:
        url = "/api/history?limit=2" + (f"&before={cursor}" if cursor else "")
        page = client.get(url).json()
        seen += [item["inspection_id"] for item in page["items"]]
        assert len(page["items"]) <= 2
        if not page["has_more"]:
            assert page["next_cursor"] is None
            break
        cursor = page["next_cursor"]
        assert cursor == page["items"][-1]["inspection_id"]
    assert seen == [6, 5, 4, 3, 2, 1]
    exact = client.get("/api/history?limit=6").json()
    assert exact["has_more"] is False and exact["next_cursor"] is None
    assert client.get("/api/history?limit=1&before=1").json()["items"] == []


def test_rows_carry_all_analyses_in_the_inspection_shape(
    history: tuple[TestClient, list[int]],
) -> None:
    """A matching inspection returns every analysis, not only the matching ones."""
    client, profiles = history
    page = client.get(f"/api/history?profile={profiles[1]}&class={CAP}").json()
    assert [i["inspection_id"] for i in page["items"]] == [2]
    row = page["items"][0]
    assert [a["profile"]["id"] for a in row["analyses"]] == profiles
    assert [a["predicted_class"] for a in row["analyses"]] == [GOOD, CAP]
    assert row["original_name"] == "beta_001.jpg" and row["ground_truth"] == GOOD
    assert list(row["analyses"][0]["scores"]) == CLASSES
    detail = client.get("/api/inspections/2").json()
    assert detail["analyses"] == row["analyses"]


@pytest.mark.parametrize(
    "query,expected",
    [
        ("", [6, 5, 4, 3, 2, 1]),
        (f"&class={CAP}", [6, 4, 2]),
        (f"&class={CAP}&class={LABEL}", [6, 5, 4, 2]),
        (f"&class={GOOD},{LABEL}", [5, 4, 3, 2, 1]),
        ("&class=not_a_class", [6, 5, 4, 3, 2, 1]),
        ("&verdict=pass", [3, 2, 1]),
        ("&verdict=reject&verdict=review", [6, 5, 4, 2]),
        ("&source=dataset", [5, 3, 2]),
        ("&source=upload&source=dataset", [6, 5, 4, 3, 2, 1]),
        (f"&truth={GOOD}", [2]),
        (f"&truth={CAP}&truth={LABEL}", [5, 3]),
        ("&ground_truth_only=true", [5, 3, 2]),
        ("&only_wrong=true", [3, 2]),
        ("&disagree=true", [4, 2]),
        ("&disagree=true&only_wrong=true", [2]),
        ("&min_confidence=0.9", [4, 1]),
        ("&max_confidence=0.25", [4]),
        ("&min_confidence=0.45&max_confidence=0.55", [3]),
        ("&name=BETA", [3, 2]),
        ("&name=100%25", [6]),
        ("&inspection=4", [4]),
    ],
)
def test_filters(history: tuple[TestClient, list[int]], query: str, expected: list[int]) -> None:
    """Each filter selects the inspections with at least one matching analysis."""
    client, _ = history
    assert ids_of(client, query) == expected


def test_profile_filter_and_same_analysis_semantics(
    history: tuple[TestClient, list[int]],
) -> None:
    """All analysis-level filters must hold for the same analysis; disagree uses all of them."""
    client, profiles = history
    assert ids_of(client, f"&profile={profiles[1]}") == [5, 4, 3, 2, 1]
    assert ids_of(client, f"&profile={profiles[0]}&class={CAP}") == [6, 4]
    # inspection 4 has a CAP analysis (profile 0) and a LABEL one (profile 1): never both on one.
    assert ids_of(client, f"&profile={profiles[1]}&class={CAP}") == [2]
    # disagree looks at all analyses of the inspection, even when the filter hides some
    assert ids_of(client, f"&profile={profiles[0]}&disagree=true") == [4, 2]
    assert ids_of(client, f"&profile={profiles[0]}&class={LABEL}&disagree=true") == []


def test_date_range_and_pagination_with_filters(history: tuple[TestClient, list[int]]) -> None:
    """The date range applies, and a filtered list paginates without gaps."""
    client, _ = history
    start = (BASE + timedelta(minutes=2)).isoformat() + "Z"
    end = (BASE + timedelta(minutes=5)).isoformat() + "Z"
    assert ids_of(client, f"&start={start}&end={end}") == [4, 3, 2]
    first = client.get("/api/history?limit=2&source=dataset").json()
    assert [i["inspection_id"] for i in first["items"]] == [5, 3] and first["has_more"]
    second = client.get(f"/api/history?limit=2&source=dataset&before={first['next_cursor']}").json()
    assert [i["inspection_id"] for i in second["items"]] == [2] and not second["has_more"]


def test_camera_filter_and_deleted_image(history: tuple[TestClient, list[int]]) -> None:
    """The camera filter works and rows without a stored image degrade to a null url."""
    client, _ = history
    ctx = client.app.state.ctx  # type: ignore[attr-defined]
    with ctx.db.session() as db:
        db.get(Inspection, 4).camera_name = "cam-a"  # type: ignore[union-attr]
    assert ids_of(client, "&camera=cam-a") == [4]
    assert ids_of(client, "&camera=other") == []
    row = client.get("/api/history?limit=1").json()["items"][0]
    assert row["image"]["url"] is None and row["analyses"][0]["image"]["url"] is None
