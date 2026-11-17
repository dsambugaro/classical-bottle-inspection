"""Reports on a seeded DB: every endpoint lists the 7 classes; filters; CSV; 100k rows < 1 s."""

from __future__ import annotations

import csv
from collections.abc import Iterator
import io
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import insert

from bottle_inspection.config import CLASSES

from ..app import create_app
from ..db.models import Analysis, Inspection
from .conftest import make_config

ENDPOINTS = ("summary", "timeseries", "pareto", "profiles", "confidence", "timings", "confusion")


def seed(client: TestClient, n: int, days: float = 6.0) -> list[int]:
    """Inserts `n` inspections, each analysed by every fixture profile of fx/rois_hog/."""
    ctx = client.app.state.ctx  # type: ignore[attr-defined]
    profiles = [
        p for p in client.get("/api/profiles").json() if p["model_ref"].startswith("fx/rois_hog/")
    ]
    rng = random.Random(1)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with ctx.db.engine.begin() as connection:
        inspections, analyses = [], []
        for i in range(1, n + 1):
            created = now - timedelta(seconds=rng.uniform(0, days * 86400))
            truth = rng.choice(CLASSES) if i % 3 == 0 else None
            inspections.append(
                {
                    "id": i,
                    "source": "dataset" if truth else "upload",
                    "created_at": created,
                    "ground_truth_class": truth,
                }
            )
            for profile in profiles:
                predicted = truth if truth and rng.random() < 0.9 else rng.choice(CLASSES)
                analyses.append(
                    {
                        "inspection_id": i,
                        "profile_id": profile["id"],
                        "profile_version": 1,
                        "modified_params": False,
                        "predicted_class": predicted,
                        "verdict": "pass" if predicted == "bottle_no_defect" else "reject",
                        "confidence": rng.random(),
                        "scores_json": {c: rng.random() for c in CLASSES},
                        "score_kind": "rf_probability",
                        "timings_json": {
                            "decode": 3.0,
                            "normalize": 5.0,
                            "foreground_mask": 2.0,
                            "localize": 0.4,
                            "crop_regions": 0.1,
                            "predict": 1.0,
                            "total": 30.0,
                            "features": {
                                "cap_hog": 2.0,
                                "label_hog": 1.0,
                                "body_hog": 2.0,
                            },
                        },
                        "total_ms": rng.uniform(25, 45),
                        "warnings_json": [],
                        "source": "dataset" if truth else "upload",
                        "ground_truth_class": truth,
                        "created_at": created,
                    }
                )
        connection.execute(insert(Inspection), inspections)
        connection.execute(insert(Analysis), analyses)
    return [p["id"] for p in profiles]


@pytest.fixture()
def report_client(tmp_path: Path, code_dir: Path) -> Iterator[TestClient]:
    """Client with an empty DB (seeded by each test)."""
    app = create_app(make_config(tmp_path, code_dir))
    with TestClient(app) as client:
        yield client


def test_reports_list_all_classes_and_filters(report_client: TestClient) -> None:
    """Every report has all 7 classes; filters, versions and the KPI delta work."""
    ids = seed(report_client, 300)
    summary = report_client.get("/api/reports/summary?preset=7d").json()
    assert [c["class"] for c in summary["class_distribution"]] == CLASSES
    assert summary["kpis"]["inspected"] == 300
    assert summary["kpis"]["analyses"] == 300 * len(ids)
    assert summary["previous"] is not None
    pareto = report_client.get("/api/reports/pareto").json()
    assert sorted(i["class"] for i in pareto["items"]) == sorted(CLASSES)
    assert pareto["items"][-1]["cumulative"] == pytest.approx(1.0)
    series = report_client.get("/api/reports/timeseries?preset=7d").json()
    assert series["bucket"] == "day"  # data spread over 6 days
    assert all(list(item["rejects_by_class"]) == CLASSES for item in series["series"])
    profiles = report_client.get("/api/reports/profiles").json()
    assert all(list(p["classes"]) == CLASSES for p in profiles["profiles"])
    assert len(profiles["agreement"]) == 3
    confusion = report_client.get("/api/reports/confusion").json()
    assert all(len(p["matrix"]) == 7 and p["accuracy"] > 0.5 for p in confusion["profiles"])
    timings = report_client.get("/api/reports/timings").json()
    assert timings["profiles"][0]["steps"]["normalize"] == pytest.approx(5.0)
    histogram = report_client.get("/api/reports/confidence").json()
    assert sum(histogram["profiles"][0]["bins"]) == 300
    one = report_client.get(f"/api/reports/summary?profile={ids[0]}&class=defect_no_cap").json()
    assert one["kpis"]["analyses"] == one["class_distribution"][3]["count"]
    assert (
        report_client.get(f"/api/reports/summary?version={ids[0]}:2").json()["kpis"]["analyses"]
        == 0
    )


@pytest.mark.parametrize("lang,delimiter,decimal", [("en", ",", "."), ("pt-BR", ";", ",")])
def test_csv_export(report_client: TestClient, lang: str, delimiter: str, decimal: str) -> None:
    """One row per analysis, the 7 score columns always present, locale formats."""
    seed(report_client, 20)
    response = report_client.get(f"/api/reports/export.csv?lang={lang}")
    text = response.content.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    header, body = rows[0], rows[1:]
    scores = [c for c in header if c.startswith("score_") and c != "score_kind"]
    assert scores == [f"score_{c}" for c in CLASSES]
    assert len(body) == 60
    confidence = body[0][header.index("confidence")]
    assert decimal in confidence


def test_reports_are_fast_with_100k_analyses(tmp_path: Path, code_dir: Path) -> None:
    """REQ-RP6: the reports screen (all its requests, 6 in parallel like a browser) loads in
    under 1 s with 100k analyses, through a real uvicorn server."""
    import socket
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import httpx
    import uvicorn

    config = make_config(tmp_path, code_dir)
    with TestClient(create_app(config)) as client:
        seed(client, 34_000)  # x 3 profiles = 102k analyses
    app = create_app(config)  # a fresh app: the lifespan above shut its executor down
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 30
        while not server.started:
            assert thread.is_alive() and time.time() < deadline, "server did not start"
            time.sleep(0.05)
        base = f"http://127.0.0.1:{port}"
        urls = [f"{base}/api/reports/{e}?preset=7d" for e in (*ENDPOINTS, "options")]
        urls.append(f"{base}/api/analyses?preset=7d&limit=50")
        with httpx.Client(timeout=30) as http, ThreadPoolExecutor(6) as pool:
            http.get(urls[0])
            walls = []
            for _ in range(3):
                start = time.perf_counter()
                codes = list(pool.map(lambda url: http.get(url).status_code, urls))
                walls.append(time.perf_counter() - start)
                assert set(codes) == {200}
        print("reports wall time (s):", [round(w, 3) for w in walls])
        assert min(walls) < 1.0, walls
    finally:
        server.should_exit = True
        thread.join(timeout=10)
