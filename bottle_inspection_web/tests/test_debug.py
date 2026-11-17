"""Debug mode: step-by-step trace of one image kept in memory, run by any profile."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from bottle_inspection.config import LBP_POINTS
from bottle_inspection.params import InspectionParams
from bottle_inspection.preprocessing import (
    foreground_mask,
    locate_bottle,
    normalize_lighting,
    reference_patch_mean,
)

from ..engine.trace import mask_steps, profile_steps
from .conftest import COLOR_SIZE, synthetic_background
from .test_api_core import profiles_by_name


def test_debug_session_traces_preprocessing_and_profiles(client: TestClient, jpeg: bytes) -> None:
    """Debug mode: one image, every preprocessing step, then any profile; nothing is stored."""
    profiles = profiles_by_name(client)
    svm, knn = profiles["SVM rois_hog (fx)"]["id"], profiles["KNN rois_hog (fx)"]["id"]
    created = client.post("/api/debug/sessions", files={"file": ("a.jpg", jpeg, "image/jpeg")})
    assert created.status_code == 201, created.text
    session = created.json()
    assert (session["width"], session["height"]) == (1280, 720)
    assert client.get(session["image_url"]).content == jpeg
    base = f"/api/debug/sessions/{session['id']}"

    trace = client.post(f"{base}/preprocessing/{svm}", json={}).json()
    assert trace["original"]["shape"] == [720, 1280, 3]
    assert (
        len(trace["localization"]["columns"]) == 640 and len(trace["localization"]["rows"]) == 360
    )
    assert sum(trace["difference"]["histogram"]) == 640 * 360
    assert {r["name"] for r in trace["regions"]} >= {"cap", "label", "body", "auto_1", "bottle"}
    for key in ("raw", "closed", "opened"):
        assert trace["mask"][key].startswith("data:image/png;base64,")
    defaults = InspectionParams.defaults()
    x0, x1 = trace["localization"]["x"]
    box = trace["localization"]["box"]
    assert box[2] - box[0] == pytest.approx(defaults.localization.bottle_width, abs=1e-4)
    assert x0 < trace["localization"]["centre"] < x1
    assert (box[0] + box[2]) / 2 == pytest.approx(trace["localization"]["centre"] / 640, abs=1e-3)

    # The stages shown are those of the pipeline: the last one is foreground_mask itself.
    background = synthetic_background()
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    small = cv2.resize(normalize_lighting(image, reference_patch_mean(background)), (640, 360))
    mask = foreground_mask(small, background)
    assert np.array_equal(mask_steps(small, background, defaults)["opened"], mask)
    box = locate_bottle(mask)
    steps = profile_steps(mask, defaults)
    assert steps["centre"] / 640 == pytest.approx(box.center_x, abs=1e-6)
    assert steps["x"][0] < steps["centre"] < steps["x"][1]

    # Each profile runs on the same session; the result equals a normal inspection.
    for pid in (svm, knn):
        result = client.post(f"{base}/profiles/{pid}", json={}).json()
        assert result["preprocessing_key"] == trace["key"]
        assert [g["name"] for g in result["groups"]] == ["cap_hog", "label_hog", "body_hog"]
        assert sum(g["length"] for g in result["groups"]) == result["vector"]["length"] == 2592
        assert result["groups"][1]["offset"] == 1260
        assert len(result["scaled"]["values"]) == 2592
        inspected = client.post(
            "/api/inspections",
            data={"profile_ids": str(pid)},
            files={"files": ("a.jpg", jpeg, "image/jpeg")},
        ).json()["inspections"][0]["analyses"][0]
        assert result["analysis"]["predicted_class"] == inspected["predicted_class"]
        assert result["analysis"]["scores"] == inspected["scores"]
        assert result["analysis"]["geometry"] == inspected["geometry"]
    before = client.get("/api/analyses").json()["total"]
    client.post(f"{base}/profiles/{svm}", json={})
    assert client.get("/api/analyses").json()["total"] == before

    # Temporary overrides change the trace but never the profile.
    wide = client.post(
        f"{base}/preprocessing/{svm}", json={"overrides": {"localization.column_fraction": 0.05}}
    ).json()
    assert wide["key"] != trace["key"] and wide["params"]["localization.column_fraction"] == 0.05
    changed = client.post(
        f"{base}/profiles/{svm}", json={"overrides": {"localization.column_fraction": 0.05}}
    ).json()
    assert changed["preprocessing_key"] == wide["key"]
    assert changed["analysis"]["profile"]["status"] == "modified"
    assert profiles_by_name(client)["SVM rois_hog (fx)"]["status"] == "ready"
    invalid = client.post(f"{base}/profiles/{svm}", json={"overrides": {"hog.orientations": 5}})
    assert (
        invalid.status_code == 422 and invalid.json()["error"]["code"] == "profile.invalid_params"
    )

    # Other descriptors, a dataset-less app and the end of the session.
    lbp = client.post(f"{base}/profiles/{profiles['SVM bottle_lbp (fx)']['id']}", json={}).json()
    assert lbp["groups"][0]["lbp"]["n_codes"] == LBP_POINTS + 2
    assert lbp["groups"][0]["lbp"]["cells"] == [6, 24]
    color = client.post(
        f"{base}/profiles/{profiles['SVM auto_rois_color (fx)']['id']}", json={}
    ).json()
    assert [g["length"] for g in color["groups"]] == [COLOR_SIZE] * 3 and "h" in color["groups"][0][
        "color"
    ]
    assert client.delete(base).json() == {"deleted": True}
    assert client.post(f"{base}/profiles/{svm}", json={}).status_code == 404
    assert client.get(f"{base}/image").json()["error"]["code"] == "debug.session_not_found"
