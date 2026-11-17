"""Cameras: CRUD, test snapshot, server sessions (modes), browser WebSocket, storage policy."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ..app import create_app
from ..cameras.capture import CameraConfig
from ..cameras.service import CameraService, MotionTrigger
from .conftest import make_config, synthetic_background, synthetic_image


class FakeCapture:
    """A camera that returns synthetic bottle frames (or nothing when `broken`)."""

    def __init__(self, broken: bool = False) -> None:
        """Opened unless broken."""
        self.broken = broken
        self.count = 0

    def isOpened(self) -> bool:  # noqa: N802
        """True unless broken."""
        return not self.broken

    def read(self) -> tuple[bool, Any]:
        """A new synthetic frame each call."""
        self.count += 1
        time.sleep(0.01)
        return True, synthetic_image(self.count % 5)

    def release(self) -> None:
        """Nothing to release."""

    def set(self, propId: int, value: float) -> bool:  # noqa: N803
        """Accepts any property."""
        return True


def fake_factory(config: CameraConfig) -> FakeCapture:
    """Broken when the camera is called 'broken'."""
    return FakeCapture(broken=config.name == "broken")


@pytest.fixture()
def cam_client(tmp_path: Path, code_dir: Path) -> Iterator[TestClient]:
    """Test client whose camera service uses the fake capture."""
    app = create_app(make_config(tmp_path, code_dir))
    service: CameraService = app.state.ctx.extras["cameras"]
    service.factory = fake_factory
    with TestClient(app) as client:
        yield client


def svm_id(client: TestClient) -> int:
    """Id of the SVM fixture profile."""
    return next(
        p["id"] for p in client.get("/api/profiles").json() if p["name"] == "SVM rois_hog (fx)"
    )


def test_camera_crud_and_test(cam_client: TestClient) -> None:
    """Create/edit/delete; the test button returns a snapshot or a clear error."""
    bad = cam_client.post("/api/cameras", json={"name": "x", "kind": "rtsp", "url": "http://a"})
    assert bad.status_code == 422
    camera = cam_client.post(
        "/api/cameras", json={"name": "Line 1", "kind": "usb", "device_index": 0}
    ).json()
    snapshot = cam_client.post(f"/api/cameras/{camera['id']}/test").json()
    assert snapshot["ok"] and snapshot["width"] == 1280 and snapshot["snapshot"]
    broken = cam_client.post(
        "/api/cameras", json={"name": "broken", "kind": "usb", "device_index": 1}
    ).json()
    result = cam_client.post(f"/api/cameras/{broken['id']}/test").json()
    assert result == {"ok": False, "code": "camera.open_failed", "message": result["message"]}
    assert "usbipd" in result["message"]
    renamed = cam_client.patch(f"/api/cameras/{camera['id']}", json={"fps": 5}).json()
    assert renamed["fps"] == 5
    assert cam_client.delete(f"/api/cameras/{camera['id']}").status_code == 200
    assert [c["name"] for c in cam_client.get("/api/cameras").json()] == ["broken"]


def wait_for(predicate: Any, timeout: float = 15.0) -> None:
    """Polls until `predicate()` is true."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError("timed out")


def test_server_session_continuous_and_single(cam_client: TestClient) -> None:
    """Continuous mode analyses frames by itself; single mode only on trigger."""
    camera = cam_client.post(
        "/api/cameras", json={"name": "Line 1", "kind": "usb", "device_index": 0}
    ).json()
    pid = svm_id(cam_client)
    session = cam_client.post(
        "/api/sessions",
        json={
            "source": "server_camera",
            "camera_id": camera["id"],
            "profile_ids": [pid],
            "mode": "continuous",
            "fps": 10,
        },
    ).json()
    count = lambda: cam_client.get("/api/analyses", params={"session": session["id"]}).json()[
        "total"
    ]  # noqa: E731
    wait_for(lambda: count() >= 2)
    assert cam_client.get("/api/health").json()["active_sessions"] == 1
    cam_client.post(f"/api/sessions/{session['id']}/stop")
    assert cam_client.get("/api/health").json()["active_sessions"] == 0
    analyses = cam_client.get("/api/analyses", params={"session": session["id"]}).json()["items"]
    assert all(a["source"] == "server_camera" and len(a["scores"]) == 7 for a in analyses)

    single = cam_client.post(
        "/api/sessions",
        json={
            "source": "server_camera",
            "camera_id": camera["id"],
            "profile_ids": [pid],
            "mode": "single",
        },
    ).json()
    time.sleep(0.5)
    single_count = lambda: cam_client.get("/api/analyses", params={"session": single["id"]}).json()[
        "total"
    ]  # noqa: E731
    assert single_count() == 0
    cam_client.post(f"/api/sessions/{single['id']}/trigger")
    wait_for(lambda: single_count() == 1)
    cam_client.post(f"/api/sessions/{single['id']}/stop")


def test_storage_policy_rejects_only(cam_client: TestClient) -> None:
    """With rejects_only, camera frames are stored only when a profile does not pass;
    unstored frames are still available in memory for the live view."""
    pid = svm_id(cam_client)
    # Make every class a pass: nothing should be stored.
    cam_client.patch(f"/api/profiles/{pid}", json={"verdict": {"reject_classes": []}})
    session = cam_client.post(
        "/api/sessions",
        json={"source": "browser_camera", "profile_ids": [pid], "mode": "continuous"},
    ).json()
    ok, jpeg = cv2.imencode(".jpg", synthetic_image(1))
    assert ok
    with cam_client.websocket_connect(f"/ws/camera/browser?session_id={session['id']}") as ws:
        ws.send_bytes(jpeg.tobytes())
        message = ws.receive_json()
    assert message["type"] == "inspection"
    analysis = message["inspection"]["analyses"][0]
    assert analysis["verdict"] == "pass"
    assert analysis["image"]["id"] is None
    assert cam_client.get(analysis["image"]["url"]).status_code == 200
    assert cam_client.get("/api/storage").json()["images_rows"] == 0


def test_browser_camera_websocket(cam_client: TestClient) -> None:
    """Browser frames in, one inspection out per frame; rejects are stored."""
    pid = svm_id(cam_client)
    session = cam_client.post(
        "/api/sessions",
        json={"source": "browser_camera", "profile_ids": [pid], "mode": "continuous"},
    ).json()
    ok, jpeg = cv2.imencode(".jpg", synthetic_image(2))
    assert ok
    with cam_client.websocket_connect(f"/ws/camera/browser?session_id={session['id']}") as ws:
        for _ in range(2):
            ws.send_bytes(jpeg.tobytes())
            message = ws.receive_json()
            assert message["type"] == "inspection"
            assert list(message["inspection"]["analyses"][0]["scores"]) == [
                "bottle_no_defect",
                "bottle_other_brand",
                "defect_low_fill",
                "defect_no_cap",
                "defect_loose_cap",
                "defect_debris",
                "defect_label_damage",
            ]
        ws.send_bytes(b"not a jpeg")
        assert ws.receive_json() == {"type": "error", "code": "upload.not_an_image"}
    analyses = cam_client.get("/api/analyses", params={"session": session["id"]}).json()["items"]
    assert len(analyses) == 2 and all(a["source"] == "browser_camera" for a in analyses)
    stored = [a for a in analyses if a["verdict"] != "pass"]
    assert all(a["image"]["id"] is not None for a in stored)


def test_motion_trigger() -> None:
    """Fires when the fraction rises above the threshold after being low; re-arms below half."""
    trigger = MotionTrigger(0.1)
    fired = [trigger.update(f) for f in (0.0, 0.02, 0.2, 0.25, 0.08, 0.04, 0.3)]
    assert fired == [False, False, True, False, False, False, True]


def test_motion_fraction_uses_profile_background(cam_client: TestClient) -> None:
    """An empty scene (the background itself) has almost no foreground; a bottle has some."""
    from ..cameras.service import motion_fraction

    ctx = cam_client.app.state.ctx  # type: ignore[attr-defined]
    pid = svm_id(cam_client)
    empty = cv2.resize(synthetic_background(), (1280, 720))
    assert motion_fraction(empty, ctx, pid) < 0.01
    assert motion_fraction(synthetic_image(0), ctx, pid) > 0.05
    _ = np


def test_background_capture_makes_profile_modified(cam_client: TestClient) -> None:
    """REQ-M5: the per-pixel median of N frames becomes the profile's background."""
    camera = cam_client.post(
        "/api/cameras", json={"name": "Line 1", "kind": "usb", "device_index": 0}
    ).json()
    pid = svm_id(cam_client)
    response = cam_client.post(
        f"/api/profiles/{pid}/background/capture", json={"camera_id": camera["id"], "frames": 5}
    )
    assert response.status_code == 200, response.text
    profile = response.json()
    assert profile["background_source"] == "captured" and profile["has_background"]
    assert profile["status"] == "modified" and profile["version"] == 2
    image = cam_client.get(f"/api/profiles/{pid}/background")
    assert image.status_code == 200 and image.headers["content-type"] == "image/png"
