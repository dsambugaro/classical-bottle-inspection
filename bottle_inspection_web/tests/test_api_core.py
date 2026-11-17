"""API tests: discovery (nested layout, broken models), profiles, status, inspections, storage."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bottle_inspection.config import CLASSES
from bottle_inspection.params import ARTIFACT_FORMAT_VERSION

from ..app import create_app
from .conftest import AUTO_BANDS, GROUP_SIZES, make_config


def profiles_by_name(client: TestClient) -> dict[str, dict]:
    """Profiles keyed by name."""
    return {p["name"]: p for p in client.get("/api/profiles").json()}


def test_classes_and_catalog(client: TestClient) -> None:
    """The 7 classes in id order; the catalog has editable and locked entries."""
    classes = client.get("/api/classes").json()
    assert [c["name"] for c in classes] == CLASSES
    assert classes[0]["default_verdict"] == "pass"
    assert all(c["default_verdict"] == "reject" for c in classes[1:])
    catalog = client.get("/api/params/catalog").json()
    modes = {e["key"]: e["mode"] for e in catalog["entries"]}
    assert modes["rois.cap"] == "editable" and modes["hog.cell_cap"] == "locked"
    assert next(e for e in catalog["entries"] if e["key"] == "rois.cap")["default"] == [0.0, 0.12]
    assert modes["localization.close_size"] == "editable"
    assert modes["hog.cell_auto"] == "locked" and modes["auto_rois.bands"] == "locked"
    removed = {"localization.diff_threshold", "localization.search_x_range", "bottle.width"}
    assert removed.isdisjoint(modes)


def test_discovery_and_seed_profiles(client: TestClient) -> None:
    """Every nested model is found with its feature vector; a model that fails to load is listed with its error."""
    models = {m["model_ref"]: m for m in client.get("/api/models").json()}
    fx = {
        "fx/rois_hog/svm",
        "fx/rois_hog/knn",
        "fx/rois_hog/random_forest",
        "fx/bottle_lbp/svm",
        "fx/auto_rois_color/svm",
    }
    assert set(models) == fx | {"broken/rois_hog/svm"}
    assert models["fx/rois_hog/svm"]["feature_vector"] == "rois_hog"
    assert models["fx/rois_hog/svm"]["folder"] == "fx"
    assert models["fx/rois_hog/svm"]["algorithm"] == "svm"
    assert models["fx/rois_hog/svm"]["feature_length"] == 1260 + 540 + 792
    assert models["fx/bottle_lbp/svm"]["feature_vector"] == "bottle_lbp"
    assert models["fx/bottle_lbp/svm"]["feature_length"] == GROUP_SIZES["bottle_lbp"]
    assert all(models[ref]["status"] == "ok" for ref in fx)
    assert models["broken/rois_hog/svm"]["status"] == "error"
    assert "feature_vector" in models["broken/rois_hog/svm"]["load_error"]
    full = client.get("/api/models/fx/rois_hog/svm").json()
    assert full["training_params"]["features.groups"] == ["cap_hog", "label_hog", "body_hog"]
    assert list(full["metrics"]["per_class"]) == CLASSES
    assert len(full["metrics"]["confusion_matrix"]) == 7
    assert full["metrics"]["feature_vector"] == "rois_hog"
    assert full["metrics"]["format_version"] == ARTIFACT_FORMAT_VERSION
    assert full["training_params"]["auto_rois.bands"] == [list(b) for b in AUTO_BANDS]
    profiles = profiles_by_name(client)
    # Every feature vector (rois_*, auto_rois_*, bottle_*) gets a seed profile.
    assert set(profiles) == {
        "SVM rois_hog (fx)",
        "KNN rois_hog (fx)",
        "RF rois_hog (fx)",
        "SVM bottle_lbp (fx)",
        "SVM auto_rois_color (fx)",
    }
    assert not any("broken" in name for name in profiles)
    descriptions = {name: p["description"] for name, p in profiles.items()}
    assert len({descriptions[n] for n in ("SVM rois_hog (fx)", "SVM bottle_lbp (fx)")}) == 2
    assert "Automatic regions" in descriptions["SVM auto_rois_color (fx)"]
    assert all(p["is_builtin"] and p["status"] == "ready" for p in profiles.values())


def test_discovery_of_a_feature_vector_folder(tmp_path: Path, code_dir: Path) -> None:
    """A model folder that directly holds `<algorithm>/model.joblib` (one feature vector folder
    of the CLI output) works too; its reference is `<folder>/<algorithm>`."""
    app = create_app(make_config(tmp_path, code_dir, folders=["fx/bottle_lbp", "fx/rois_hog"]))
    with TestClient(app) as client:
        models = {m["model_ref"]: m for m in client.get("/api/models").json()}
        assert set(models) == {
            "fx/bottle_lbp/svm",
            "fx/rois_hog/svm",
            "fx/rois_hog/knn",
            "fx/rois_hog/random_forest",
        }
        assert models["fx/rois_hog/knn"]["feature_vector"] == "rois_hog"
        assert models["fx/rois_hog/knn"]["folder"] == "fx"
        assert models["fx/rois_hog/knn"]["accuracy"] == 1.0


def test_scan_folders_layouts(tmp_path: Path, code_dir: Path) -> None:
    """scan_folders yields nested and direct models with their references and metric folders."""
    from ..models.discovery import scan_folders

    nested = scan_folders([("fx", code_dir / "fx"), ("missing", tmp_path / "nope")])
    assert [loc.ref for loc in nested] == [
        "fx/auto_rois_color/svm",
        "fx/bottle_lbp/svm",
        "fx/rois_hog/knn",
        "fx/rois_hog/random_forest",
        "fx/rois_hog/svm",
    ]
    assert all(loc.directory == code_dir / "fx" / loc.vector_dir for loc in nested)
    direct = scan_folders([("fx/bottle_lbp", code_dir / "fx" / "bottle_lbp")])
    assert [(loc.ref, loc.vector_dir) for loc in direct] == [("fx/bottle_lbp/svm", None)]
    assert direct[0].directory == code_dir / "fx" / "bottle_lbp"


def test_profile_crud_versions_diff(client: TestClient) -> None:
    """Create, edit (version bump, modified status), diff, restore, duplicate, delete."""
    created = client.post("/api/profiles", json={"name": "Mine", "model_ref": "fx/rois_hog/svm"})
    assert created.status_code == 201, created.text
    profile = created.json()
    assert profile["version"] == 1 and profile["status"] == "ready"
    pid = profile["id"]

    edited = client.patch(f"/api/profiles/{pid}", json={"params": {"rois.cap": [0.07, 0.2]}})
    assert edited.status_code == 200, edited.text
    assert edited.json()["version"] == 2
    assert edited.json()["status"] == "modified"
    assert "rois.cap" in edited.json()["status_detail"]["modified_keys"]

    renamed = client.patch(f"/api/profiles/{pid}", json={"name": "Mine 2"})
    assert renamed.json()["version"] == 2, "a rename is not a versioned change"

    locked = client.patch(f"/api/profiles/{pid}", json={"params": {"hog.cell_cap": [4, 4]}})
    assert locked.status_code == 422
    assert locked.json()["error"]["details"]["errors"][0]["code"] == "param.locked"

    invalid = client.patch(f"/api/profiles/{pid}", json={"params": {"rois.cap": [0.3, 0.1]}})
    assert invalid.status_code == 422

    versions = client.get(f"/api/profiles/{pid}/versions").json()
    assert [v["version"] for v in versions] == [2, 1]
    assert versions[0]["changes"][0]["key"] == "rois.cap"
    diff = client.get(f"/api/profiles/{pid}/diff", params={"from": 1, "to": 2}).json()
    assert [c["key"] for c in diff["changes"]] == ["rois.cap"]

    restored = client.post(f"/api/profiles/{pid}/versions/1/restore").json()
    assert restored["version"] == 3 and restored["status"] == "ready"

    copy = client.post(f"/api/profiles/{pid}/duplicate").json()
    assert copy["name"] == "Mine 2 (copy)" and not copy["is_builtin"]

    assert client.delete(f"/api/profiles/{pid}").status_code == 200
    assert "Mine 2" not in profiles_by_name(client)
    builtin = profiles_by_name(client)["SVM rois_hog (fx)"]
    assert client.delete(f"/api/profiles/{builtin['id']}").status_code == 409


def test_export_import(client: TestClient) -> None:
    """Export then import with a dry-run preview diff; unknown model_ref is rejected."""
    pid = profiles_by_name(client)["KNN rois_hog (fx)"]["id"]
    exported = client.get(f"/api/profiles/{pid}/export").json()
    assert "model" not in exported and exported["model_ref"] == "fx/rois_hog/knn"
    exported["params"]["localization.close_size"] = 21
    dry = client.post("/api/profiles/import", json=exported).json()
    assert dry["dry_run"] is True
    assert dry["preview"]["diff_from_training"][0]["key"] == "localization.close_size"
    applied = client.post("/api/profiles/import?dry_run=false", json=exported).json()
    assert applied["profile"]["status"] == "modified"
    exported["model_ref"] = "nope/svm"
    assert client.post("/api/profiles/import", json=exported).status_code == 422


def test_inspection_upload_returns_all_scores(client: TestClient, jpeg: bytes) -> None:
    """Upload with several profiles: one result per profile, all 7 scores, timings."""
    profiles = profiles_by_name(client)
    ids = [
        profiles[n]["id"] for n in ("SVM rois_hog (fx)", "KNN rois_hog (fx)", "RF rois_hog (fx)")
    ]
    response = client.post(
        "/api/inspections",
        data={"profile_ids": ",".join(map(str, ids))},
        files={"files": ("bottle.jpg", jpeg, "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    inspection = response.json()["inspections"][0]
    assert [a["profile"]["id"] for a in inspection["analyses"]] == ids
    kinds = set()
    for analysis in inspection["analyses"]:
        assert list(analysis["scores"]) == CLASSES
        assert analysis["predicted_class"] in CLASSES
        assert 0.0 <= analysis["confidence"] <= 1.0
        assert analysis["verdict"] in ("pass", "reject", "review")
        assert analysis["timings_ms"]["total"] > 0
        assert set(analysis["timings_ms"]["features"]) == {"cap_hog", "label_hog", "body_hog"}
        assert "adhoc" not in analysis
        assert set(analysis["geometry"]["rois"]) == {"cap", "label", "body"}
        kinds.add(analysis["score_kind"])
    assert kinds == {"svm_ovr_score", "knn_class_distance", "rf_probability"}
    assert len(inspection["analyses"][0]["details"]["svm_votes"]) == 7
    image_url = inspection["image"]["url"]
    assert client.get(image_url).status_code == 200
    assert client.get(inspection["image"]["thumb_url"]).status_code == 200
    stored = client.get(f"/api/inspections/{inspection['inspection_id']}").json()
    assert len(stored["analyses"]) == 3
    listed = client.get("/api/analyses", params={"profile": ids[0]}).json()
    assert listed["total"] == 1


def test_inspection_with_automatic_regions_and_whole_bottle(
    client: TestClient, jpeg: bytes
) -> None:
    """auto_rois_* and bottle_* models: the groups of their own vector, the regions they use in
    the geometry (automatic ones from the learned bands), the new step names, no edge warning."""
    profiles = profiles_by_name(client)
    auto, bottle = profiles["SVM auto_rois_color (fx)"], profiles["SVM bottle_lbp (fx)"]
    response = client.post(
        "/api/inspections",
        data={"profile_ids": f"{auto['id']},{bottle['id']}"},
        files={"files": ("bottle.jpg", jpeg, "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    by_profile = {a["profile"]["id"]: a for a in response.json()["inspections"][0]["analyses"]}
    auto_result, bottle_result = by_profile[auto["id"]], by_profile[bottle["id"]]
    assert set(auto_result["timings_ms"]["features"]) == {f"auto_{k}_color" for k in (1, 2, 3)}
    assert set(auto_result["geometry"]["rois"]) == {"auto_1", "auto_2", "auto_3"}
    assert set(bottle_result["timings_ms"]["features"]) == {"bottle_lbp"}
    assert bottle_result["geometry"]["rois"] == {}
    for result in (auto_result, bottle_result):
        assert result["predicted_class"] in CLASSES
        timings = result["timings_ms"]
        assert {"normalize", "foreground_mask", "localize", "crop_regions"} <= set(timings)
        assert "find_top" not in timings and "crop_rois" not in timings
        assert "localization_at_edge" not in result["warnings"]
        assert not {"search_x_range", "localization_y_range"} & set(result["geometry"])
    box = auto_result["geometry"]["bottle_box"]
    first, last = (
        auto_result["geometry"]["rois"]["auto_1"],
        auto_result["geometry"]["rois"]["auto_3"],
    )
    assert first[1] == pytest.approx(box[1], abs=1e-4)
    assert last[3] == pytest.approx(box[3], abs=1e-4)


def test_invalid_upload_is_rejected(client: TestClient) -> None:
    """Files that do not decode as images get 415."""
    pid = profiles_by_name(client)["SVM rois_hog (fx)"]["id"]
    response = client.post(
        "/api/inspections",
        data={"profile_ids": str(pid)},
        files={"files": ("fake.jpg", b"not an image", "image/jpeg")},
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "upload.not_an_image"


def test_delete_all_images_keeps_reports(client: TestClient, jpeg: bytes) -> None:
    """REQ-D1: images are deleted, analyses stay."""
    pid = profiles_by_name(client)["SVM rois_hog (fx)"]["id"]
    inspection = client.post(
        "/api/inspections",
        data={"profile_ids": str(pid)},
        files={"files": ("a.jpg", jpeg, "image/jpeg")},
    ).json()["inspections"][0]
    assert client.get("/api/storage").json()["image_files"] >= 1
    assert client.request("DELETE", "/api/images", json={"confirm": False}).status_code == 400
    deleted = client.request("DELETE", "/api/images", json={"confirm": True}).json()
    assert deleted["deleted_files"] >= 1
    assert client.get(inspection["image"]["url"]).status_code == 404
    assert client.get("/api/analyses").json()["total"] == 1
    after = client.get(f"/api/inspections/{inspection['inspection_id']}").json()
    assert after["image"]["id"] is None
    wrong = client.request("DELETE", "/api/analyses", json={"confirm": "yes"})
    assert wrong.status_code == 400
    client.request("DELETE", "/api/analyses", json={"confirm": "DELETE"})
    assert client.get("/api/analyses").json()["total"] == 0


def test_preview(client: TestClient, jpeg: bytes) -> None:
    """Preview returns geometry, masks and the modified keys of unsaved params."""
    pid = profiles_by_name(client)["SVM rois_hog (fx)"]["id"]
    response = client.post(
        f"/api/profiles/{pid}/preview",
        data={"params": '{"rois.cap": [0.05, 0.2]}', "classify": "true"},
        files={"file": ("a.jpg", jpeg, "image/jpeg")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    box = body["geometry"]["bottle_box"]
    cap = body["geometry"]["rois"]["cap"]
    # The band is inside the detected box: 0.05 of its height, not of the image.
    assert cap[1] == pytest.approx(box[1] + 0.05 * (box[3] - box[1]), abs=1e-4)
    assert cap[3] == pytest.approx(box[1] + 0.2 * (box[3] - box[1]), abs=1e-4)
    assert body["modified_keys"] == ["rois.cap"]
    assert body["masks"]["foreground"]
    assert list(body["classification"]["scores"]) == CLASSES


def test_status_model_missing_incompatible_and_update(tmp_path: Path) -> None:
    """model_missing when the folder disappears; incompatible when locked params change;
    'model updated' when the file changes (make run executed again)."""
    import joblib

    from .conftest import write_model_folder

    work = tmp_path / "code"
    write_model_folder(work / "m")
    app = create_app(make_config(tmp_path, work, folders=["m"]))
    with TestClient(app) as client:
        svm = profiles_by_name(client)["SVM rois_hog (m)"]
        assert svm["status"] == "ready"

        path = work / "m" / "rois_hog" / "svm" / "model.joblib"
        artifact = joblib.load(path)
        artifact["params"]["hog"]["cell_cap"] = [4, 4]
        joblib.dump(artifact, path)
        os.utime(path, (path.stat().st_atime, path.stat().st_mtime + 10))
        client.post("/api/models/refresh")
        changed = client.get(f"/api/profiles/{svm['id']}").json()
        assert changed["status"] == "incompatible"
        assert changed["model"]["updated_on"] is not None
        relinked = client.post(f"/api/profiles/{svm['id']}/relink").json()
        assert relinked["status"] == "ready"

        (work / "m" / "rois_hog" / "knn" / "model.joblib").unlink()
        client.post("/api/models/refresh")
        knn = profiles_by_name(client)["KNN rois_hog (m)"]
        assert knn["status"] == "model_missing"
        response = client.post(
            "/api/inspections",
            data={"profile_ids": str(knn["id"])},
            files={"files": ("a.jpg", b"x", "image/jpeg")},
        )
        assert response.status_code == 409


def test_empty_model_folders(tmp_path: Path) -> None:
    """Without trained models the app starts and lists nothing."""
    app = create_app(make_config(tmp_path, tmp_path / "empty", folders=["results"]))
    with TestClient(app) as client:
        assert client.get("/api/models").json() == []
        assert client.get("/api/profiles").json() == []
        assert client.get("/api/health").json()["models"] == 0


def test_sse_publishes_analysis_completed(client: TestClient, jpeg: bytes) -> None:
    """REQ-I10: an `analysis.completed` event is published for every analysis."""
    import threading

    pid = profiles_by_name(client)["SVM rois_hog (fx)"]["id"]
    events: list[str] = []

    def post_later() -> None:
        """Runs an inspection once the SSE stream is open."""
        client.post(
            "/api/inspections",
            data={"profile_ids": str(pid)},
            files={"files": ("a.jpg", jpeg, "image/jpeg")},
        )

    with client.stream("GET", "/api/events") as stream:
        worker = threading.Thread(target=post_later)
        worker.start()
        for line in stream.iter_lines():
            if line.startswith("event:"):
                events.append(line.split(":", 1)[1].strip())
            if "inspection.completed" in events:
                break
        worker.join()
    assert events[:2] == ["analysis.completed", "inspection.completed"]
