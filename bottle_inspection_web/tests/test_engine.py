"""Engine: same features as the CLI path, confidence formulas, catalog validation, parity."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bottle_inspection.config import AUTO_ROIS, FEATURE_VECTORS, WHOLE_BOTTLE
from bottle_inspection.extraction import extract_image
from bottle_inspection.params import InspectionParams

from ..engine.analysis import StepTimer, features_timed, geometry_of, prepare_timed
from ..engine.scores import knn_confidence, svm_confidence
from ..models.discovery import ModelLocation, scan_folders
from ..params_catalog import catalog_json, validate_params
from .conftest import AUTO_BANDS, CODE_DIR, synthetic_background, synthetic_image

DATASET = CODE_DIR.parent / "archive"


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize(
    "vector", ["rois_hog", "bottle_lbp", "auto_rois_hog", "auto_rois_lbp", "auto_rois_color"]
)
@pytest.mark.parametrize(
    "changes",
    [
        {},
        {
            "rois.cap": [0.05, 0.2],
            "localization.close_size": 9,
            "localization.column_fraction": 0.3,
            "localization.row_fraction": 0.4,
        },
    ],
)
def test_timed_steps_equal_extract_image(normalize: bool, vector: str, changes: dict) -> None:
    """For the groups of a model's vector the step-by-step engine computes exactly what
    extraction.extract_image computes (params of a trained model: with the learned automatic
    region bands), and only those groups are computed."""
    groups = tuple(FEATURE_VECTORS[vector])
    params = (
        InspectionParams.from_dict(changes)
        .with_auto_bands(AUTO_BANDS)
        .with_normalization(normalize)
        .with_feature_groups(groups)
    )
    image, background = synthetic_image(3), synthetic_background()
    prepared, prep_timings = prepare_timed(image, background, params, StepTimer())
    assert set(prepared.rois) == {"cap", "label", "body", *AUTO_ROIS, WHOLE_BOTTLE}
    assert list(prep_timings) == ["normalize", "foreground_mask", "localize", "crop_regions"]
    features, timings = features_timed(prepared, params, StepTimer())
    assert list(features) == list(groups) == list(timings)
    expected, view = extract_image(image, background, normalize, params=params)
    for group in groups:
        assert np.array_equal(features[group], expected[group]), group
    vector_x = np.concatenate([features[g] for g in groups])
    assert np.array_equal(vector_x, np.concatenate([expected[g] for g in groups]))
    assert prepared.box == view.box


@pytest.mark.parametrize(
    "vector, regions",
    [
        ("rois_hog", ["cap", "label", "body"]),
        ("auto_rois_color", list(AUTO_ROIS)),
        ("bottle_lbp", []),
    ],
)
def test_geometry_has_the_regions_of_the_model(vector: str, regions: list[str]) -> None:
    """The box is the detected one and the rectangles of the regions of the model's feature
    groups are the bands of the params inside it (image coordinates)."""
    params = (
        InspectionParams.defaults()
        .with_auto_bands(AUTO_BANDS)
        .with_feature_groups(tuple(FEATURE_VECTORS[vector]))
    )
    prepared, _ = prepare_timed(synthetic_image(3), synthetic_background(), params, StepTimer())
    geometry = geometry_of(prepared, params)
    box = prepared.box
    assert geometry["bottle_box"] == pytest.approx([box.x0, box.y0, box.x1, box.y1], abs=1e-5)
    assert geometry["bottle_center_x"] == pytest.approx(box.center_x, abs=1e-5)
    assert geometry["bottle_top_y"] == pytest.approx(box.y0, abs=1e-5)
    assert list(geometry["rois"]) == regions
    bands = params.region_bands()
    for name, (x0, y0, x1, y1) in geometry["rois"].items():
        assert (x0, x1) == pytest.approx((box.x0, box.x1), abs=1e-5)
        assert y0 == pytest.approx(box.y0 + bands[name][0] * box.height, abs=1e-5)
        assert y1 == pytest.approx(box.y0 + bands[name][1] * box.height, abs=1e-5)
    assert 0.0 < geometry["foreground_fraction"] < 1.0
    assert not {"search_x_range", "localization_y_range"} & set(geometry)
    # The synthetic bottle spans x 0.515..0.66 and y 0.083..0.903 of the image.
    assert box.height == pytest.approx(0.82, abs=0.03)


def test_confidence_formulas() -> None:
    """REQ-R2 definitions."""
    assert svm_confidence({"a": 6.3, "b": 5.2, "c": 1.0}, "a") == pytest.approx((6.3 - 5.2) / 6.3)
    assert svm_confidence({"a": 0.5, "b": 0.1}, "a") == pytest.approx(0.4)
    assert svm_confidence({"a": 1.0, "b": 3.0}, "a") == 0.0
    assert knn_confidence({"a": 50.0, "b": 100.0, "c": 70.0}) == pytest.approx(1 - 50 / 70)
    assert knn_confidence({"a": 0.0, "b": 0.0}) == 0.0


def test_catalog_defaults_validate() -> None:
    """Defaults pass; ranges, order, locked values and the HOG block fit are checked."""
    defaults = InspectionParams.defaults().to_flat_dict()
    assert validate_params(defaults, defaults) == []
    keys = {e["key"] for e in catalog_json()["entries"]}
    assert set(defaults) <= keys

    def codes(changes: dict) -> list[str]:
        """Error codes for one change."""
        return [e.code for e in validate_params({**defaults, **changes}, defaults)]

    assert codes({"localization.close_size": 0}) == ["param.below_min"]
    assert codes({"localization.open_size": 3.5}) == ["param.invalid_type"]
    assert codes({"localization.column_fraction": 1.2}) == ["param.above_max"]
    assert codes({"localization.row_fraction": 0.0}) == ["param.below_min"]
    assert codes({"localization.profile_smoothing": 500}) == ["param.above_max"]
    assert codes({"rois.cap": [0.0, 1.2]}) == ["param.above_max"]
    assert codes({"rois.cap": [0.3, 0.2]}) == ["param.range_order"]
    assert codes({"normalization.reference_patch": [0.9, 0.1, 0.8, 0.2]}) == ["param.rect_order"]
    assert codes({"hog.cell_cap": [4, 4]}) == ["param.locked"]
    assert codes({"hog.cell_bottle": [8, 8]}) == ["param.locked"]
    assert codes({"lbp.points": 16}) == ["param.locked"]
    assert codes({"lbp.cell_bottle": [8, 8]}) == ["param.locked"]
    assert codes({"bottle.crop_size": [64, 128]}) == ["param.locked"]
    assert codes({"hog.cell_auto": [8, 8]}) == ["param.locked"]
    assert codes({"lbp.cell_auto": [8, 8]}) == ["param.locked"]
    assert codes({"auto_rois.bands": [[0.0, 0.5], [0.5, 1.0]]}) == ["param.locked"]
    assert codes({"localization.diff_threshold": 40}) == ["param.unknown"]
    assert codes({"bottle.width": 0.15}) == ["param.unknown"]
    assert codes({"adhoc.cap_blue_hsv_min": [0, 0, 0]}) == ["param.unknown"]
    assert codes({"bogus.key": 1}) == ["param.unknown"]
    hog_fit = {**defaults, "hog.cell_cap": [64, 64]}
    assert "param.hog_block_fit" in [e.code for e in validate_params(hog_fit)]
    bottle_fit = {**defaults, "hog.cell_bottle": [128, 128]}
    assert "param.hog_block_fit" in [e.code for e in validate_params(bottle_fit)]


def test_catalog_learned_bands() -> None:
    """`auto_rois.bands` is a locked list of ranges of any length: it must match the model,
    each range must be inside 0..1 and ordered, and the cell must fit the learned band."""
    trained = InspectionParams.defaults().with_auto_bands(AUTO_BANDS).to_flat_dict()
    assert validate_params(trained, trained) == []
    catalog = {e["key"]: e for e in catalog_json()["entries"]}
    assert catalog["auto_rois.bands"]["type"] == "ranges"
    assert catalog["auto_rois.bands"]["mode"] == "locked"
    assert catalog["auto_rois.bands"]["default"] == []
    assert catalog["hog.cell_auto"]["mode"] == catalog["lbp.cell_auto"]["mode"] == "locked"
    assert not any(key.startswith(("hog.cell_auto_", "lbp.cell_auto_")) for key in catalog)

    def codes(changes: dict, training: dict | None = trained) -> list[str]:
        """Error codes of the learned params with some changes."""
        return [e.code for e in validate_params({**trained, **changes}, training)]

    assert codes({"auto_rois.bands": [[0.0, 0.5], [0.5, 1.0]]}) == ["param.locked"]
    assert codes({"auto_rois.bands": [[0.0, 0.5], [0.7, 0.6]]}, None) == ["param.range_order"]
    assert codes({"auto_rois.bands": [[0.0, 1.5]]}, None) == ["param.above_max"]
    assert codes({"auto_rois.bands": [[0.0, 0.5, 1.0]]}, None) == ["param.invalid_type"]
    assert codes({"auto_rois.bands": 3}, None) == ["param.invalid_type"]
    thin = {"auto_rois.bands": [[0.0, 0.02], [0.02, 0.5], [0.5, 1.0]]}
    assert codes(thin, None) == ["param.hog_block_fit"]


def parity_folder() -> Path | None:
    """CLI output folder for the parity test (BOTTLE_PARITY_RESULTS or code/results)."""
    folder = Path(os.environ.get("BOTTLE_PARITY_RESULTS") or CODE_DIR / "results")
    return folder if scan_folders([(folder.name, folder)]) else None


def parity_locations(vectors: tuple[str, ...]) -> list[ModelLocation]:
    """Models of the CLI output folder that were trained on one of the feature vectors."""
    folder = parity_folder()
    assert folder is not None
    return [
        location
        for location in scan_folders([(folder.name, folder)])
        if location.directory.name in vectors
    ]


@pytest.mark.skipif(
    parity_folder() is None or not DATASET.is_dir(), reason="needs make run + dataset"
)
def test_parity_with_cli_predictions() -> None:
    """MUST: with the training params, the web engine predicts the CLI class on every
    validation image (validation_predictions.csv)."""
    from bottle_inspection.dataset import load_dataset
    from bottle_inspection.preprocessing import read_image

    from ..engine.analysis import ProfileRuntime, analyse, configure_threads
    from ..models.discovery import describe_model, load_model

    configure_threads()  # the same thread configuration as the web app

    vectors = tuple(
        os.environ.get("BOTTLE_PARITY_VECTORS", "rois_hog,auto_rois_hog,bottle_lbp").split(",")
    )
    locations = parity_locations(vectors)
    assert locations, f"no model of {vectors} found"
    samples = {s.image_path.name: s for s in load_dataset(DATASET) if s.split == "val"}
    images = {name: read_image(sample.image_path) for name, sample in samples.items()}
    mismatches: list[str] = []
    for location in locations:
        algorithm = location.algorithm
        info, artifact = describe_model(location)
        loaded = load_model(info, DATASET, artifact)
        runtime = ProfileRuntime(
            profile_id=1,
            name=info.model_ref,
            version=1,
            algorithm=algorithm,
            model_ref=info.model_ref,
            status="ready",
            color="#2457C5",
            params=loaded.params,
            background=loaded.background,
            model=loaded,
            verdict={"reject_classes": [], "min_confidence": None},
            modified=False,
        )
        expected = pd.read_csv(location.directory / algorithm / "validation_predictions.csv")
        assert len(expected) == 300
        for name, predicted in zip(expected["image"], expected["predicted"]):
            result = analyse(images[name], runtime)
            if result.predicted_class != predicted:
                mismatches.append(
                    f"{info.model_ref}/{name}: web={result.predicted_class} cli={predicted}"
                )
    assert mismatches == []


@pytest.mark.skipif(
    parity_folder() is None or not DATASET.is_dir(), reason="needs make run + dataset"
)
def test_nf1_latency_and_nf3_startup(tmp_path: Path) -> None:
    """NF1: p95 <= 80 ms per image per profile at 1280x720, alone and with 3 profiles in
    parallel (3 fps must not saturate the CPU). NF3: start-up <= 5 s with 10 profiles."""
    import time
    from concurrent.futures import ThreadPoolExecutor

    from fastapi.testclient import TestClient

    from bottle_inspection.preprocessing import read_image

    from ..app import create_app
    from ..engine.analysis import ProfileRuntime, analyse, configure_threads
    from ..models.discovery import describe_model, load_model
    from ..settings import AppConfig, AppPaths

    configure_threads()
    locations = parity_locations(("rois_hog",))
    assert len(locations) == 3
    paths = sorted((DATASET / "images" / "val").glob("*.jpg"))[::10][:30]
    images = [read_image(p) for p in paths]
    runtimes = []
    for location in locations:
        algorithm = location.algorithm
        info, artifact = describe_model(location)
        loaded = load_model(info, DATASET, artifact)
        runtimes.append(
            ProfileRuntime(
                1,
                algorithm,
                1,
                algorithm,
                info.model_ref,
                "ready",
                "#000",
                loaded.params,
                loaded.background,
                loaded,
                {"reject_classes": []},
                False,
            )
        )
    for runtime in runtimes:
        analyse(images[0], runtime)
    p95 = lambda values: float(np.percentile(values, 95))  # noqa: E731
    single = {rt.algorithm: p95([analyse(img, rt).total_ms for img in images]) for rt in runtimes}
    parallel: dict[str, list[float]] = {rt.algorithm: [] for rt in runtimes}
    with ThreadPoolExecutor(6) as pool:
        start = time.perf_counter()
        for img in images:
            futures = [(rt, pool.submit(analyse, img, rt)) for rt in runtimes]
            for rt, future in futures:
                parallel[rt.algorithm].append(future.result().total_ms)
        rate = len(images) / (time.perf_counter() - start)
    print(
        "NF1 p95 single",
        single,
        "parallel",
        {k: p95(v) for k, v in parallel.items()},
        "img/s",
        rate,
    )
    assert all(v <= 80 for v in single.values()), single
    assert all(p95(v) <= 80 for v in parallel.values()), parallel
    assert rate >= 3 * 2, "3 profiles at 3 fps would saturate the CPU"

    config = AppConfig(
        paths=AppPaths(root=tmp_path / "webdata"), frontend_dist=tmp_path / "no_dist"
    )
    with TestClient(create_app(config)) as client:
        base = client.get("/api/profiles").json()
        for i in range(10 - len(base)):
            client.post(f"/api/profiles/{base[0]['id']}/duplicate")
        assert len(client.get("/api/profiles").json()) >= 10
    start = time.perf_counter()
    with TestClient(create_app(config)) as client:
        assert client.get("/api/health").status_code == 200
    startup = time.perf_counter() - start
    print("NF3 startup with 10 profiles:", round(startup, 2), "s")
    assert startup <= 5.0
