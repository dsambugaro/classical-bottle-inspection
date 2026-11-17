# Bottle Inspection Web Interface — Specification

| | |
|---|---|
| Status | Draft for implementation (no code written yet). Revision 2: owner decisions applied (§21) |
| Date | 2026-11-09 |
| Audience | The implementing agent / developer |
| Scope | A new local web interface on top of the existing `bottle_inspection` package |
| Language of this document | English (code and identifiers are English; the UI is bilingual) |

This document is self-contained. Read it fully before writing code. Every requirement has an
ID (`REQ-…`) so that tasks, tests and commits can reference it. Words **MUST**, **SHOULD** and
**MAY** follow RFC 2119.

---

## Table of contents

1. Context
2. Goals, non-goals and hard constraints
3. Glossary
4. Architecture
5. Changes allowed in the existing package
6. Profiles (recipes)
7. Parameter catalog
8. Trained models: discovery, compatibility and background calibration
9. Inspection (main screen)
10. Inspection result: data contract
11. Reports
12. Integrations: conditions → actions
13. Data model (SQLite)
14. HTTP / WebSocket / SSE API
15. UI / UX specification
16. Internationalization (en, pt-BR)
17. Non-functional requirements
18. Project layout, Makefile and dependencies
19. Testing and acceptance criteria
20. Implementation phases
21. Decisions
22. Research sources

---

## 1. Context

The repository `code/` contains the Python package `bottle_inspection` (Python 3.13). It
classifies photos of plastic bottles from the Kaggle dataset *Water Bottle Defect-Level
Detection* (`../archive`, YOLO layout, original split 1200 train / 300 val) into **7 classes**:

| id | class | meaning | pass/reject (default) |
|---:|---|---|---|
| 0 | `bottle_no_defect` | good bottle (called `bottle_ron88` in the Kaggle `data.yaml`) | **pass** |
| 1 | `bottle_other_brand` | bottle of another brand | reject |
| 2 | `defect_low_fill` | liquid level too low | reject |
| 3 | `defect_no_cap` | cap missing | reject |
| 4 | `defect_loose_cap` | cap loose / raised | reject |
| 5 | `defect_debris` | debris inside the liquid | reject |
| 6 | `defect_label_damage` | damaged label | reject |

**Rule (MUST, everywhere in the UI, API and reports):** all 7 classes are always listed,
always with these exact identifiers, in this id order. Never group, abbreviate, or omit a
class (including `bottle_no_defect`). Localized descriptions MAY appear next to the
identifier, never instead of it.

Current pipeline (see `bottle_inspection/` modules and `README.md`):

1. **Background model** (`preprocessing.build_background`): per-pixel median of 200 training
   images at 640×360.
2. **Photometric normalization** (`normalize_lighting`): per-channel gain so the wall
   `REFERENCE_PATCH` matches the background.
3. **Localization** (`foreground_mask`, `locate_bottle`): the bottle is found **only by
   background subtraction**, with no hand-picked size or position. |image − background| is
   thresholded with Otsu's method (computed per image), a morphological closing and opening
   clean the mask, and the column and row profiles of the mask give the bottle box
   (`BottleBox`, relative coordinates).
4. **Regions** (`crop_band`, `crop_regions`): vertical bands **inside the bottle box** (0 = top
   of the bottle, 1 = bottom). The ad hoc ROIs `cap`, `label` and `body` have configured
   bands. The automatic regions `auto_1`, `auto_2`, `auto_3` have bands learned from the
   training images by the CLI and saved in the model (`auto_rois.bands`; the web app never
   edits them). The `bottle` region is the whole box.
5. **Features** (`features.extract_group` / `features.extract_features`): three generic
   descriptors (HOG, LBP, HSV colour) on seven regions (`cap`, `label`, `body`, `auto_1`,
   `auto_2`, `auto_3` and the whole `bottle`) give 21 feature groups named
   `<region>_<descriptor>` (split with `rpartition("_")`: `auto_1_hog` has two underscores).
   Sizes with the defaults: `cap_hog` 1260, `cap_lbp` 120, `cap_color` 40, `label_hog` 540,
   `label_lbp` 240, `label_color` 40, `body_hog` 792, `body_lbp` 360, `body_color` 40,
   `bottle_hog` 4140, `bottle_lbp` 1440, `bottle_color` 40; an automatic region is a slice of
   the 96×384 bottle crop, so its sizes follow from its learned band. No feature depends on
   the colour or shape of a specific product.
6. **Classifier** (`models.MODELS`), trained once per **feature vector** of
   `config.FEATURE_VECTORS` (9 vectors: `rois_<d>` = the descriptor over the three ad hoc ROIs,
   e.g. `rois_hog` = cap_hog + label_hog + body_hog; `auto_rois_<d>` = over the three
   automatic regions; `bottle_<d>` = over the whole bottle; `<d>` is `hog`, `lbp` or `color`;
   the groups are never all concatenated): `svm` (StandardScaler + SVC; the kernel is chosen by the grid search), `knn`
   (StandardScaler + KNeighbors), `random_forest`; hyperparameters chosen by `GridSearchCV`
   (5 stratified folds on the training split, `f1_macro`).
7. `didactic.py` already computes the *real* per-class decision values of each model (KNN
   distances to the closest image of each class, RF averaged probabilities, SVM one-vs-one
   votes and one-vs-rest scores). The web app MUST reuse that logic for confidence and
   per-class scores (§10).

Measured reference numbers (validation split, default constants): SVM 100 %, KNN 99.7 %,
RF 99.7 % accuracy; feature extraction ≈ 31–36 ms per image; prediction 0.02–1.3 ms.

The thesis proposal (`../tcc-data/tcc.md`, section 4.1) asks for **alerts when defects
repeat**. The integrations module (§12) generalizes that.

## 2. Goals, non-goals and hard constraints

### Goals

- **REQ-G1** Analyse bottles from uploaded images and from cameras (browser webcam, USB and
  IP/RTSP cameras attached to the server).
- **REQ-G2** Let the user create any number of **profiles**. A profile holds the name, the
  trained model it uses (which fixes the algorithm: `svm`, `knn`, `random_forest`, …), every
  ad hoc parameter and other settings.
- **REQ-G3** On the main screen, the user selects **one or more profiles**. Each image is
  analysed by every selected profile, and the results are shown **separately, side by side,
  for comparison**. There is no combined/consensus verdict.
- **REQ-G8** The web app **never trains models**. It is the *consumer* of the models trained
  and validated by the existing CLI pipeline (`make run` / `make quick` / …). Training,
  grid search and validation stay exclusively in the CLI.
- **REQ-G4** After processing, highlight the ROIs on the image, show the class, a confidence
  measure, the 7-class scores and execution times per step.
- **REQ-G5** A **reports** screen with counts per class, good vs defective, ratios, trends,
  a Pareto chart and per-profile comparison.
- **REQ-G6** An **integrations** screen with condition → action rules: e-mail, HTTP webhook
  and other actions (§12).
- **REQ-G7** Functional, beautiful, UX-focused design.

### Non-goals (v1)

- User accounts, login or authorization of any kind (**REQ-C3**).
- Light/dark themes or a theme switcher (**REQ-C4**). There is exactly one visual theme.
- Multi-user concurrency, cloud deployment, horizontal scaling.
- Deep-learning models (YOLO). The algorithms are the ones in `bottle_inspection.models`
  plus any added later through the same `ModelSpec` registry.
- Editing or re-labelling the Kaggle dataset; the original split is used as is.
- Training, retraining or grid search from the web (REQ-G8), and labelling captured images
  for training (decided: not planned).
- Combining the verdicts of several profiles into one (decided: each profile's result is
  shown and stored separately).

### Hard constraints

- **REQ-C1** The CLI pipeline MUST keep working through the Makefile
  (`make setup|run|quick|recompute|no-normalization|all|format|lint|clean|distclean`,
  variables `DATA`, `OUTPUT`, `MODELS`, `DIDACTIC`). The web interface is a separate entry
  point, not a replacement.
- **REQ-C2** Storage MUST be embedded (SQLite file + files on disk). No database server.
- **REQ-C3** No authentication.
- **REQ-C4** A single fixed theme; no light/dark mode.
- **REQ-C5** UI in **English (default)** and **Brazilian Portuguese**, switchable at runtime.
- **REQ-C6** All code, identifiers, comments, docstrings, log messages and API fields in
  English. Type hints and docstrings on every Python function; black (line length 100) for
  Python; the frontend uses the formatter/linter of §18.
- **REQ-C7** The app binds to `127.0.0.1` by default (no auth ⇒ do not expose it on the
  network unless the user explicitly passes another host).
- **REQ-C8** The Kaggle dataset is used with its original train/val split, never reorganized.

## 3. Glossary

| Term | Meaning |
|---|---|
| Profile | A named, versioned set of all settings needed to analyse a bottle (the industrial "recipe"/"job"): a trained model + ad hoc parameters + verdict settings. |
| Parameter | One ad hoc value of a profile (e.g. `rois.cap`, `localization.close_size`). |
| Feature vector | The feature groups, concatenated in order, a model is trained on (`config.FEATURE_VECTORS`, e.g. `rois_hog`). A model is identified by its feature vector plus its algorithm, e.g. "rois_hog · SVM". |
| Trained model | A `model.joblib` written by the CLI under `<OUTPUT>/<feature vector>/<algorithm>/` (pipeline + background + classes + the parameters used in training), plus the validation results of that run. The web app only reads it. |
| Training parameters | The parameter values the trained model was trained with. Profiles start from them. |
| Analysis | One image processed by one profile. |
| Inspection | One image (one bottle) processed by one or more profiles; groups its analyses (each keeps its own verdict). |
| Session | A continuous run (camera stream or batch upload) grouping inspections. |
| Verdict | `pass` or `reject`, derived from the predicted class (and optional confidence rule). |
| Rule | A condition → action automation. |
| Target | A configured destination of actions (SMTP server, webhook URL, MQTT broker, …). |

## 4. Architecture

```
┌──────────────────────────── browser (SPA) ─────────────────────────────┐
│ React + TypeScript + Vite, shadcn/ui (Radix) + Tailwind, ECharts,      │
│ i18next. Screens: Inspect · Profiles · Reports · Integrations · System  │
└───────────────▲───────────────────────▲────────────────────▲───────────┘
          REST (JSON)           SSE (events, job progress)    WebSocket (camera frames)
┌───────────────┴───────────────────────┴────────────────────┴───────────┐
│ FastAPI app  `bottle_inspection_web` (single uvicorn worker, 127.0.0.1) │
│  api/      routers: profiles, inspections, cameras, reports, rules, …   │
│  engine/   InspectionEngine: loads trained models, runs analysis        │
│  models/   discovery of models trained by the CLI (read-only)           │
│  cameras/  server-side capture threads (OpenCV VideoCapture)            │
│  rules/    rule evaluation + action workers (asyncio.Queue)             │
│  db/       SQLAlchemy 2.x models, Alembic migrations, SQLite (WAL)      │
└───────────────┬────────────────────────────────────────────────────────┘
                │ imports the same functions the CLI uses, passing profile params
┌───────────────┴────────────────────────────────────────────────────────┐
│ package `bottle_inspection` (CLI + Makefile)                            │
│   make run → results/<vector>/<algorithm>/model.joblib ─read by▶ web app │
└────────────────────────────────────────────────────────────────────────┘
Storage: code/webdata/ → app.db (SQLite), images/, backgrounds/, exports/, logs/, secrets.toml
Trained models: read in place from the CLI output folders (default results/; the `model.joblib` files are in `<folder>/<feature vector>/<algorithm>/`)
```

Decisions (with rationale; details and sources in §18 and §22):

- **Backend: FastAPI + uvicorn.** It has built-in WebSockets, native SSE
  (`fastapi.sse.EventSourceResponse`, FastAPI ≥ 0.135), uploads and static SPA serving, and
  it is Python, so it imports `bottle_inspection` directly.
- **Frontend: React + Vite + TypeScript + shadcn/ui + Tailwind + ECharts + i18next.** It
  handles image overlays with zoom/pan, a live camera, parameter-heavy forms and dashboards
  better than server-rendered HTMX, and builds to a static `dist/` that FastAPI serves. Node
  is needed only to build.
- **Single process, single worker.** Camera threads, the WebSocket registry and in-memory
  model caches live in one process. Inference runs in a thread pool. There is no training
  in the web app.
- **One code path for the pipeline.** The web app calls the same preprocessing and feature
  functions as the CLI. The CLI uses the constants in `config.py`; the web app passes the
  profile's values (§5). The pipeline logic is not duplicated.
- **SQLite + files.** Images are stored as files; the DB keeps paths. Profile parameters are
  stored as a validated JSON column.

## 5. Changes allowed in the existing package

The ad hoc values are **module-level constants** in `bottle_inspection/config.py` (plus
two in other modules: `features.HOG_BLOCK`, `features.HSV_CHANNEL_RANGES`). The functions read them directly, so a profile cannot change
them per call. Monkey-patching module globals per request is **forbidden** (not thread-safe,
leaks between profiles).

**REQ-P1** Add a frozen dataclass `InspectionParams` (new module
`bottle_inspection/params.py`) whose fields mirror the catalog of §7. Its
`InspectionParams.defaults()` builds the values **from the existing constants**, so `config.py`
stays the single source of default values.

**REQ-P2 (owner decision).** The CLI/`make` uses the fixed constants; the web uses the
profile's values; **no duplicated pipeline code**. Implementation: the functions that read
those constants gain an **optional keyword argument**
`params: InspectionParams | None = None`. When it is `None`, they use
`InspectionParams.defaults()`, i.e. exactly the `config.py` constants. This affects the
preprocessing and feature functions (`preprocessing.*`, `features.*`) and
`extraction.extract_image`. The CLI, the pipeline and the Makefile do not pass `params` and
therefore use the `config.py` constants. The web app MUST NOT re-implement any of these functions; it calls
them with the profile's `InspectionParams`.

**REQ-P2b Training parameters saved with each model.** `pipeline.save_model_outputs`
writes to `<OUTPUT>/<feature vector>/<algorithm>/model.joblib` a dict with `model`,
`background`, `classes` and these keys:
- `format_version`: `params.ARTIFACT_FORMAT_VERSION` (currently 1);
- `params`: `InspectionParams` as a plain dict, with normalization reflecting the
  `--no-normalization` flag;
- `best_params`;
- `feature_length`;
- `feature_groups` (`{group: size}` of the model's own vector, in order; they are also saved
  as `params.features.groups`);
- `feature_vector` (name of the vector, e.g. `rois_hog`);
- `library_versions` (python, numpy, scikit-learn, scikit-image, opencv);
- `trained_at`.

**REQ-P3** No other change to existing modules. Everything web-related lives in the new
package `bottle_inspection_web/` (§18). Reading the trained-model metadata and building
`InspectionParams` from a dict are helpers in `bottle_inspection/params.py`, so both sides
share them.

## 6. Profiles (recipes)

A profile is everything needed to turn an image into a verdict with an already trained
model.

**REQ-PR1 Fields**

| Field | Type | Notes |
|---|---|---|
| `id` | int | |
| `name` | str, unique, 1–60 chars | |
| `description` | str, optional | |
| `color` | one of 8 preset accent colours | Identifies the profile in charts and result cards. |
| `model_ref` | str | Identifier of a trained model found by discovery (§8.1), e.g. `results/rois_hog/svm`. Fixes the feature vector, the algorithm and the locked parameters. |
| `algorithm` | `svm` \| `knn` \| `random_forest` (read-only) | Copied from the trained model, for display and filtering. |
| `params` | JSON | All ad hoc parameters (§7), validated by Pydantic against the catalog and against the model's locked parameters. |
| `background_source` | `model` \| `captured` \| `uploaded` | §8.3. `model` = the background saved with the trained model (default). |
| `verdict` | JSON | `reject_classes` (default: all except `bottle_no_defect`), `min_confidence` (default `null` = off), `low_confidence_verdict` (`reject` \| `review`, default `review`). |
| `version` | int | Incremented on every saved change to `model_ref`, `params`, `background_source` or `verdict`. |
| `status` | `ready` \| `modified` \| `model_missing` \| `incompatible` | Derived (§8.2), not stored. |
| `is_builtin` | bool | Seed profiles (§6.2) cannot be deleted, but can be duplicated. |
| `created_at`, `updated_at` | datetime | |

**REQ-PR2 Seed profiles.** On start, model discovery (§8.1) creates one built-in profile per
trained model of **any** of the 9 feature vectors that has no profile yet (every vector is a
complete view of the bottle). It is named after the algorithm, vector and folder, e.g.
`SVM rois_hog (results)`, `KNN auto_rois_lbp (results)` or `RF bottle_color (results)`,
and its description says which scope it is (ad hoc ROIs, automatic regions or whole bottle);
its `params` are exactly the model's training parameters.
A user-created profile whose model is gone stays listed as `model_missing` (it cannot
analyse) and can be moved to another model, which takes the parameters from that model. When
no trained model exists, the Profiles and Inspect screens show an empty state:
"No trained models found. Run `make run` in `code/` to train and validate them, then
click Refresh."

**REQ-PR3 Operations.** Create (choose a trained model; the parameters start from its
training parameters), duplicate, rename, edit, delete (soft delete: the profile is hidden but
its analyses stay in reports), export as JSON (parameters and `model_ref`, never the model
file), import JSON (schema-validated, with a preview diff before applying; `model_ref` must
exist), and reset to the training value per field, per group and for the whole profile.

**REQ-PR4 Versioning.** Each analysis stores `profile_id` and `profile_version`. Reports can
filter by version. The History tab lists versions with a diff and "restore this version".

## 7. Parameter catalog

**REQ-PC1** The catalog is defined **once** in the backend (`bottle_inspection_web/params_catalog.py`,
built on `bottle_inspection.params.InspectionParams`) and served at
`GET /api/params/catalog`. The frontend renders the profile editor **from the catalog**: no
hard-coded parameter forms. Each entry has:

- `key` (dot path) and `group`;
- `type`: `int`, `float`, `bool`, `range` = (min,max) pair, `rect` = (x0,y0,x1,y1),
  `size` = (w,h), `int3` = three integers, `enum` or `multiselect`;
- `default`, the value from `InspectionParams.defaults()`;
- `min`, `max`, `step`, `unit`;
- `mode`: `editable` or `locked`;
- `affects_preview` (bool): whether the live preview can show its effect;
- `label_key` and `help_key` (i18n keys);
- `config_constant`, the name of the constant in `config.py`, shown in the help tooltip;
- `constraints`: cross-field rules, e.g. `y0 < y1`.

**REQ-PC2 Editable vs locked.** The web app does not train (REQ-G8), so a profile can only
change parameters that keep the trained model usable:

- **`editable`**: changes *values* of the features but not the **length or meaning of the
  feature vector** (geometry, thresholds, colour ranges, normalization). The profile works
  with the same trained model. When a value differs from the model's training value, the
  editor shows a "differs from training" marker with the training value. Result cards show
  a "modified parameters" badge, because accuracy was validated only with the training
  values.
- **`locked`**: defines the shape of the feature vector or the background (ROI resize
  sizes, whole-bottle size, HOG and LBP settings, histogram bins, feature groups, localization
  size). It is read-only in the web, always equal to the trained model's value, and shown
  with a lock icon and "defined by training — change it in `config.py` and run `make run`".

Coordinates are **relative** (0–1) to the image size, as in `config.py`. The editor shows
them as percentages (0–100 %) but stores them as fractions.

### 7.1 Localization

| key | type | default | min–max (step) | mode | Description |
|---|---|---|---|:---:|---|
| `localization.size` | size | (640, 360) | — | locked | Resolution used to locate the bottle; must match the stored background. |
| `localization.close_size` | int | 15 | 1–101 (2) | editable | Diameter (px at the localization size) of the morphological closing of the foreground mask (joins the holes of the transparent bottle). |
| `localization.open_size` | int | 5 | 1–51 (2) | editable | Diameter of the opening applied after the closing (removes thin lines and specks). |
| `localization.profile_smoothing` | int | 31 | 1–101 (2) | editable | Width of the moving average applied to the column profile of the mask. |
| `localization.column_fraction` | float | 0.4 | 0.05–0.95 (0.01) | editable | The bottle spans the columns around the peak of the column profile above this fraction of the peak. |
| `localization.bottle_width` | float | 0.145 | 0.02–1.0 (0.005) | editable | Width of the bottle box, as a fraction of the image width, centred on the bottle found in the column profile. |
| `localization.top_fraction` | float | 0.3 | 0.05–0.95 (0.01) | editable | Inside those columns, the top of the bottle is the first row that reaches this fraction of the densest row. |
| `localization.row_fraction` | float | 0.5 | 0.05–0.95 (0.01) | editable | Inside those columns, the bottom of the bottle is the last row that reaches this fraction of the densest row. |

There is no foreground threshold: the difference to the background is thresholded with Otsu's
method for each image, and the box is measured on the mask (no bottle size or position is
configured).

### 7.2 Photometric normalization

| key | type | default | min–max | mode | Description |
|---|---|---|---|:---:|---|
| `normalization.enabled` | bool | training value | — | editable | Rescale each channel so the reference patch matches the background. |
| `normalization.reference_patch` | rect | (0.80, 0.02, 0.97, 0.25) | 0–1 | editable | Wall region never covered by a bottle. Editable by drawing on the preview. |

### 7.3 Bottle crop

| key | type | default | min–max | mode | Description |
|---|---|---|---|:---:|---|
| `bottle.crop_size` | size | (96, 384) | — | locked | Canonical size the detected bottle box is resized to before HOG and LBP; the automatic regions are slices of it. |

The box itself is detected, not configured (§1, step 3).

### 7.4 Regions of interest

| key | type | default | mode | Description |
|---|---|---|:---:|---|
| `rois.cap` | range | (0.00, 0.12) | editable | Vertical band of the cap ROI inside the bottle box. |
| `rois.label` | range | (0.20, 0.52) | editable | Vertical band of the label ROI inside the bottle box. |
| `rois.body` | range | (0.12, 1.00) | editable | Vertical band of the body ROI inside the bottle box. |
| `rois.cap_size` / `rois.label_size` / `rois.body_size` | size | (64,48) / (64,96) / (48,192) | locked | Canonical sizes the ROIs are resized to before HOG. |

ROI bands are relative to the detected bottle box: 0 is the top of the bottle and 1 its
bottom, so each band must be inside 0–1 and start before it ends. They are editable by
dragging their edges on the preview image (§15.5; the dragged height is converted to the box),
as well as through the numeric inputs.

### 7.4b Automatic regions (locked)

| key | type | default | mode | Description |
|---|---|---|:---:|---|
| `auto_rois.bands` | list of ranges | `[]` (a trained model has 3) | locked | Bands (inside the bottle box) of the regions `auto_1`, `auto_2`, `auto_3`, top to bottom, **learned from the training images by the CLI**. Read-only, shown as learned at training time, like the background. |

The list has no fixed length, so it is its own catalog type (`ranges`). The check of the
HOG block (each region must hold at least one 2×2 block of cells) also covers the automatic
regions, whose size follows from their learned bands.

### 7.5 HOG (all locked)

| key | type | default | mode |
|---|---|---|:---:|
| `hog.orientations` | int | 9 | locked |
| `hog.cell_cap` / `hog.cell_label` / `hog.cell_body` / `hog.cell_auto` / `hog.cell_bottle` | size | (8,8) / (16,16) / (16,16) / (16,16) / (16,16) | locked |
| `hog.block` | size | (2, 2) | locked |

### 7.6 LBP and colour (all locked)

| key | type | default | mode | Description |
|---|---|---|:---:|---|
| `lbp.points` | int | 8 | locked | Neighbours compared with each pixel by the LBP operator. |
| `lbp.radius` | int | 1 | locked | Radius (px) of the circle of neighbours. |
| `lbp.cell_cap` / `lbp.cell_label` / `lbp.cell_body` / `lbp.cell_auto` / `lbp.cell_bottle` | size | (16,16) each | locked | Cell size (px) of the LBP histograms of each region (`cell_auto` is shared by the three automatic regions). |
| `color.hsv_bins` | (int,int,int) | (18, 8, 8) | locked | Histogram bins for H, S and V. |

The preview has one mask overlay: the foreground mask.

### 7.7 Feature groups and model (read-only)

| key | mode | Description |
|---|:---:|---|
| `features.groups` | locked | The feature groups of the model's own vector, in order (e.g. `cap_hog`, `label_hog`, `body_hog` for `rois_hog`; 2592 values with the defaults). They are a subset of the 21 `FEATURE_GROUPS`. |
| model algorithm, best hyperparameters, CV score, validation metrics | read-only | Shown from the trained model's metadata and `summary.json` / `classification_report.txt`. |

### 7.8 Verdict

| key | type | default | Description |
|---|---|---|---|
| `verdict.reject_classes` | multiselect of the 7 classes | all except `bottle_no_defect` | Classes that produce `reject`. |
| `verdict.min_confidence` | float 0–1 or null | null | When set and the confidence is below it, the verdict becomes `low_confidence_verdict`. |
| `verdict.low_confidence_verdict` | `reject` \| `review` | `review` | `review` is a third verdict shown in amber. |

## 8. Trained models: discovery, compatibility and background calibration

### 8.1 Discovery (read-only)

- **REQ-M1** The web app finds trained models in the **model folders** set in System
  settings (default: `results`, relative to `code/`).

  - **Shape of a trained model:** `<folder>/<feature vector>/<algorithm>/model.joblib`. Its
    `model_ref` is `<folder>/<feature vector>/<algorithm>` (e.g. `results/rois_hog/svm`).
    A folder that directly contains `<algorithm>/model.joblib` (for example a model folder
    set to `results/rois_hog`) is also accepted; its `model_ref` is then
    `<folder>/<algorithm>`.
  - **Data read from the trained model:**
    - from `model.joblib`: the pipeline, the background, the classes and the training
      params (REQ-P2b), best params, feature length and library versions;
    - from the `<feature vector>/` folder (it looks like a results folder of its own):
      `summary.json`, `model_comparison.csv`,
      `<algorithm>/classification_report.txt` and `<algorithm>/confusion_matrix.png`.

  The model list and the model selector show the feature vector next to the algorithm
  (`feature_vector` in the model JSON, e.g. "rois_hog · svm").

  These validation results are shown read-only in the profile's Model tab: accuracy, macro
  F1, per-class metrics for all 7 classes, the confusion matrix and timing.
- **REQ-M2 Models that fail to load** are listed with the status `error` and the load error
  (for example, a `model.joblib` without `feature_vector` or `params`, or with a
  `format_version` other than `params.ARTIFACT_FORMAT_VERSION`), a warning is logged, and no
  profile is created for them. Discovery does not fail because of them. A `trained_models` row
  with a load error is described again at the next discovery.
- **REQ-M3** Discovery runs at start-up and on "Refresh models" (System and Profiles). A model
  is identified by its `model_ref` plus the `model.joblib` modification time. When the
  file changes, because `make run` was executed again, the app reloads it. Profiles that use
  it keep working and show "model updated on <date>".
- **REQ-M4** Only files inside the configured model folders are loaded. Joblib/pickle is
  unsafe for untrusted files, so there is no model upload. On a library-version mismatch
  (`InconsistentVersionWarning`), the model is still listed, with a warning that it should
  be retrained with `make run`.

### 8.2 Profile status (derived)

| status | when | can analyse? |
|---|---|:---:|
| `ready` | model found and all editable parameters equal to its training values | yes |
| `modified` | model found and at least one editable parameter differs from training | yes (badge "modified parameters — not validated") |
| `model_missing` | `model_ref` not found (folder deleted or renamed) | no |
| `incompatible` | model found but its locked parameters or feature length differ from what the profile expects (e.g. `config.py` changed and `make run` re-trained with other HOG settings) | no; offer "Re-link to this model's parameters" (copies the new locked values, keeps the editable ones) |

### 8.3 Background calibration

The models were trained on the Kaggle camera. A different camera or scene needs its own
background. Training data for other cameras is out of scope; the UI says the validated
accuracy applies only to the training setup.

- **REQ-M5** `background_source`:
  - `model` (default): the background saved with the trained model.
  - `captured`: in the profile editor, "Capture background" takes N frames (default 30,
    configurable) from a chosen camera with the scene **empty**, and stores their per-pixel
    median at the locked localization size under `webdata/backgrounds/`.
  - `uploaded`: an image file supplied by the user.

  A non-`model` background makes the profile `modified`. Calibration is not training: no
  classifier changes.
- **REQ-M6** The system warns, without blocking, when the input image resolution or aspect
  ratio differs from the background's, or when the localization foreground fraction is
  outside the range seen on the dataset (§10, `warnings`).

## 9. Inspection (main screen)

### 9.1 Sources

- **REQ-I1 Upload.** Drag-and-drop or a file picker. Multiple files are accepted (JPEG, PNG,
  BMP, WebP), up to 25 MB each and 500 per batch (both configurable). The type is validated
  by decoding with OpenCV. A batch becomes a session and is processed sequentially with a
  progress bar. Files are copied to `webdata/images/<yyyy>/<mm>/<dd>/<uuid>.<ext>`.
- **REQ-I2 Dataset samples.** "Try with dataset images" picks random images from the
  validation split (or lets the user browse it). These carry a **ground-truth label** taken
  from the YOLO file, so the result shows correct or incorrect per profile.
- **REQ-I3 Browser camera.** Uses `getUserMedia` (works on `localhost`/HTTPS only; handle
  `NotAllowedError`, `NotFoundError`, `NotReadableError` and a permission prompt that is
  never answered). There is a device picker. Frames are drawn to a canvas and sent as JPEG
  (quality 0.85) binary messages over the WebSocket `/ws/camera/browser`, with **one frame in
  flight**: the next frame is sent only after the result of the previous one arrives. The
  target rate is configurable (1–15 fps, default 3).
- **REQ-I4 Server cameras.** Configured in System → Cameras: name, type (`usb` with device
  index, `rtsp`/`http` with URL), resolution, fps cap and a test button that grabs a
  snapshot. Each active camera has one capture thread that keeps only the latest frame. A
  live preview is streamed as MJPEG (`multipart/x-mixed-replace`) at a low rate. Analysis
  pulls the latest frame at the configured rate. Note for WSL2: USB cameras need extra setup
  (usbipd). The UI MUST show a clear error when a device cannot be opened.
- **REQ-I5 Acquisition modes** (cameras): `continuous` (analyse at N fps), `single`
  (button or the Space key captures and analyses one frame) and `motion` (analyse when the
  fraction of the foreground mask (Otsu threshold, closing and opening, whole image) rises
  above a threshold after having been low, i.e. a bottle arrived; default threshold 0.04, the
  same fraction below which `low_foreground` is raised). `motion` uses the first selected
  profile's background and parameters.
- **REQ-I6** Images from cameras are saved according to a storage policy (System
  settings): `all`, `rejects_only` (default) or `none`. With `none`, the analysis record has
  no image, and reports still count it.

### 9.2 Profile selection

- **REQ-I7** A multi-select of profiles. Only `ready` or `modified` profiles are selectable;
  `model_missing` and `incompatible` profiles are shown disabled with the reason. The
  selection is remembered (local storage) and ordered (drag to reorder). At most 6 profiles
  at once (configurable).
- **REQ-I8 (owner decision)** With several profiles selected, each profile's result is shown
  **separately**, in its own card, side by side, for comparison: its own verdict, class,
  confidence, scores, overlays and timings. The app does **not** compute a combined or
  consensus verdict, and rules evaluate each profile's analysis on its own.

### 9.3 Processing

- **REQ-I9** One image is decoded once. Each profile runs on it in the thread pool. Results
  are pushed to the client as soon as each profile finishes, and the inspection completes
  when all have. The step timings of §10 are measured with `time.perf_counter()`.
- **REQ-I10** After each analysis: persist it (§13), evaluate the rules (§12), and publish
  an SSE `analysis.completed` event.

## 10. Inspection result: data contract

`AnalysisResult` (JSON; used by the REST API, SSE and WebSocket):

```json
{
  "analysis_id": 1234,
  "inspection_id": 987,
  "profile": {"id": 2, "name": "Default SVM", "version": 3, "algorithm": "svm",
              "model_ref": "results/rois_hog/svm", "status": "ready"},
  "image": {"id": 555, "width": 1280, "height": 720, "url": "/api/images/555"},
  "predicted_class": "defect_loose_cap",
  "verdict": "reject",
  "confidence": 0.83,
  "scores": {
    "bottle_no_defect": 4.24, "bottle_other_brand": -0.26, "defect_low_fill": 2.93,
    "defect_no_cap": 5.19, "defect_loose_cap": 6.29, "defect_debris": 0.74,
    "defect_label_damage": 1.77
  },
  "score_kind": "svm_ovr_score",
  "details": {"svm_votes": {"bottle_no_defect": 4, "...": 0}},
  "geometry": {
    "bottle_center_x": 0.512, "bottle_top_y": 0.091,
    "bottle_box": [0.422, 0.091, 0.602, 0.90],
    "rois": {"cap": [0.422, 0.091, 0.602, 0.19], "label": [0.422, 0.27, 0.602, 0.56],
             "body": [0.422, 0.19, 0.602, 0.90]},
    "reference_patch": [0.80, 0.02, 0.97, 0.25],
    "foreground_fraction": 0.09
  },
  "timings_ms": {
    "decode": 4.1, "normalize": 3.0, "foreground_mask": 1.2, "localize": 0.4,
    "crop_regions": 0.2,
    "features": {"cap_hog": 2.0, "label_hog": 1.4, "body_hog": 1.9},
    "predict": 1.3, "total": 33.8
  },
  "ground_truth": null,
  "warnings": [],
  "created_at": "2026-11-09T14:03:22.120Z"
}
```

Rules:

- **REQ-R1** `scores` always has **all 7 classes** in id order. Each algorithm uses real
  values from the trained model, reusing the logic of `didactic.py`:
  - `svm`: one-vs-rest score (votes plus confidence). `details.svm_votes` holds the 21
    pairwise votes counted per class.
  - `knn`: the distance to the closest training image of each class
    (`score_kind = "knn_class_distance"`, lower is better). `details` holds the nearest
    neighbours (class, distance and, when the dataset path is configured, the training image
    file name for a thumbnail). Row *i* of the fitted KNN corresponds to the *i*-th training
    sample of `dataset.load_dataset(dataset_path)` filtered to the `train` split; that order is
    what the CLI used.
  - `random_forest`: the averaged tree probability per class.
- **REQ-R2** `confidence` ∈ [0, 1] is defined per algorithm. Document it in the UI tooltip and
  say plainly that confidences are **not comparable across algorithms**.
  - `random_forest`: the maximum probability.
  - `knn`: `1 − d_win / d_runner_up`, where `d_win` is the smallest class distance and
    `d_runner_up` the second smallest.
  - `svm`: `(s_win − s_runner_up) / max(1, |s_win|)`, clipped to [0, 1].
- **REQ-R3** All coordinates are relative (0–1) to the image. The client draws the overlays;
  the server never burns them into the stored image (the raw image is kept separate).
- **REQ-R4** The timings mirror the real pipeline steps. `total` is wall time for the
  analysis, excluding queueing. The UI shows `total` prominently and the breakdown as a
  stacked horizontal bar. Only the feature groups of the model's own vector are computed and
  timed (`features` holds one entry per group of that vector, e.g. `cap_hog`, `label_hog`,
  `body_hog` for `rois_hog`, `auto_1_color`, `auto_2_color`, `auto_3_color` for `auto_rois_color`).
  The steps follow `preprocessing.prepare_image`: `normalize` → `foreground_mask` → `localize`
  (bottle box) → `crop_regions` (every available region). `geometry.bottle_box` is the
  detected box (`bottle_center_x` and `bottle_top_y` are its centre and top) and
  `geometry.rois` has one rectangle per **region of the model's feature groups** (the ad hoc
  ROIs, or the automatic regions from the model's learned bands; none for `bottle_*` models,
  whose only region is the box): the band of the params applied to the box. The frontend
  draws exactly those regions.
- **REQ-R5** `warnings` codes: `low_foreground` (bottle possibly absent),
  `resolution_mismatch`, `modified_parameters`, `low_confidence`. `low_foreground` uses the
  fraction of the **whole** foreground mask that is bottle (below 0.04), not the area of the
  detected box: the box is measured on the mask, so a speck would fill its own box.

## 11. Reports

- **REQ-RP1 Filters** (top bar, kept in the URL query string so views can be shared or
  bookmarked):
  - date range (presets: last hour, today, 7 days, 30 days, custom);
  - profiles (multi);
  - profile version;
  - source (upload, dataset sample, browser camera, server camera by name);
  - session;
  - verdict;
  - class (multi; all 7 listed).
- **REQ-RP2 KPI tiles:**
  - bottles inspected;
  - pass count and rate;
  - reject count and rate;
  - review count;
  - first-pass yield (pass ÷ inspected);
  - mean and p95 analysis time;
  - throughput (bottles per minute in the range);
  - rules fired.

  Each tile shows the delta against the previous equivalent period.
- **REQ-RP3 Charts** (ECharts; class colours from §15.2):
  1. Class distribution: a horizontal bar of the 7 classes, absolute and %.
  2. Defect Pareto: the reject classes sorted by count, with a cumulative-% line.
  3. Good vs defective over time: a stacked area; bucket size adapts to the range (minute,
     hour, day).
  4. Rejects by class over time: stacked bars.
  5. Per-profile comparison: a table of profiles × the 7 classes (counts), plus verdict
     agreement between profiles for the same inspections (pairwise agreement %).
  6. Confidence distribution per profile (histogram), with the `min_confidence` line when set.
  7. Timing per step: the mean of each step, per profile.
  8. When ground truth exists (dataset samples): accuracy, macro F1, and the 7×7 confusion
     matrix per profile.
- **REQ-RP4 Table of analyses:** paginated and sortable, with a thumbnail. Clicking a row
  opens the result viewer (the same component as the main screen) in a side panel.
- **REQ-RP5 Export:** CSV of the filtered analyses (one row per analysis, the 7 score
  columns always present) and a PNG of each chart. Number and date formats follow the
  selected language.
- **REQ-RP6** All aggregation is done in SQL with indexes (§13). The reports screen loads in
  under 1 s with 100k analyses.

## 12. Integrations: conditions → actions

The model follows Home Assistant (trigger, condition, action) and Grafana (contact points,
grouping, repeat interval). Webhooks follow the Standard Webhooks specification.

### 12.1 Targets (connections)

- **REQ-IN1** A target is reusable connection settings, configured once and referenced by
  actions. Each target has a "Test" button that sends a test message and stores the result.

| type (v1) | fields |
|---|---|
| `smtp` | host, port, security (`starttls` (587) \| `ssl` (465) \| `none`), username, password (secret), from address, timeout_s (15) |
| `webhook` | base URL, default method, default headers, signing secret (secret, optional; `whsec_…`), timeout_s (10), allow private IPs (bool, default true because the app is local) |
| `mqtt` | host, port (1883/8883), TLS (bool + CA file), username, password (secret), client_id, default QoS (1), retain (false) |
| `file` | directory, format (`csv` \| `jsonl`), rotation (`daily` \| `size_mb`) |

**v2, first priority (owner decision): industrial signals** for a reject actuator / PLC:

| type | fields |
|---|---|
| `modbus_tcp` | host, port (502), device/unit id, function (`coil` \| `register`), address, value, pulse_ms (optional: write value, wait, write reset value), timeout_s (3) |
| `opcua` | endpoint URL, node id, data type, value, security policy, username/password (secret, optional) |
| `gpio` | chip/line (Linux gpiod) or serial relay board, pulse_ms |

Actuator writes are **never retried automatically** (a late reject pulse could hit the wrong
bottle). Failures are logged and notified.

v3 (later): messaging (`slack`, `discord`, `telegram`, `teams` via Teams Workflows) and
`script`.

- **REQ-IN2 Secrets** (passwords, tokens, signing secrets) are never stored in `app.db` and
  never returned by the API. They live in `webdata/secrets.toml` (created with mode 0600).
  The DB references them by name (`secret_ref`), and the API shows them as `••••` with a
  "replace" action. Environment variables `BOTTLE_WEB_SECRET_<NAME>` override the file.

### 12.2 Rules

- **REQ-IN3 Rule fields:** name, description, enabled, severity (`info` \| `warning` \|
  `critical`), scope (profiles: all or a list), trigger, conditions (list; `all` or `any`),
  actions (ordered list), cooldown_s (default 60), dedupe key (`rule`, `rule+class` or
  `rule+profile`; default `rule+class`), run_mode (`single` (default) \| `queued` \|
  `parallel`).
- **REQ-IN4 Trigger** (v1): `analysis_completed`, fired once per analysis, i.e. per image ×
  profile. Each profile's result is evaluated independently.
- **REQ-IN5 Condition types:**

| type | params | v |
|---|---|---|
| `class_is` | classes (multi, from the 7) | v1 |
| `verdict_is` | `pass` \| `reject` \| `review` | v1 |
| `any_defect` | — (verdict = reject) | v1 |
| `repeated_class` | class (or "any reject class"), n, last_m bottles (default n=4, m=10, the CLI alert) | v1 |
| `consecutive_rejects` | n | v1 |
| `confidence_below` | value | v1 |
| `processing_time_above` | ms | v1 |
| `reject_rate_above` | percent, window_minutes, min_samples (default 20) | v1 |
| `warning_present` | warning codes | v1 |
| `schedule` | weekdays + time window | v2 |
| `no_bottles_for` | minutes (watchdog; needs a timer trigger) | v2 |

Each condition has `negate` (bool). Window conditions use the analyses of the same scope,
ordered by time. Rate conditions use hysteresis: once fired, they re-arm only when the rate
falls back below the threshold (or the cooldown elapses, whichever is later).

- **REQ-IN6 Action types:**

| type | params (templated fields marked *) | v |
|---|---|---|
| `notify_ui` | title*, message*, level | v1 |
| `play_sound` | sound (`chime` \| `alarm` \| `beep`) | v1 |
| `email` | target (smtp), to (list), subject*, body* (plain text and optional HTML), attach image (bool) | v1 |
| `webhook` | target, method, path*, headers*, body* (JSON template; default = the AnalysisResult) | v1 |
| `mqtt_publish` | target, topic* (default `bottle_inspection/{{profile.name}}/{{verdict}}`), payload*, qos, retain | v1 |
| `write_file` | target (file), fields | v1 |
| others listed in §12.1 (v2) | | v2 |

- **REQ-IN7 Templates:** Jinja2 `SandboxedEnvironment`, rendering a context with `analysis`,
  `inspection`, `profile`, `rule`, `window_stats` (counts per class in the window) and
  `app_url`. Output is limited to 64 KB. Rendering errors fail the action and are logged. The
  editor shows a **live preview** rendered against the last analysis (or a sample payload)
  and a variable picker.
- **REQ-IN8 Execution:**
  - Rule evaluation is synchronous and cheap, and happens after the analysis is persisted.
  - Actions go to a bounded `asyncio.Queue` (maxsize 1000) consumed by 4 worker tasks. The
    inspection loop never blocks; if the queue is full, the execution is logged as
    `dropped_queue_full`.
  - Retries: 3 attempts (1 s, 5 s, 30 s, with jitter) on network errors, HTTP 429 and 5xx,
    honouring `Retry-After`. No retry on other 4xx.
  - Webhooks treat any 3xx as a failure (redirects are not followed); HTTP 410 disables the
    action and creates a UI notification.
- **REQ-IN9 Webhook signing** (when the target has a secret), per Standard Webhooks:
  - headers `webhook-id` (the execution UUID, the same on retries), `webhook-timestamp`
    (Unix seconds) and `webhook-signature` (`v1,<base64 HMAC-SHA256 of
    "{id}.{timestamp}.{body}">`);
  - the body is signed byte for byte as sent.
- **REQ-IN10 Cooldown and dedupe:** a rule does not fire again for the same dedupe key within
  `cooldown_s`. Suppressed firings are logged as `skipped_cooldown` and are visible in the log
  (counted, not one row each: keep one row per suppression window with a counter).
- **REQ-IN11 Test and log:**
  - "Test rule" evaluates the conditions against the last N analyses (dry run, showing which
    would fire).
  - "Send test" runs the actions once with `is_test = true`.
  - The execution log screen shows status, attempts, HTTP status, duration, the rendered
    payload (secrets redacted) and the first 2 KB of the response. It can be filtered by rule,
    target and status.
  - Retention: 30 days (configurable).

## 13. Data model (SQLite)

SQLAlchemy 2.x ORM, Alembic migrations (`render_as_batch=True`, named constraints).
Connection setup on every connect: `PRAGMA journal_mode=WAL`, `PRAGMA foreign_keys=ON`,
`PRAGMA busy_timeout=5000`. Keep transactions short; never hold a transaction during
inference. The DB file MUST be on a local filesystem (not a WSL-mounted Windows
path).

| table | key columns |
|---|---|
| `profiles` | id, name (unique), description, color, model_ref, algorithm, params_json, verdict_json, background_source, background_path, version, is_builtin, deleted_at, created_at, updated_at |
| `profile_versions` | id, profile_id, version, snapshot_json (full profile settings), created_at |
| `trained_models` | id, model_ref (`<folder>/<algorithm>`, unique), algorithm, path, file_mtime, training_params_json, best_params_json, feature_length, metrics_json (per-class for all 7, read from the CLI outputs), library_versions_json, params_inferred (bool, always false), discovered_at, missing (bool) |
| `cameras` | id, name, kind (`usb`, `rtsp`, `http`), device_index, url, width, height, fps, enabled |
| `sessions` | id, source (`upload`, `dataset`, `browser_camera`, `server_camera`), camera_id, started_at, ended_at, profile_ids_json, note |
| `images` | id, path, sha256, width, height, source, ground_truth_class (nullable), created_at |
| `inspections` | id, session_id, image_id (nullable when not stored), created_at |
| `analyses` | id, inspection_id, profile_id, profile_version, trained_model_id, model_mtime, modified_params (bool), predicted_class, verdict, confidence, scores_json, details_json, geometry_json, timings_json, total_ms, warnings_json, created_at |
| `targets` | id, name, type, config_json, secret_ref, enabled, last_test_at, last_test_ok, last_test_detail |
| `rules` | id, name, description, enabled, severity, scope_json, trigger, condition_logic, cooldown_s, dedupe_key, run_mode, created_at, updated_at |
| `rule_conditions` | id, rule_id, position, type, params_json, negate |
| `rule_actions` | id, rule_id, position, type, target_id, params_json, enabled, timeout_s, max_retries |
| `action_executions` | id, rule_id, action_id, target_id, analysis_id, inspection_id, event_uuid, status (`queued`, `running`, `success`, `failed`, `skipped_cooldown`, `dropped_queue_full`), attempt, http_status, error, rendered_payload (redacted, ≤ 64 KB), response_snippet (≤ 2 KB), duration_ms, is_test, suppressed_count, created_at, finished_at |
| `notifications` | id, level, title, message, link, read_at, created_at |
| `settings` | key, value_json (language, storage policy, model folders, limits, retention) |

Indexes: `analyses(created_at)`, `analyses(profile_id, created_at)`,
`analyses(predicted_class, created_at)`, `analyses(verdict, created_at)`,
`inspections(session_id)`, `action_executions(rule_id, created_at)`.

Retention job (daily): delete images older than N days according to the storage policy
(default: keep 30 days); keep analysis rows forever unless the user purges them in System.

**REQ-D1 (owner decision)** System → Storage has **"Delete all images"**: it removes every
stored image file and thumbnail and sets `inspections.image_id` to NULL, after a confirmation
dialog that shows the number of files and the disk space to be freed. Analyses and reports are
kept, so counts and charts still work and only the pictures disappear. There is also
"Delete all analyses" (separate, with a stronger confirmation that requires typing `DELETE`).

## 14. HTTP / WebSocket / SSE API

All under `/api`, JSON, snake_case, ISO-8601 UTC timestamps, errors as
`{"error": {"code": "...", "message": "...", "details": {...}}}`. The OpenAPI schema at
`/api/docs` is generated by FastAPI.

| Method & path | Purpose |
|---|---|
| `GET /api/health` | Status, versions, dataset found, active cameras, queue sizes |
| `GET /api/classes` | The 7 classes (id, name, default verdict, colour) |
| `GET /api/params/catalog` | Parameter catalog (§7) |
| `GET/POST /api/profiles` · `GET/PATCH/DELETE /api/profiles/{id}` | CRUD |
| `POST /api/profiles/{id}/duplicate` · `GET /api/profiles/{id}/export` · `POST /api/profiles/import` (with `dry_run`) | |
| `GET /api/profiles/{id}/versions` · `GET /api/profiles/{id}/diff?from=&to=` | Version history and diff |
| `GET /api/models` · `POST /api/models/refresh` · `GET /api/models/{ref}` (metadata, training params, validation metrics for all 7 classes, links to the CLI figures) | Trained models (read-only) |
| `POST /api/profiles/{id}/relink` | Copy the current locked values of the model into an `incompatible` profile |
| `POST /api/profiles/{id}/preview` | Multipart image + unsaved params → geometry, the foreground mask (base64 PNG, small) and the geometry for the live preview; no model needed |
| `POST /api/profiles/{id}/background/capture` · `POST /api/profiles/{id}/background/upload` | Background calibration |
| `POST /api/inspections` | Multipart image(s) + `profile_ids` → inspection(s) with analyses (sync for ≤ 10 images, otherwise returns a session and streams via SSE) |
| `GET /api/inspections/{id}` · `GET /api/analyses?filters…` · `GET /api/images/{id}` · `GET /api/images/{id}/thumb` | Read results |
| `GET /api/dataset/samples?split=val&n=&class=` | Dataset sample images with ground truth |
| `GET/POST /api/cameras` · `PATCH/DELETE /api/cameras/{id}` · `POST /api/cameras/{id}/test` · `GET /api/cameras/{id}/mjpeg` | Server cameras |
| `POST /api/sessions` (start: source, camera, profiles, mode, fps) · `POST /api/sessions/{id}/stop` · `POST /api/sessions/{id}/trigger` | Camera sessions |
| `GET /api/reports/summary` · `/timeseries` · `/pareto` · `/profiles` · `/confusion` · `/timings` · `GET /api/reports/export.csv` | Reports (same filter query params) |
| `GET/POST /api/targets` · `PATCH/DELETE /api/targets/{id}` · `POST /api/targets/{id}/test` | Targets |
| `GET/POST /api/rules` · `PATCH/DELETE /api/rules/{id}` · `POST /api/rules/{id}/dry-run` · `POST /api/rules/{id}/test` | Rules |
| `POST /api/templates/preview` | Render a template against a sample context |
| `GET /api/executions?filters…` | Action log |
| `DELETE /api/images` (all images; body `{"confirm": true}`) · `DELETE /api/analyses` (body `{"confirm": "DELETE"}`) | REQ-D1 |
| `GET/PATCH /api/settings` · `GET /api/notifications` · `POST /api/notifications/{id}/read` | |
| `GET /api/events` (SSE) | Events: `analysis.completed`, `inspection.completed`, `models.changed`, `rule.fired`, `action.finished`, `notification.created`, `camera.status` |
| `WS /ws/camera/browser?session_id=` | Client → binary JPEG frames; server → `AnalysisResult` JSON per frame (one frame in flight) |

## 15. UI / UX specification

### 15.1 Principles

1. **The verdict first.** The most important information (PASS / REJECT / REVIEW, and the
   class) is readable from 3 m away on the inspect screen.
2. **Colour only for status.** Use an ISA-101-inspired, calm neutral interface: colour is
   reserved for state (pass, reject, review, warnings) and for class identity in charts and
   overlays. Colour is never the only cue (WCAG 1.4.1): every status also has an icon shape
   and a text label (✓ PASS circle, ✕ REJECT square, ! REVIEW triangle), as in Cognex.
3. **Direct manipulation with precise fallback.** Every geometric parameter can be dragged
   on the image *and* typed. Sliders are always paired with a numeric input.
4. **Show consequences live.** The profile editor previews the effect of each change (ROIs,
   masks, which values differ from training) before saving.
5. **Nothing silent.** Modified parameters, missing or incompatible models, warnings, dropped actions and camera errors are visible
   where they matter, not only in logs.
6. **Keyboard-friendly.** Shortcuts for frequent actions; full keyboard navigation and
   visible focus rings.

### 15.2 Visual design (single fixed theme)

- **Theme.** Light neutral "instrument panel" (no dark mode).
  - Surfaces: background `#F3F4F6`, panels `#FFFFFF`, sunken areas `#ECEEF1`.
  - Borders: `#D9DDE3`.
  - Text: primary `#1B1F24`, secondary `#4A5361`, muted `#6B7480`.
  - Accent for interactive elements: `#2457C5` (links, primary buttons, focus ring).
  - The image viewer stage uses a dark neutral `#1E2329`, so photos and overlays pop. It is
    the only dark surface.
- **Status colours.** All have ≥ 3:1 contrast against their surface; text on them must meet
  4.5:1.
  - PASS: `#1E7B34`, background tint `#E6F4EA`.
  - REJECT: `#C62828`, tint `#FDECEA`.
  - REVIEW and warnings: `#B26A00`, tint `#FFF4E0`.
  - Info: `#2457C5`.
- **Class colours.** Fixed, the same in every chart, overlay and legend (the same palette as
  `didactic.CLASS_COLORS`):
  - `bottle_no_defect` `#2B8A3E`
  - `bottle_other_brand` `#868E96`
  - `defect_low_fill` `#1C7ED6`
  - `defect_no_cap` `#E03131`
  - `defect_loose_cap` `#F08C00`
  - `defect_debris` `#7048E8`
  - `defect_label_damage` `#C2255C`
- **ROI overlay colours** (match the existing figures): cap red `#E03131`, label green
  `#2F9E44`, body blue `#1C7ED6`, reference patch dashed white, bottle box dashed yellow
  `#FAB005`, top line magenta `#D6336C`.
- **Typography.** "IBM Plex Sans" for UI (400/500/600) and "IBM Plex Mono" for numbers,
  class identifiers, timings and parameter keys, with tabular numerals. Base 14 px. The
  inspect verdict banner is 48–64 px semibold. Self-host the fonts (the app works offline).
- **Shape and spacing.** 8 px grid. Radius 6 px (controls) and 10 px (panels). Shadows only
  on floating layers (menus, dialogs, toasts).
- **Motion.** 150–200 ms ease-out for state changes. The verdict banner does a single
  250 ms fade/scale on change, with no looping animation. Honour
  `prefers-reduced-motion`.
- **Icons.** Lucide (shadcn default), 16/20 px.

### 15.3 App shell

```
┌────┬─────────────────────────────────────────────────────────────────────────┐
│ ◉  │  Inspect                                     EN ▾   🔔 3   ● Camera 1  │  top bar
│ ▣  ├─────────────────────────────────────────────────────────────────────────┤
│ ⚙  │                                                                         │
│ ▤  │                          (screen content)                               │
│ ⇄  │                                                                         │
│ ⋯  │                                                                         │
└────┴─────────────────────────────────────────────────────────────────────────┘
 left rail: Inspect · Profiles · Reports · Integrations · System (labels on hover / expanded)
```

- The top bar holds:
  - the screen title;
  - the language switch (EN | PT-BR);
  - the notifications bell (unread count; a drawer with notifications and fired rules);
  - global status chips (active camera sessions; the action queue only when it is backing
    up; a warning when a profile's model is missing or incompatible).
- Toasts appear bottom-right. Critical rule notifications use a persistent toast plus an
  optional sound.

### 15.4 Inspect screen

Two layouts, chosen automatically by the number of selected profiles.

**Single profile:** a large viewer on the left and the result card on the right.

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ Source: [Upload ▾ | Dataset | Browser cam | Camera 1]   Profiles: [SVM (results) ✕][+]│
│ Mode: (•) Single ( ) Continuous 3 fps ( ) Motion      [▶ Run  Space]  [■ Stop]     │
├───────────────────────────────────────────────┬───────────────────────────────────┤
│  ┌───────────── viewer (dark stage) ────────┐ │  ┌─ SVM (results) · v3 · ready ─┐ │
│  │                                          │ │  │   ✕  REJECT                   │ │
│  │        image + overlays (SVG)            │ │  │   defect_loose_cap            │ │
│  │   cap ▭ red · label ▭ green · body ▭ blue│ │  │ confidence 0.83 ⓘ             │ │
│  │                                          │ │  │ 7 scores ▸ (bar list, 7 rows) │ │
│  └──────────────────────────────────────────┘ │  │ 33.8 ms ▸ ▇▇▇▇▇▇▇▇▇▇ (steps)   │ │
│  [Overlays: ☑ROIs ☑Bottle ☐Patch ☐Mask]       │  └───────────────────────────────┘ │
│  [Compare: ◐ raw | normalized]  [🔍 fit 1:1]   │                                   │
├───────────────────────────────────────────────┴───────────────────────────────────┤
│ History ◂ [thumb ✓][thumb ✕][thumb ✓][thumb !][thumb ✓] ... ▸   Session: 214 ✓ 12 ✕ │
└───────────────────────────────────────────────────────────────────────────────────┘
```

**Comparison (2–6 profiles):** one column per profile, up to 3 per row, wrapping to a second
row. Every column is a complete, independent result: its own verdict banner, its own viewer
drawing **its own** geometry (ROIs can differ between profiles), and its own scores and
timings. Below the columns, a comparison table shows one row per profile: verdict, class,
confidence, total ms and modified parameters. Cells whose value differs from the other
rows get a neutral outline, so differences stand out. This is not a combined verdict.

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ Source …   Profiles: [SVM ✕][KNN ✕][RF ✕][+]   [▶ Run]   ☑ Sync zoom/pan           │
├───────────────────────────┬───────────────────────────┬───────────────────────────┤
│ ▌SVM (results) · v3       │ ▌KNN (results) · v1       │ ▌RF (results) · v2 · mod. │
│  ✕ REJECT                 │  ✕ REJECT                 │  ✓ PASS                   │
│  defect_label_damage      │  defect_label_damage      │  bottle_no_defect         │
│ ┌───────────────────────┐ │ ┌───────────────────────┐ │ ┌───────────────────────┐ │
│ │ viewer + its overlays │ │ │ viewer + its overlays │ │ │ viewer + its overlays │ │
│ └───────────────────────┘ │ └───────────────────────┘ │ └───────────────────────┘ │
│ conf 0.17 · 34.1 ms       │ conf 0.11 · 33.0 ms       │ conf 0.30 · 33.4 ms       │
│ 7 scores ▸  timings ▸     │ 7 scores ▸  timings ▸     │ 7 scores ▸  timings ▸     │
├───────────────────────────┴───────────────────────────┴───────────────────────────┤
│ Profile        verdict  class                 confidence  total ms  params        │
│ SVM (results)  REJECT   defect_label_damage   0.17        34.1      training      │
│ KNN (results)  REJECT   defect_label_damage   0.11        33.0      training      │
│ RF (results)   [PASS]   [bottle_no_defect]    0.30        33.4      [modified]    │
├───────────────────────────────────────────────────────────────────────────────────┤
│ History ◂ [thumb ✓✕✕][thumb ✓✓✓] ... ▸   Session per profile: SVM 214✓ 12✕ · KNN … │
└───────────────────────────────────────────────────────────────────────────────────┘
```

- **Viewer.**
  - Zoom: wheel or pinch, buttons, "fit" and "1:1". Pan by dragging. Double-click toggles
    fit and 1:1.
  - In comparison mode, "Sync zoom/pan" (default on) applies the same view to every
    viewer.
  - Overlays are an SVG layer in image coordinates (the relative geometry × image size).
    Each is toggleable: ROIs, bottle box and top line, reference patch, and the foreground
    mask (semi-transparent, from the preview endpoint when requested).
  - Every ROI has a small label tag ("cap", "label", "body").
  - "Raw | normalized" compare uses a draggable before/after slider.
- **Verdict.** Each viewer gets a 4 px status border in its profile's verdict colour
  (Cognex-style). Each result has its own large verdict banner with an icon, the text and
  the class identifier.
- **Result card per profile** (in the user's selection order):
  - the profile colour bar and name, version and status badge (`modified parameters` when
    applicable);
  - the verdict chip;
  - the predicted class (mono);
  - confidence with a tooltip explaining the per-algorithm definition;
  - expandable "7 scores": horizontal bars for all 7 classes, the winner outlined. For KNN
    it says "lower is better" and lists the nearest training thumbnails;
  - expandable "timings": a stacked bar of the steps with a hover breakdown, total in large
    mono digits;
  - warnings as amber chips;
  - when ground truth exists, "✓ matches ground truth" or "✕ expected
    `defect_label_damage`".
- **History filmstrip.** The last 50 inspections. Each thumbnail shows one verdict icon per
  selected profile. Click one to reload it. Session counters are kept per profile: total,
  pass, reject, review, and a pass-rate sparkline.
- **Empty state.** A large drop zone ("Drop bottle images here, or try dataset samples"),
  plus a quick link to connect a camera. When no trained model exists, the message is "No
  trained models found — run `make run` in `code/`, then Refresh".
- **Errors.** Inline, actionable messages, e.g. "Camera permission denied — allow camera
  access in the browser and retry".
- **Shortcuts:**

| Key | Action |
|---|---|
| Space | Run / capture |
| ← / → | Previous / next in history or batch |
| O | Toggle all overlays |
| 1–6 | Focus the result of profile n (comparison mode) |
| F | Fit |
| Esc | Stop session |
| ? | Shortcut sheet |

### 15.5 Profiles screen

```
┌──────────────────────┬────────────────────────────────────────────────────────────┐
│ [+ New] [Import]     │ SVM (results) · v3 · ● Ready      (Duplicate) (Export) ⋯   │
│ [⟳ Refresh models]   │ [Overview] [Parameters] [Model] [History]                   │
│ ┌──────────────────┐ ├───────────────────────────────┬────────────────────────────┤
│ │● SVM (results) ✓ │ │ Search params…  ☐ Only changed│  PREVIEW (sticky)          │
│ │● KNN (results) ✓ │ │ ▾ Localization                 │  [sample image ▾] [upload] │
│ │● RF (results)  ✓ │ │   close_size ─●── [15]  ↺      │  ┌──────────────────────┐  │
│ │● My camera     ◐ │ │ ▸ Photometric normalization    │  │ image with draggable │  │
│ │● Old HOG test  ⊘ │ │ ▸ Bottle crop                  │  │ ROI bands & patch    │  │
│ └──────────────────┘ │ ▾ Regions of interest          │  └──────────────────────┘  │
│ ✓ ready ◐ modified   │   cap  [0.00]–[0.12] ≠ train ↺  │  foreground_fraction 0.09 │
│ ⊘ incompatible       │   cap_size 64×48  🔒 training   │  2 values differ from      │
│ ? model missing      │ ▸ HOG 🔒 · Colour · Ad hoc · …  │  training (not validated)  │
├──────────────────────┴───────────────────────────────┴────────────────────────────┤
│ Unsaved changes (2)                                          [Discard]  [Save]    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

- **List pane.** Shows the profile colour, name, status icon and text, the feature vector and
  algorithm of the trained model it uses (e.g. "rois_hog · svm"). It can be searched. "Refresh models" rescans the model folders.
- **Tabs.**
  - Overview: name, description, colour, trained model (a picker of discovered models
    showing the algorithm, folder and date), background source, verdict settings.
  - Parameters: all groups from the catalog. Locked parameters are shown read-only with a
    lock icon and the training value.
  - Model (read-only): algorithm, best hyperparameters, CV score, and the validation results
    of the CLI run. These are accuracy, macro F1, a per-class table for **all 7 classes**,
    the 7×7 confusion matrix and timing. It also shows the training parameters
    and library versions. A note explains that models are trained and validated by
    `make run`, not here.
  - History: versions with a diff and "restore this version".
- **Parameter rows.**
  - Each row has:
    - the label;
    - a help icon (tooltip with the description and the `config.py` constant name);
    - the control: a slider plus a numeric input; range rows have two handles and two
      inputs; rects are edited on the preview;
    - the unit;
    - a "≠ training" marker showing the training value when the value differs;
    - a per-field reset ↺ back to the training value.
  - Invalid values show an inline error and block saving.
  - Numeric inputs parse locale-aware (pt-BR uses the decimal comma).
- **"Only changed" filter** (values different from training) and a **"Compare with…"**
  action (another profile or a version) showing a side-by-side diff.
- **Live preview.** It calls `POST /api/profiles/{id}/preview`, debounced 250 ms, at a
  reduced resolution while dragging and full resolution on release. It shows:
  - the ROI geometry, patch and mask on the chosen sample image;
  - live geometry values (bottle top, foreground fraction);
  - optionally, "Classify this image" with the unsaved values, as a dry run that is not
    stored.
- **Sticky save bar.** Appears only when there are unsaved changes. Leaving the page with
  unsaved changes asks for confirmation.

### 15.6 Reports screen

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│ [Last 7 days ▾] [Profiles: all ▾] [Source ▾] [Verdict ▾] [Class ▾]   (Export CSV)  │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────────────┤
│ 3 214    │ 94.1 %   │ 189      │ 7        │ 33.9 ms  │ 2.9 /min │ 4 rules fired    │
│ inspected│ pass rate│ rejects  │ review   │ mean time│ thruput  │                  │
├──────────┴──────────┴──────┬───┴──────────┴──────────┴──────────┴──────────────────┤
│ Good vs defective over time│ Defect Pareto (bars + cumulative line)                │
├────────────────────────────┼───────────────────────────────────────────────────────┤
│ Class distribution (7 rows)│ Profiles × classes table · agreement %                 │
├────────────────────────────┴───────────────────────────────────────────────────────┤
│ Analyses table (thumb, time, profile, class, verdict, confidence, ms) → side panel │
└────────────────────────────────────────────────────────────────────────────────────┘
```

- Charts share hover tooltips with exact counts and percentages. Clicking a bar or segment
  applies it as a filter (cross-filtering).
- Empty states explain what produces data ("No analyses in this period").

### 15.7 Integrations screen

Three tabs: **Rules**, **Targets** and **Log**.

- **Rules list.** Enable toggle, name, severity chip, a one-line summary in plain language
  ("When `defect_no_cap` appears 4 times in the last 10 bottles → e-mail QA team, webhook
  MES"), last fired, and the count in 24 h.
- **Rule editor.** A sentence-like builder of WHEN (trigger) → IF (conditions, all/any) →
  THEN (actions), with each block as a card. Condition and action params use the same form
  components as profiles. Template fields use a code editor with variable autocomplete and
  a live preview pane. It has the cooldown and dedupe settings, plus "Dry run on last 100
  analyses" (shows how many times and when it would have fired) and "Send test".
- **Targets.** Cards per target with type icon, status of the last test, and a "Test"
  button.
- **Log.** A table with a status chip; the expandable row shows the payload and response.

### 15.8 System screen

- **Cameras:** CRUD and test snapshot.
- **Trained models:** model folders (default `results`), the list
  of discovered models with feature vector, algorithm, date, validation accuracy and status, and "Refresh".
- **Storage:** image policy, retention, disk usage, **"Delete all images"** and "Delete
  all analyses" (REQ-D1).
- **Dataset path** (for dataset samples) check, with the class counts per split (all 7).
- **Defaults:** max profiles at once, fps limits.
- **About:** versions, the link to the API docs, and export/import of all settings as JSON.

### 15.9 Accessibility

- WCAG 2.2 AA:
  - text contrast 4.5:1;
  - non-text 3:1;
  - focus visible;
  - every control labelled;
  - charts have a "view as table" toggle;
  - status changes are announced with `aria-live="polite"` (`assertive` for REJECT in
    continuous mode, throttled).

## 16. Internationalization (en, pt-BR)

- **REQ-L1** i18next + react-i18next with ICU message format (`i18next-icu`). The locale
  files are `frontend/src/locales/en.json` (the source of truth) and `pt-BR.json`. English is
  the default; the choice is persisted in `settings.language` and local storage.
- **REQ-L2** Numbers, percentages, dates and durations use `Intl` with the active locale.
  Numeric inputs accept and display the locale decimal separator. The API always uses JSON
  numbers.
- **REQ-L3** Class identifiers (`defect_no_cap`, …), parameter keys, algorithm identifiers
  and API fields are **not translated**. Human descriptions are: for example, the
  `defect_no_cap` description is "Cap missing" in English and "Tampa ausente" in Portuguese.
- **REQ-L4** Backend messages shown to users (validation errors, notifications) carry a
  `code` plus parameters; the frontend translates them. Rule templates are user content and
  are not translated.
- **REQ-L5** A CI check fails if `pt-BR.json` lacks any key present in `en.json`.

## 17. Non-functional requirements

| ID | Requirement |
|---|---|
| NF1 | Analysis latency: p95 ≤ 80 ms per image per profile at 1280×720 on the development machine (the CLI measures ≈ 35 ms). With 3 profiles and 3 fps, the CPU must not saturate. |
| NF2 | The UI reflects a camera result within 150 ms of the analysis completing. |
| NF3 | Startup ≤ 5 s with 10 profiles. Models load lazily and are cached in memory (LRU, keyed by artifact id). |
| NF4 | Resilience: a failing action, camera or model load never crashes the server. Each error is logged and surfaced as a notification. |
| NF5 | Logging: structured logs (JSON lines) in `webdata/logs/`, rotated daily, kept 14 days. |
| NF6 | Security baseline without auth: bind to 127.0.0.1; uploads have size limits and are decode-validated; never load user-supplied pickles; Jinja sandbox; secrets file 0600; webhook signing optional. |
| NF7 | Data safety: SQLite WAL; Alembic migrations run on startup with an automatic backup copy of `app.db` before a schema upgrade. |
| NF8 | Offline: no CDN at runtime (fonts, icons and chart libraries are bundled). |
| NF9 | The browser build supports current Chrome, Edge and Firefox. |

## 18. Project layout, Makefile and dependencies

```
code/
├── bottle_inspection/            # CLI package
│   └── params.py                 # InspectionParams (defaults from config.py)
├── bottle_inspection_web/        # NEW backend package
│   ├── __init__.py, __main__.py  # `python -m bottle_inspection_web`
│   ├── app.py                    # FastAPI factory, lifespan (DB, workers, cameras)
│   ├── settings.py               # paths (webdata/), limits, host/port
│   ├── params_catalog.py         # §7
│   ├── api/                      # routers
│   ├── engine/                   # analysis, confidence/scores (reusing didactic logic)
│   ├── models/                   # discovery of CLI-trained models (read-only)
│   ├── cameras/                  # capture threads, MJPEG
│   ├── rules/                    # evaluation, actions, targets, templates, workers
│   ├── reports/                  # SQL aggregations
│   ├── db/                       # models, session, migrations (alembic/)
│   └── tests/
├── frontend/                     # NEW React + Vite + TS app (builds to frontend/dist)
└── webdata/                      # runtime data (git-ignored): app.db, images/, artifacts/,
                                  # feature_cache/, exports/, logs/, secrets.toml
```

**Makefile.** Web targets:

| target | does |
|---|---|
| `make web-setup` | `pip install -r requirements-web.txt` into the existing `.venv`; `npm ci` in `frontend/` |
| `make web-build` | `npm run build` (skipped when `frontend/dist` is newer than the sources) |
| `make web` | web-setup + web-build, then `python -m bottle_inspection_web --host 127.0.0.1 --port 8000` (variables `WEB_HOST`, `WEB_PORT`) |
| | `make web` never trains. It warns, without failing, when no trained model exists ("run `make run` first"). |
| `make web-dev` | uvicorn with `--reload` plus `vite` dev server with an `/api` proxy |
| `make web-test` | pytest for `bottle_inspection_web` + frontend unit tests + the i18n key check |
| `make web-lint` | black --check on `bottle_inspection_web`, eslint and prettier --check on `frontend/` |

`make clean` and `make distclean` are not modified. `webdata/` is never deleted by any
target; it holds user data.

**Python dependencies** (`requirements-web.txt`; pin exact versions at implementation time
and verify them; the values below were the latest on PyPI on 2026-11-09): `fastapi`
(0.162.x), `uvicorn[standard]` (0.54.x), `python-multipart`, `sqlalchemy` (2.1.x),
`alembic` (1.20.x), `pydantic` (v2), `jinja2`, `httpx` (webhooks), `aiosmtplib`,
`paho-mqtt` (2.x), `tomli-w`; dev: `pytest`, `pytest-asyncio`, `respx`.

**Frontend dependencies** (latest on npm on 2026-11-09; verify): `react` 19.x,
`react-dom`, `vite` 8.x (Node ≥ 20.19 / 22.12), `typescript`, `@tanstack/react-query`
(server state), `react-router`, `tailwindcss` + shadcn/ui (Radix primitives), `echarts` 6.x
(with a thin React wrapper), `i18next` + `react-i18next` + `i18next-icu`, `lucide-react`,
`zod` (form validation), `react-hook-form`, `@fontsource/ibm-plex-sans`,
`@fontsource/ibm-plex-mono`; dev: `vitest`, `@testing-library/react`, `eslint`,
`prettier`, `playwright` (e2e).

FastAPI serves `frontend/dist` with an SPA fallback: `app.frontend(...)` if available in the
pinned version, otherwise `StaticFiles(html=True)` plus a catch-all GET route returning
`index.html` for non-`/api` paths.

## 19. Testing and acceptance criteria

**Automated**

- Unit tests:
  - `InspectionParams` defaults equal the `config.py` constants;
  - catalog validation (ranges, cross-field constraints, HOG block fit);
  - confidence formulas;
  - each condition type, including windows, hysteresis and cooldown/dedupe;
  - template sandbox limits;
  - webhook signature against the Standard Webhooks reference test vector;
  - secret redaction.
- **Parity test (MUST):** for every validation image, the web engine with a profile whose
  params equal the training params gives the **same predicted class** as the CLI model on
  the same image (the predictions saved in `<OUTPUT>/<algorithm>/validation_predictions.csv`).
- API tests with a temporary DB and small fixture models: model discovery (including a
  `model.joblib` that fails to load, REQ-M2); profile CRUD/version/diff/import; status derivation
  (`ready`, `modified`, `model_missing`, `incompatible`); inspection through upload returns
  all 7 scores; delete-all-images; the reports endpoints on a seeded DB.
- An end-to-end Playwright test: with models produced by `make quick`, upload a dataset
  validation image and see the verdict, overlays and timings; switch language; create a
  webhook rule pointing to a local test server and observe the delivery in the log.

**Acceptance checklist** (manual demo)

1. `make run` and the other CLI targets run the pipeline.
2. `make web` opens the app at `http://127.0.0.1:8000` in English; switching to PT-BR
   translates every visible string.
3. After `make run`, the app lists the trained models of `results/` and
   creates their built-in profiles. The Model tab shows the
   CLI validation metrics (SVM 100 %, KNN 99.7 %, RF 99.7 %) for all 7 classes. The app has
   no button that trains anything.
4. Upload `damaged_label_0058_20260210_151601.jpg` with all three profiles.
   - Expected: SVM and KNN predict `defect_label_damage`; RF predicts `bottle_no_defect`
     (the known tie at 0.30).
   - Each profile shows its own result in its own column, and the comparison table
     highlights RF's different verdict. There is no combined verdict.
   - The ROIs and timings are shown.
5. Duplicate the SVM profile and move the `cap` ROI in the preview.
   - The profile becomes `modified` and the moved value shows "≠ training".
   - Saving produces v2, which analyses with the same trained model and the new ROI.
   - Locked parameters (e.g. HOG cells) cannot be edited.
   - Reports can separate v1 and v2.
   - "Delete all images" in System removes the pictures, while the reports keep their
     counts.
6. Connect the browser camera and run in continuous mode at 3 fps; results stream without
   UI lag.
7. A rule "`defect_no_cap` 4 times in last 10" sends an e-mail (to a local SMTP debug server)
   and a signed webhook; both appear in the log, and cooldown suppresses repeats.
8. Reports show all 7 classes, the Pareto, the time series, the per-profile table, and the
   confusion matrix for dataset samples. The CSV export opens correctly in a spreadsheet
   with the pt-BR locale.
9. Run `make run` again: the app detects the updated `model.joblib` ("model updated"), and
   the CLI outputs include the keys of REQ-P2b.

## 20. Implementation phases

| Phase | Content | Exit criterion |
|---|---|---|
| 0 | `InspectionParams` + optional `params` args (REQ-P1/P2); training params saved with each model (REQ-P2b) | `make run` re-run once to write the metadata |
| 1 | Backend skeleton: settings, DB + migrations, classes/catalog endpoints, model discovery, profiles CRUD + versions + status, inspection engine (upload + dataset samples), SSE | API tests green; parity test green; analyse via API |
| 2 | Frontend shell, theme, i18n, Inspect (upload + dataset) with viewer and overlays, Profiles editor with live preview | Acceptance items 2–5 |
| 3 | Cameras: browser WebSocket, server capture + MJPEG, modes, storage policy | Item 6 |
| 4 | Reports (SQL aggregations, charts, table, export) | Item 8 |
| 5 | Integrations: targets, rules, conditions, actions (UI, sound, e-mail, webhook, MQTT, file), log, dry run/test | Item 7 |
| 6 | Hardening: accessibility pass, e2e tests, docs (`README` section in pt-BR, docstrings), performance check (NF1–NF3) | Full checklist |

## 21. Decisions (owner, 2026-11-09)

| # | Question | Decision |
|---|---|---|
| 1 | Training on user-labelled images | **No.** Not planned. |
| 2 | Combined verdict of several profiles | **No.** Each profile's result is shown separately for comparison (REQ-G3, REQ-I8). |
| 3 | Image retention | `rejects_only`, 30 days, **plus a "Delete all images" option** (REQ-D1). |
| 4 | Integrations after v1 | **Industrial signals first** (Modbus TCP, OPC UA, GPIO), messaging later (§12.1). |
| 5 | Host/port | `127.0.0.1:8000`. |
| 6 | Training in the web | **No.** The web uses the models trained and validated by the CLI (REQ-G8, §8). |
| 7 | Parameters in the existing package | The CLI/`make` uses the fixed constants; the web passes the profile's values to the same functions; no duplicated code (REQ-P2). |

No open questions remain. Any new ambiguity found during implementation must be raised to
the owner before deviating from this document.

## 22. Research sources

Industrial vision UI patterns:

- Cognex In-Sight results table, status border, job pass/fail:
  - https://docs.cognex.com/is_580/web/EN/ise/Content/EasyBuilder/ResultsTable.htm
  - https://docs.cognex.com/is_590/web/EN/ise/Content/User_Interface/WebHMI/HMI_StatusBorder.htm
  - https://support.cognex.com/docs/is_610/web/en/ezb/content/EasyBuilder/Run_Results.htm
  - https://support.cognex.com/docs/is_572/web/EN/ise/Content/Dialogs/JobPassFailDialog.htm
- Keyence judgment and statistics: https://www.keyence.com/support/manual/measure/tm-x5000/TM-X_UM_03.jsp
- Aurora Vision live preview and HMI:
  - https://docs.adaptive-vision.com/studio/getting_started/PreviewAndDataPresenting.html
  - https://docs.adaptive-vision.com/studio/hmi/HmiControls.html
- MVTec MERLIC recipes: https://www.mvtec.com/doc/merlic/5.7/manual/en-us/Content/RTE_setup/recipes_tab.html
- LandingLens confidence threshold: https://landinglens.docs.landing.ai/performance-report.md
- ISA-101-inspired HMI colour use:
  - https://www.controleng.com/high-performance-hmis-designs-to-improve-operator-effectiveness/
  - https://docs.tatsoft.com/display/FX/ISA-101+HMI+Compliance+How-to
- WCAG 2.2 use of colour and non-text contrast:
  - https://www.w3.org/WAI/WCAG22/Understanding/use-of-color.html
  - https://w3.org/WAI/WCAG22/Understanding/non-text-contrast
- Sliders with numeric input:
  - https://designsystem.maersk.com/updates/articles/2026-03-19-sliders/
  - https://patternfly.org/components/slider/design-guidelines
- Quality dashboards (FPY, Pareto):
  - https://flowfuse.com/blog/2025/08/pareto-chart-manufacturing-guide/
  - https://shoplogix.com/blog/automated-pareto-charts/

Rules and integrations:

- Home Assistant automations, modes:
  - https://www.home-assistant.io/docs/automation/basics/
  - https://www.home-assistant.io/docs/automation/modes/
- Grafana notification policies (grouping, repeat interval):
  https://grafana.com/docs/grafana/latest/alerting/fundamentals/notifications/notification-policies/
- Jinja2 sandbox: https://jinja.palletsprojects.com/en/stable/sandbox/
- Standard Webhooks:
  - https://www.standardwebhooks.com/
  - https://raw.githubusercontent.com/standard-webhooks/standard-webhooks/main/spec/standard-webhooks.md
- aiosmtplib: https://aiosmtplib.readthedocs.io/en/stable/client.html
- paho-mqtt: https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html
- pymodbus: https://pymodbus.readthedocs.io/en/latest/source/client.html
- asyncua: https://github.com/FreeOpcUa/opcua-asyncio
- asyncio queues: https://docs.python.org/3/library/asyncio-queue.html

Stack:

- FastAPI:
  - async: https://fastapi.tiangolo.com/async/
  - SSE: https://fastapi.tiangolo.com/tutorial/server-sent-events/
  - WebSockets: https://fastapi.tiangolo.com/advanced/websockets/
  - files: https://fastapi.tiangolo.com/tutorial/request-files/
  - frontend: https://fastapi.tiangolo.com/tutorial/frontend/
- Process pools: https://docs.python.org/3/library/concurrent.futures.html
- SQLite WAL: https://sqlite.org/wal.html; blobs vs files: https://sqlite.org/fasterthanfs.html
- SQLAlchemy SQLite dialect: https://docs.sqlalchemy.org/en/20/dialects/sqlite.html
- Alembic batch mode: https://alembic.sqlalchemy.org/en/latest/batch.html
- scikit-learn model persistence: https://scikit-learn.org/stable/model_persistence.html
- getUserMedia: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
- canvas.toBlob: https://developer.mozilla.org/en-US/docs/Web/API/HTMLCanvasElement/toBlob
- shadcn/ui with Vite: https://ui.shadcn.com/docs/installation/vite
- i18next plugins (ICU): https://www.i18next.com/overview/plugins-and-utils
- Intl.NumberFormat: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/NumberFormat
