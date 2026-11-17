# Dataset folder: DATA=... > BOTTLE_DATA from the environment > BOTTLE_DATA in .env > ../archive
# (see .env.example; the Python scripts follow the same order, in bottle_inspection/env.py)
ENV_BOTTLE_DATA := $(BOTTLE_DATA)
-include .env

PYTHON  ?= python3
VENV    := .venv
PY      := $(VENV)/bin/python
DATA    ?= $(or $(ENV_BOTTLE_DATA),$(BOTTLE_DATA),../archive)
OUTPUT  ?= results
MODELS  ?= svm knn random_forest
FEATURES ?=
QUICK_FEATURES ?= rois_hog rois_lbp rois_color
DIDACTIC ?= 0
THESIS_FIGURES ?= ../TCC_2/figuras/tcc2
PACKAGE := bottle_inspection
RUN     := $(PY) -m $(PACKAGE) --data $(DATA) --models $(MODELS) $(if $(filter 1,$(DIDACTIC)),--didactic-plots,)
FEATURE_ARGS = $(if $(FEATURES),--features $(FEATURES),)

.PHONY: help setup run quick recompute no-normalization all format lint clean distclean thesis-figures theory-figures

help:
	@echo "make setup             create the venv and install dependencies (including black)"
	@echo "make run               extract features, train every model on every feature vector and compare them"
	@echo "make quick             same as run, only on the vectors in QUICK_FEATURES ($(QUICK_FEATURES))"
	@echo "make recompute         ignore the feature cache"
	@echo "make no-normalization  run without photometric normalization (comparison)"
	@echo "make all               run + no-normalization"
	@echo "make format            format the code with black"
	@echo "make lint              check the formatting with black (no changes)"
	@echo "make thesis-figures    draw the thesis figures (Portuguese) into $(THESIS_FIGURES)"
	@echo "make theory-figures    draw the didactic figures of the theory chapter (synthetic data) into $(THESIS_FIGURES)"
	@echo "make clean             remove result folders"
	@echo "make distclean         remove result folders and the venv"
	@echo "Dataset: set BOTTLE_DATA in .env (copy .env.example) or pass DATA=/path"
	@echo "Variables: DATA=$(DATA) OUTPUT=$(OUTPUT) MODELS='$(MODELS)' FEATURES='$(FEATURES)' DIDACTIC=$(DIDACTIC)"
	@echo "  FEATURES='rois_hog bottle_lbp' restricts the feature vectors (default: all 15)"
	@echo "  DIDACTIC=1 also draws classifier_margins.png and one decision figure per validation image"

$(VENV)/.installed: requirements.txt requirements-dev.txt
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements-dev.txt
	touch $@

setup: $(VENV)/.installed

run: setup
	$(RUN) --output $(OUTPUT) $(FEATURE_ARGS)

quick: setup
	$(RUN) --output $(OUTPUT) --features $(QUICK_FEATURES)

recompute: setup
	$(RUN) --output $(OUTPUT) $(FEATURE_ARGS) --recompute

no-normalization: setup
	$(RUN) --output $(OUTPUT)_no_normalization $(FEATURE_ARGS) --no-normalization

all: run no-normalization

# Needs the results of `make all` (results/ and results_no_normalization/)
thesis-figures: setup
	$(PY) thesis/make_figures.py --data $(DATA) --output $(THESIS_FIGURES)

# Self-contained (toy data; only figure 1 reads one dataset image)
theory-figures: setup
	$(PY) thesis/make_theory_figures.py --data $(DATA) --output $(THESIS_FIGURES)

format: setup
	$(PY) -m black $(PACKAGE) thesis

lint: setup
	$(PY) -m black --check --diff $(PACKAGE) thesis

clean:
	rm -rf $(OUTPUT) $(OUTPUT)_no_normalization

distclean: clean
	rm -rf $(VENV)

# =============================================================================
# Web interface
# =============================================================================

PYRIGHT        ?= pyright
WEB_HOST       ?= 127.0.0.1
WEB_PORT       ?= 8000
NPM            ?= npm
FRONTEND       := frontend
WEB_PACKAGE    := bottle_inspection_web

.PHONY: web-help web-setup web-build web web-dev web-test web-lint web-e2e

web-help:
	@echo "make web-setup         install requirements-web*.txt into .venv and npm ci in frontend/"
	@echo "make web-build         build the frontend (skipped when frontend/dist is up to date)"
	@echo "make web               web-setup + web-build, then serve on http://$(WEB_HOST):$(WEB_PORT)"
	@echo "make web-dev           uvicorn --reload + vite dev server (proxy /api)"
	@echo "make web-test          backend tests + frontend unit tests + i18n key check"
	@echo "make web-lint          black --check on the backend, eslint + prettier --check on frontend/"
	@echo "make web-e2e           end-to-end Playwright test (needs the models of 'make quick')"
	@echo "The web app never trains: train and validate the models with 'make run' first."

$(VENV)/.web-installed: $(VENV)/.installed requirements-web.txt requirements-web-dev.txt
	$(PY) -m pip install -q -r requirements-web-dev.txt
	touch $@

$(FRONTEND)/node_modules/.installed: $(FRONTEND)/package.json $(FRONTEND)/package-lock.json
	cd $(FRONTEND) && $(NPM) ci --no-audit --no-fund
	touch $@

web-setup: $(VENV)/.web-installed $(FRONTEND)/node_modules/.installed

FRONTEND_SOURCES := $(shell find $(FRONTEND)/src $(FRONTEND)/public -type f 2>/dev/null) \
	$(wildcard $(FRONTEND)/index.html $(FRONTEND)/vite.config.ts $(FRONTEND)/package.json)

$(FRONTEND)/dist/index.html: $(FRONTEND_SOURCES) $(FRONTEND)/node_modules/.installed
	cd $(FRONTEND) && $(NPM) run build
	touch $@

web-build: $(FRONTEND)/dist/index.html

web: web-setup web-build
	@ls results/*/*/model.joblib >/dev/null 2>&1 || \
		echo "WARNING: no trained model found - run 'make run' first (the web app never trains)"
	$(PY) -m $(WEB_PACKAGE) --host $(WEB_HOST) --port $(WEB_PORT)

web-dev: web-setup
	@echo "backend on http://$(WEB_HOST):$(WEB_PORT), frontend on http://127.0.0.1:5173"
	$(PY) -m $(WEB_PACKAGE) --host $(WEB_HOST) --port $(WEB_PORT) --reload & \
		cd $(FRONTEND) && $(NPM) run dev; kill %1 2>/dev/null || true

web-test: web-setup
	$(PY) -m pytest
	cd $(FRONTEND) && $(NPM) run test
	cd $(FRONTEND) && $(NPM) run i18n:check

web-e2e: web-setup web-build
	cd $(FRONTEND) && npx playwright install chromium && $(NPM) run e2e

web-lint: web-setup
	$(PY) -m black --check --diff $(WEB_PACKAGE)
	$(PYRIGHT) $(WEB_PACKAGE)
	cd $(FRONTEND) && $(NPM) run lint && $(NPM) run format:check
