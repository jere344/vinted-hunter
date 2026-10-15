PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help install test lint format run clean

help:
	@echo "Targets:"
	@echo "  install  Create $(VENV) and install the package with dev extras"
	@echo "  test     Run the test suite (live tests are skipped)"
	@echo "  lint     Run ruff"
	@echo "  format   Run ruff format"
	@echo "  run      Start the web app"
	@echo "  clean    Remove caches and build output"

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev]"

test:
	$(BIN)/python -m pytest -q

lint:
	$(BIN)/ruff check .

format:
	$(BIN)/ruff format .

run:
	$(BIN)/vinted-hunter

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
