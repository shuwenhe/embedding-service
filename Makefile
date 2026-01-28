.PHONY: help venv install api example test clean

PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PY := $(BIN)/python
APP ?= embedding_service.api:app
HOST ?= 0.0.0.0
PORT ?= 8000
RELOAD ?= --reload

help:
	@echo "Targets:"
	@echo "  make venv      Create virtual environment"
	@echo "  make install   Install Python dependencies"
	@echo "  make api       Run FastAPI server (uvicorn)"
	@echo "  make example   Run example script"
	@echo "  make test      Run test_service.py"
	@echo "  make clean     Remove virtualenv and caches"

venv:
	@$(PYTHON) -m venv $(VENV)
	@$(PIP) install --upgrade pip

install: venv
	@$(PIP) install -r requirements.txt

api: install
	@$(PY) -m uvicorn $(APP) --host $(HOST) --port $(PORT) $(RELOAD)

example: install
	@$(PY) example.py

test: install
	@$(PY) test_service.py

clean:
	@rm -rf $(VENV) __pycache__ .pytest_cache
