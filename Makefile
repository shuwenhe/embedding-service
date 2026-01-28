.PHONY: help venv install api example test clean web-install web-dev web-build all dev stop

PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PIP := $(BIN)/pip
PY := $(BIN)/python
APP ?= embedding_service.api:app
HOST ?= 0.0.0.0
PORT ?= 8000
RELOAD ?= --reload
NODE_PORT ?= 3000
WEB_DIR ?= web

help:
	@echo "=== Backend Targets ==="
	@echo "  make venv         Create Python virtual environment"
	@echo "  make install      Install Python dependencies"
	@echo "  make api          Run FastAPI server (port 8000)"
	@echo "  make test         Run backend tests"
	@echo "  make example      Run example script"
	@echo ""
	@echo "=== Frontend Targets ==="
	@echo "  make web-install  Install Node.js dependencies"
	@echo "  make web-dev      Run Next.js dev server (port 3000)"
	@echo "  make web-build    Build Next.js for production"
	@echo ""
	@echo "=== Development ==="
	@echo "  make all          Install both backend and frontend"
	@echo "  make dev          Run both API and web servers"
	@echo ""
	@echo "=== Cleanup ==="
	@echo "  make clean        Remove virtualenv and caches"

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

web-install:
	@cd $(WEB_DIR) && npm install

web-dev: web-install
	@cd $(WEB_DIR) && npm run dev

web-build: web-install
	@cd $(WEB_DIR) && npm run build

all: install web-install
	@echo "✓ All dependencies installed"

dev:
	@echo "Starting Embedding Service (API on :8000, Web on :3000)..."
	@echo "API: http://localhost:8000"
	@echo "Web UI: http://localhost:3000"
	@echo "API Docs: http://localhost:8000/docs"
	@echo ""
	@($(MAKE) api) & ($(MAKE) web-dev) & wait

clean:
	@rm -rf $(VENV) __pycache__ .pytest_cache
	@rm -rf $(WEB_DIR)/node_modules $(WEB_DIR)/.next
	@echo "✓ Cleaned up"
