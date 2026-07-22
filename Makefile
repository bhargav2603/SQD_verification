RUFF_CONFIG := tools/lint/ruff.toml

.PHONY: help install install-chem lint format typecheck test test-core test-ce gate

help:
	@echo "install      - sync the uv workspace (no PySCF; works on Windows)"
	@echo "install-chem - sync including the PySCF 'chem' extra (Linux/macOS only)"
	@echo "lint         - ruff check + format check"
	@echo "format       - ruff format (writes)"
	@echo "typecheck    - mypy over both packages"
	@echo "test         - full suite (PySCF tests skip when PySCF is absent)"
	@echo "gate         - the H2 correctness gate alone"

install:
	uv sync

install-chem:
	uv sync --extra chem

lint:
	uv run ruff check --config $(RUFF_CONFIG) .
	uv run ruff format --check --config $(RUFF_CONFIG) .

format:
	uv run ruff format --config $(RUFF_CONFIG) .

typecheck:
	uv run mypy packages/qdr-core/src packages/compute-engine/src

test:
	uv run pytest -q -rs

test-core:
	uv run pytest packages/qdr-core -q

test-ce:
	uv run pytest packages/compute-engine -q -rs

gate:
	uv run pytest packages/compute-engine/tests/test_stage_00_fci_gate.py -v
