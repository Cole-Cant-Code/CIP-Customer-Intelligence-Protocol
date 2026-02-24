.PHONY: test lint format schema

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

schema:
	PYTHONPATH=src python scripts/dev/generate_scaffold_schema.py
