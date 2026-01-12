.PHONY: test

test:
	PYTHONPATH=. uv run --project . python -m pytest
