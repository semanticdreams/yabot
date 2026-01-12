.PHONY: test

test:
	PYTHONPATH=. uv run --project . pytest
