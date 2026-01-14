.PHONY: test graph install download-models-data

install:
	uv tool install --force --editable .

test:
	PYTHONPATH=. uv run --project . python -m pytest

render-graph:
	PYTHONPATH=. uv run --project . python scripts/render_graph.py

download-models-data:
	wget -O yabot/models.json https://models.dev/api.json
