.PHONY: commit test

commit:
	codex exec "run `git add -A` and commit with a nice message"

test:
	PYTHONPATH=. uv run --project . pytest
