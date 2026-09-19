dev:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

prod:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers $$(nproc) --no-access-log --log-level warning

test:
	uv run pytest -q tests --ignore=tests/e2e

e2e:
	uv run pytest -q tests/e2e

lint:
	uv run ruff check app tests

type:
	uv run mypy app

check: lint type
	uv run pytest -q --cov=app --cov-report=term-missing
