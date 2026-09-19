dev:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

prod:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers $$(nproc) --no-access-log --log-level warning

test:
	uv run pytest -q tests --ignore=tests/e2e

e2e:
	uv run pytest -q tests/e2e

# Navegador HEADLESS, sempre. Fora do `check` porque sobe servidor e leva
# dezenas de segundos; o `check` tem de continuar em segundos.
browser:
	uv run pytest -q tests/browser

lint:
	uv run ruff check app tests

type:
	uv run mypy app

check: lint type
	uv run pytest -q --cov=app --cov-report=term-missing --ignore=tests/browser
