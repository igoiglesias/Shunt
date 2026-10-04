dev:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Teto de 5 s: o uvicorn instalado nao avisa uma resposta em curso (SSE do painel) no desligamento; sem o teto ele espera sem prazo e o supervisor junto. Ver tests/test_shutdown.py.
prod:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers $$(nproc) --no-access-log --log-level warning --timeout-graceful-shutdown 5

test:
	uv run pytest -q -n auto --dist loadfile tests --ignore=tests/e2e

e2e:
	uv run pytest -q -n auto --dist loadfile tests/e2e

# Navegador HEADLESS, sempre. Fora do `check` porque sobe servidor e leva
# dezenas de segundos; o `check` tem de continuar em segundos.
browser:
	uv run pytest -q -n 2 --dist loadfile tests/browser

lint:
	uv run ruff check app tests

type:
	uv run mypy app

check: lint type
	uv run pytest -q -n auto --dist loadfile --cov=app --cov-report=term-missing --ignore=tests/browser

# Troca a senha de um administrador do painel direto no banco (operacional: a
# tela de login exige a senha atual, inutil se o operador a perdeu). Imprime o
# hash anterior para auditoria. Ex.: make change_pass USER=iglesias PASS='nova'
change_pass:
	@if [ -z "$(USER)" ]; then echo "uso: make change_pass USER=<nome> [PASS=<senha>]"; exit 2; fi
	@if [ -z "$(PASS)" ]; then \
		echo "uso: make change_pass USER=$(USER) PASS=<senha>"; \
		echo "  (ou SHUNT_NEW_PASS=... para nao deixar a senha no historico do shell)"; \
		exit 2; \
	fi
	uv run python -m scripts.change_pass "$(USER)" "$(PASS)"
