"""Módulo de configuração: lê variáveis de ambiente e define defaults operacionais.

Este é o único ponto de leitura de ambiente para knobs ajustáveis. Constantes
que NÃO são knobs operacionais (mapas de protocolo, regex, prompts, strings de
exibição) FICAM nos seus módulos de origem e NÃO são movidas para cá.
"""

import os

from dotenv import load_dotenv

# Carrega .env uma vez na importação do módulo
load_dotenv()


# --- Timeouts e budgets de upstream / dispatcher / attempt ---
MAX_ATTEMPTS = int(os.environ.get("SHUNT_MAX_ATTEMPTS", "3"))
RETRY_AFTER_BUDGET = float(os.environ.get("SHUNT_RETRY_AFTER_BUDGET", "5"))
TOTAL_DEADLINE = float(os.environ.get("SHUNT_TOTAL_DEADLINE", "120"))
FIRST_EVENT_DEADLINE = float(os.environ.get("SHUNT_FIRST_EVENT_DEADLINE", "20"))

ENGINE_BOOT_TIMEOUT = float(os.environ.get("SHUNT_ENGINE_BOOT_TIMEOUT", "10"))
# `REACH_TIMEOUT`: medido no boot -- com a URL apontada para uma porta morta o
# driver do libsql NAO levanta e NAO volta (bloqueia segurando o GIL, nem uma
# thread paralela imprime): um Turso fora do ar congelaria o proxy inteiro.
# A defesa e nunca deixar o driver tocar um host inalcancavel: um `connect` de
# socket comum, com relogio, decide antes.
REACH_TIMEOUT = float(os.environ.get("SHUNT_REACH_TIMEOUT", "2"))
# `BUSY_TIMEOUT_MS`: `make prod` sobe um processo por nucleo, e oito processos
# gravando no mesmo arquivo com o journal padrao se atropelam; sem prazo o
# primeiro que pegar a trava bloqueia os outros para sempre.
BUSY_TIMEOUT_MS = int(os.environ.get("SHUNT_BUSY_TIMEOUT_MS", "5000"))

TIMEOUT_CONNECT = float(os.environ.get("SHUNT_TIMEOUT_CONNECT", "10"))
TIMEOUT_READ = float(os.environ.get("SHUNT_TIMEOUT_READ", "60"))
TIMEOUT_WRITE = float(os.environ.get("SHUNT_TIMEOUT_WRITE", "30"))
TIMEOUT_POOL = float(os.environ.get("SHUNT_TIMEOUT_POOL", "10"))

DEFAULT_MAX_OUTPUT_TOKENS = int(os.environ.get("SHUNT_DEFAULT_MAX_OUTPUT_TOKENS", "4096"))
PING_INTERVAL = float(os.environ.get("SHUNT_PING_INTERVAL", "5"))

# --- Token do Shunt e credencial do harness ---
# Cache de validacao do token: hash -> id, por processo. Curto: a revogacao
# no mesmo processo invalida na hora, em outro worker vale ate o TTL.
TOKEN_CACHE_TTL = float(os.environ.get("SHUNT_TOKEN_CACHE_TTL", "60"))
# Credencial do harness guardada so em memoria, por token e destino.
CLIENT_CREDENTIAL_TTL = float(os.environ.get("SHUNT_CLIENT_CREDENTIAL_TTL", str(12 * 60 * 60)))

# --- Passagem direta ---
PASSTHROUGH_BODY_LIMIT = int(os.environ.get("SHUNT_PASSTHROUGH_BODY_LIMIT", str(64 * 1024 * 1024)))
PASSTHROUGH_TIMEOUT_CONNECT = float(os.environ.get("SHUNT_PASSTHROUGH_TIMEOUT_CONNECT", "10"))
PASSTHROUGH_TIMEOUT_READ = float(os.environ.get("SHUNT_PASSTHROUGH_TIMEOUT_READ", "120"))
PASSTHROUGH_TIMEOUT_WRITE = float(os.environ.get("SHUNT_PASSTHROUGH_TIMEOUT_WRITE", "30"))
PASSTHROUGH_TIMEOUT_POOL = float(os.environ.get("SHUNT_PASSTHROUGH_TIMEOUT_POOL", "10"))

# --- Admin / sessão ---
ADMIN_COOKIE = "shunt_admin"
# `path` precisa ser `"/"` e nao `"/admin"` (defeito medido): as paginas do
# admin (em `/admin/...`) fazem fetch de dados em `/api/...` -- painel e
# requisicoes, ve `app/routers/dashboard.py` e `audit.py`. Com `path="/admin"`
# o navegador nao envia o cookie para `/api`, os fetch caem em 401 e o painel
# abre zerado. O mesmo dominio serve admin e API, entao o alcance inteiro nao
# vaza para fora; a protecao real e o JWT, mantido `httponly` (nenhum script le
# o token) e `samesite=strict` (so navegacao dentro do proprio dominio).
ADMIN_COOKIE_PATH = "/"
ADMIN_COOKIE_MAX_AGE = int(os.environ.get("SHUNT_ADMIN_COOKIE_MAX_AGE", str(12 * 60 * 60)))
LOGIN_URL = "/admin/login"

# --- Recorder / stats ---
MAX_QUEUE = int(os.environ.get("SHUNT_MAX_QUEUE", "10000"))
BATCH_SIZE = int(os.environ.get("SHUNT_BATCH_SIZE", "200"))
INTERVAL = float(os.environ.get("SHUNT_INTERVAL", "1"))
SUBSCRIBER_QUEUE = int(os.environ.get("SHUNT_SUBSCRIBER_QUEUE", "100"))
DRAIN_TIMEOUT = float(os.environ.get("SHUNT_DRAIN_TIMEOUT", "5"))
RECONNECT_SECONDS = float(os.environ.get("SHUNT_RECONNECT_SECONDS", "30"))

# --- Bodies / audit / queries ---
# Teto por campo de corpo gravado. `SHUNT_STORE_BODIES` nao e knob daqui: o
# interruptor da captura mora em `app/stats/bodies.py` (leitura de ambiente por
# chamada, desligado por padrao, aceita "sim"/"on") -- duplicar o default aqui
# deixaria duas fontes do mesmo interruptor.
DEFAULT_LIMIT = int(os.environ.get("SHUNT_BODY_LIMIT", "64000"))

DEFAULT_HOURS = float(os.environ.get("SHUNT_DEFAULT_HOURS", "24"))
MAX_HOURS = float(os.environ.get("SHUNT_MAX_HOURS", str(24 * 30)))
EXPORT_LIMIT = int(os.environ.get("SHUNT_EXPORT_LIMIT", "5000"))
SEARCH_LIMIT = int(os.environ.get("SHUNT_SEARCH_LIMIT", "50"))
MAX_SEARCH_LIMIT = int(os.environ.get("SHUNT_MAX_SEARCH_LIMIT", "500"))
TOOL_LIMIT = int(os.environ.get("SHUNT_TOOL_LIMIT", "8"))

# --- Dossier / analysis / sse ---
ROW_LIMIT = int(os.environ.get("SHUNT_DOSIER_ROW_LIMIT", "5000"))
ANSWER_PIECES = int(os.environ.get("SHUNT_ANSWER_PIECES", "20000"))
ANALYSIS_MAX_OUTPUT_TOKENS = int(os.environ.get("SHUNT_ANALYSIS_MAX_OUTPUT_TOKENS", "4000"))

# As constantes abaixo são INVARIANTES e ficam nos seus módulos originais:
# - PROVIDER_HINTS (app/core/resolver.py)
# - IMAGE_PART_TYPES (app/core/capabilities.py)
# - SECRET_HEADERS, REDACTED (app/core/observability.py)
# - LABEL, MAX_PATH, SESSION_LIMIT (app/core/project.py) -- manter local se invariantes
# - DEFAULT_PORTS (app/stats/engine.py)
# - BUCKETS, DAY_MINUTES (app/stats/queries.py)
# - CSV_COLUMNS (app/routers/audit.py)
# - MAX_OUTPUT_TOKENS (app/stats/analysis.py) -- este usa ANALYSIS_MAX_OUTPUT_TOKENS
# - STOP_REASONS, ERROR_TYPES (app/translate/to_anthropic.py)
# - FINISH_REASONS, MAX_STOP_SEQUENCES (app/translate/to_openai.py)
# - ANTHROPIC_PREFIX, OPENAI_PREFIX, ENCODED_MARK, CHECKSUM_LEN (app/translate/ids.py)
# - READ, WRITE (app/translate/usage.py)
# - TOOL_CHOICE (app/translate/to_anthropic_request.py)

# O catálogo de roteamento (providers/models/routes/default_model) foi movido
# para app/config/seed.py. Em runtime, o catálogo vive exclusivamente no banco.
# Este arquivo NÃO define mais providers/models/routes.
