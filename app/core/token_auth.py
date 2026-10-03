"""Token do Shunt: leitura das tres formas, hash, mascara e cache de validacao.

Modulo puro: nao conhece FastAPI nem banco. A dependencia que exige o token
(Task 1.5) usa estas funcoes.
"""
import asyncio
import hashlib
import re
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime

from fastapi import Request
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session


class TokenConflict(ValueError):
    """Token por mais de uma forma, com valores diferentes."""


def token_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def read_token(headers: Mapping[str, str], query_token: str | None, path_token: str | None) -> str | None:
    found = {name: v for name, v in (("header", headers.get("x-shunt-token")), ("path", path_token), ("query", query_token)) if v}
    if len(set(found.values())) > 1:
        raise TokenConflict("shunt token sent in more than one form with different values: " + ", ".join(sorted(found)))
    return next(iter(found.values()), None)


_PREFIX = re.compile(r"^/t/[^/]+/")
_QUERY = re.compile(r"(^|&)token=[^&]*")


def mask_path(path: str) -> str:
    return _PREFIX.sub("/t/***/", path)


def mask_query(query: str) -> str:
    return _QUERY.sub(r"\1token=***", query)


class TokenCache:
    """hash -> (id ou None, vencimento). None e resultado negativo: o valor nao
    e token, e a proxima requisicao com a mesma credencial nao vai ao banco."""

    def __init__(self, ttl: float, clock: Callable[[], float] = time.monotonic, max_entries: int = 4096) -> None:
        self._ttl, self._clock, self._max = ttl, clock, max_entries
        self._entries: dict[str, tuple[int | None, float]] = {}

    def lookup(self, digest: str) -> tuple[bool, int | None]:
        hit = self._entries.get(digest)
        if hit is None:
            return False, None
        token_id, expires = hit
        if self._clock() >= expires:
            del self._entries[digest]
            return False, None
        return True, token_id

    def put(self, digest: str, token_id: int | None) -> None:
        if len(self._entries) >= self._max:
            self.purge()
            if len(self._entries) >= self._max:
                self._entries.pop(next(iter(self._entries)))
        self._entries[digest] = (token_id, self._clock() + self._ttl)

    def invalidate(self, token_id: int) -> None:
        for digest, (tid, _) in list(self._entries.items()):
            if tid == token_id:
                del self._entries[digest]

    def purge(self) -> None:
        now = self._clock()
        for digest, (_, expires) in list(self._entries.items()):
            if now >= expires:
                del self._entries[digest]


# Caminhos que o harness chama sem cabecalho e sem prefixo (passo 0, F1).
# F1 ausente: vazio.
EXEMPT_PATHS: frozenset[str] = frozenset()


class TokenRejected(Exception):
    """Token recusado no envelope do protocolo da rota, nunca `detail`."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status, self.message = status, message


def _db_lookup(engine, digest: str) -> int | None:
    from app.stats.models import ApiToken

    with Session(engine) as session:
        row = session.execute(select(ApiToken).where(ApiToken.token_hash == digest)).scalars().first()
    if row is None:
        return None
    expires = row.expires_at
    if expires is not None:
        # `DateTime(timezone=True)` em SQLite pode voltar sem `tzinfo`; a
        # comparacao com um `now(UTC)` aware sem normalizar dava `TypeError`.
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        if expires < datetime.now(UTC):
            return None
    return row.id


async def _lookup(request: Request, digest: str, *, strict: bool) -> int | None:
    """O id do token, ou None quando o valor nao e token.

    `strict`: o valor veio por uma forma EXPLICITA do token (cabecalho
    `x-shunt-token`, prefixo `/t/`, `?token=`). Sem cache e sem banco nao ha
    como conferir, e a resposta e 503 (R4). Nao estrito: o valor veio em
    `x-api-key`/`Authorization`, que quase sempre e a credencial do harness;
    sem cache e sem banco ele conta como "nao e token" e segue como
    credencial, em vez de derrubar com 503 todo pedido de um Shunt sem banco.
    """
    cache: TokenCache = request.app.state.token_cache
    hit, token_id = cache.lookup(digest)
    if hit:
        return token_id
    recorder = getattr(request.app.state, "recorder", None)
    engine = recorder.engine if recorder is not None else None
    if engine is None:
        if strict:
            raise TokenRejected(503, "shunt has no database to check the token")
        return None  # sem cache negativo: o banco pode voltar (reconnect do recorder)
    # Driver sincrono: fora do event loop.
    try:
        token_id = await asyncio.to_thread(_db_lookup, engine, digest)
    except SQLAlchemyError:  # banco fora do ar tem a mesma resposta que banco
        # ausente: 503 no explicito, "nao e token" no nao-explicito. So erro do
        # SQLAlchemy e tratado como queda: bug do proprio codigo continua 500.
        if strict:
            raise TokenRejected(503, "shunt database is unavailable to check the token")
        return None  # sem cache negativo: o banco pode voltar (reconnect do recorder)
    cache.put(digest, token_id)
    return token_id


async def require_shunt_token(request: Request) -> int:
    """A dependencia que exige o token em `/v1`.

    Seta `request.state.shunt_token_id` (0 quando isento),
    `request.state.credential_is_token` (True quando a credencial apresentada
    E o token: a chave do provedor configurada substitui qualquer outra) e
    `request.state.token_headers` (nomes dos headers onde o token do Shunt
    casou; o repasse de rotas desconhecidas (T5) remove esses headers antes
    de enviar a requisicao ao upstream, entao o token nunca viaja).
    """
    request.state.credential_is_token = False
    matched: set[str] = set()
    if request.url.path in EXEMPT_PATHS:
        request.state.shunt_token_id = None
        request.state.token_headers = frozenset()
        return 0
    try:
        explicit = read_token(
            request.headers, request.query_params.get("token"), getattr(request.state, "shunt_path_token", None)
        )
    except TokenConflict as err:
        raise TokenRejected(400, str(err)) from err
    token_id = None
    if explicit:
        token_id = await _lookup(request, token_hash(explicit), strict=True)
        if token_id is None:
            raise TokenRejected(401, "unknown or expired shunt token")
    # `x-api-key`/`Authorization` que sao um token do Shunt: token, nunca credencial.
    for name in ("x-api-key", "authorization"):
        raw = request.headers.get(name)
        if not raw:
            continue
        value = raw.removeprefix("Bearer ").strip() if name == "authorization" else raw.strip()
        found = token_id if (explicit and value == explicit) else await _lookup(request, token_hash(value), strict=False)
        if found is not None:
            request.state.credential_is_token = True
            token_id = token_id or found
            matched.add(name)
    if token_id is None:
        raise TokenRejected(401, "missing shunt token: send x-shunt-token, /t/<token>/ or ?token=")
    request.state.token_headers = frozenset(matched)
    request.state.shunt_token_id = token_id
    recorder = getattr(request.app.state, "recorder", None)
    if recorder is not None:
        recorder.touch_token(token_id)
    return token_id


async def require_shunt_token_or_transparent(request: Request) -> int | None:
    """Dependencia do relay (T5 catch-all): permite bypass transparente.

    - Token Shunt explicito (x-shunt-token, /t/<token>/, ?token=): valida igual
      ao v1 (strict=True).
    - Credencial do caller (x-api-key ou Authorization que NAO e token Shunt):
      permite bypass transparente (credential_is_token=False, shunt_token_id=None).
      Sem banco para conferir, `strict=False` devolve None, e None aqui significa
      "nao e token do Shunt", nao "falta de credencial" -- sem esta distincao um
      Shunt sem banco derrubaria todo pedido do harness que so leva a propria
      chave (e foi o 401 em /api/oauth/usage medido em producao).
    - Nenhuma credencial: 401.
    """
    request.state.credential_is_token = False
    matched: set[str] = set()
    presented = False
    if request.url.path in EXEMPT_PATHS:
        request.state.shunt_token_id = None
        request.state.token_headers = frozenset()
        return 0
    try:
        explicit = read_token(
            request.headers, request.query_params.get("token"),
            getattr(request.state, "shunt_path_token", None)
        )
    except TokenConflict as err:
        raise TokenRejected(400, str(err)) from err
    token_id = None
    if explicit:
        token_id = await _lookup(request, token_hash(explicit), strict=True)
        if token_id is None:
            raise TokenRejected(401, "unknown or expired shunt token")
        presented = True
    # x-api-key e Authorization: se EXISTE engine (banco), checa se sao
    # token Shunt; se nao ha engine, NAO tentamos validar e tratamos
    # SEMPRE como credencial do caller (bypass transparente). Isso evita
    # vazamento do token do Shunt quando o banco esta frio (cache miss + sem
    # engine): o _lookup strict=False devolve None = "nao consegui conferir",
    # mas o codigo antigo confundia com "nao e token" e deixava o header
    # seguir para o upstream. As formas explicitas (x-shunt-token, /t/, ?token=)
    # continuam validadas (strict=True) e bloqueiam o pedido se invalido.
    recorder = getattr(request.app.state, "recorder", None)
    has_engine = recorder is not None and recorder.engine is not None
    for name in ("x-api-key", "authorization"):
        raw = request.headers.get(name)
        if not raw:
            continue
        presented = True
        value = raw.removeprefix("Bearer ").strip() if name == "authorization" else raw.strip()
        if has_engine:
            found = token_id if (explicit and value == explicit) else await _lookup(request, token_hash(value), strict=False)
            if found is not None:
                request.state.credential_is_token = True
                token_id = token_id or found
                matched.add(name)
        # else: sem engine -> NAO checa, trata como credencial do caller.
        # Header NAO entra em matched, segue para o upstream (bypass transparente).
    if not presented:
        raise TokenRejected(401, "missing shunt token or caller credentials: send x-shunt-token, /t/<token>/, ?token=, or provide x-api-key/Authorization")
    request.state.token_headers = frozenset(matched)
    request.state.shunt_token_id = token_id
    recorder = getattr(request.app.state, "recorder", None)
    if recorder is not None and token_id is not None:
        recorder.touch_token(token_id)
    return token_id
