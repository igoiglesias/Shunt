"""Token do Shunt: leitura das tres formas, hash, mascara e cache de validacao.

Modulo puro: nao conhece FastAPI nem banco. A dependencia que exige o token
(Task 1.5) usa estas funcoes.
"""
import hashlib, re, time
from collections.abc import Callable, Mapping


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
