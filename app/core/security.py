"""Primitivas de seguranca do painel admin: hash de senha e sessao JWT.

Modulo puro -- nao conhece FastAPI, Request nem banco. Quem usa (a dependencia
`require_admin`, o login, o CRUD de usuarios) injeta o que precisa por
parametro. Separar as primitivas da aplicacao deixa cada uma testavel isolada
e impossibilita uma chamada HTTP vazar para aqui sem passar por essas funcoes.
"""

import time

import argon2
import jwt

# Algoritmo de hash de senha. Argon2id e o vencedor do Password Hashing
# Competition 2015 (resistente a GPU/ASIC e a tempo-espaco). Os parametros do
# `PasswordHasher` padrao ja sao seguros para um painel local; nao os afrouxamos.
_hasher = argon2.PasswordHasher()

_ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    """Gera o hash Argon2 de uma senha. O salt e aleatorio e embutido no retorno."""
    return _hasher.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """Confere uma senha contra o hash, sem levantar.

    Hash invalido, corrompido ou de versao desconhecida vira `False` em vez de
    excecao: o fluxo de login precisa decidir "aceita ou nao", e um erro aqui
    viraria 500 na tela de login.
    """
    try:
        return _hasher.verify(hashed, password)
    except argon2.exceptions.Argon2Error:
        return False


def issue_jwt(user_id: int, secret: str, ttl_seconds: int) -> str:
    """Assina um JWT de sessao (HS256) com as claims `sub`, `iat` e `exp`.

    `sub` e o id do usuario logado. `ttl_seconds` pode ser negativo em teste,
    para forjar uma sessao ja expirada sem dormir.

    O RFC 7519 exige `sub` como string, e o PyJWT rejeita um `sub` inteiro
    (levanta `InvalidSubjectError` no decode). Por isso assina `str(user_id)`
    e converte de volta no `decode_jwt`.
    """
    now = int(time.time())
    payload = {"sub": str(user_id), "iat": now, "exp": now + ttl_seconds}
    return jwt.encode(payload, secret, algorithm=_ALGORITHM)


def decode_jwt(token: str, secret: str) -> dict | None:
    """Valida assinatura + validade e devolve as claims, ou `None`.

    Nunca levanta: token expirado, forjado, com segredo errado ou malformado
    todas caem na `except` e viram `None`. Quem chama trata `None` como "nao
    autenticado". `verify_exp` confere `exp` contra o relogio no decode.
    """
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[_ALGORITHM],
            options={"require": ["exp"]},
        )
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None
    # O `sub` sai da assinatura como string (exigencia do RFC); o contrato
    # interno deste modulo e "o id do usuario como int", entao normaliza aqui.
    # Um `sub` que nao parseia e um token para quem este modulo nao emite --
    # forjado ou de outra aplicacao -- e cai em "nao autenticado" como qualquer
    # outro decode ruim, em vez de subir um `ValueError` pelo `require_admin`.
    sub = claims.get("sub")
    if sub is None:
        return None
    try:
        claims["sub"] = int(sub)
    except (ValueError, TypeError):
        return None
    return claims
