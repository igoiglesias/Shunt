"""Pecas do contrato publicado no /docs, repetidas entre os routers.

So documento: nada aqui valida nem muda resposta. O nome do esquema de
seguranca (`adminSession`) e o que `app/main.py` declara em `securitySchemes`;
o texto e em ingles porque e o que quem integra le no Swagger.
"""

from typing import Any

# A sessao do admin (cookie `shunt_admin`) que `require_admin` confere.
ADMIN_SESSION: list[dict[str, list[str]]] = [{"adminSession": []}]

# `require_admin` em rota `/api/...` sempre responde 401, nunca redirect.
UNAUTHORIZED: dict[int | str, dict[str, Any]] = {
    "401": {"description": "No valid admin session cookie."}
}

# O 409 das rotas de historico quando nao ha banco configurado.
NO_DATABASE: dict[int | str, dict[str, Any]] = {
    "409": {"description": "No database is configured, so there is no history."}
}
