"""Tira o prefixo `/t/<token>/` ANTES do roteamento.

`/t/abc/v1/messages` e servido como `/v1/messages`, com o token em
`scope["state"]["shunt_path_token"]` (o `request.state` do Starlette le dali).
"""
from starlette.types import ASGIApp, Receive, Scope, Send


class TokenPrefixMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket") and scope["path"].startswith("/t/"):
            token, sep, tail = scope["path"][3:].partition("/")
            if sep and token:
                scope = dict(scope)
                scope["path"] = "/" + tail
                scope["raw_path"] = scope["path"].encode()
                scope.setdefault("state", {})["shunt_path_token"] = token
        await self.app(scope, receive, send)
