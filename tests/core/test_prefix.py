"""Middleware ASGI do prefixo /t/<token>/: o token sai do caminho antes do roteamento."""
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.core.prefix import TokenPrefixMiddleware


def _app():
    app = FastAPI(redirect_slashes=False)
    app.add_middleware(TokenPrefixMiddleware)

    def echo(request: Request):
        return {"path": request.url.path, "token": getattr(request.state, "shunt_path_token", None)}

    @app.post("/v1/messages")
    async def m(request: Request):
        return echo(request)

    # Captura so do teste: toda afirmacao abaixo le o caminho e o token que
    # CHEGARAM ao roteamento, nunca um 404 que teria varias causas possiveis.
    @app.post("/{rest:path}")
    async def anything(rest: str, request: Request):
        return echo(request)

    return app


def test_prefix_is_stripped_before_routing_and_token_lands_in_state():
    with TestClient(_app()) as c:
        r = c.post("/t/abc/v1/messages")
    assert r.status_code == 200 and r.json() == {"path": "/v1/messages", "token": "abc"}


def test_without_prefix_nothing_changes():
    with TestClient(_app()) as c:
        assert c.post("/v1/messages").json() == {"path": "/v1/messages", "token": None}


def test_an_empty_token_segment_is_not_a_prefix():
    with TestClient(_app()) as c:
        assert c.post("/t//v1/messages").json() == {"path": "/t//v1/messages", "token": None}


def test_a_path_that_only_starts_with_t_is_not_a_prefix():
    with TestClient(_app()) as c:
        assert c.post("/tx/abc/v1/messages").json() == {"path": "/tx/abc/v1/messages", "token": None}


def test_the_first_segment_after_t_is_always_the_token():
    # A regra e posicional: `/t/v1/messages` le "v1" como token e serve
    # `/messages`. A dependencia da Task 1.5 recusa "v1" como token
    # desconhecido; o middleware nao tenta adivinhar.
    with TestClient(_app()) as c:
        assert c.post("/t/v1/messages").json() == {"path": "/messages", "token": "v1"}
