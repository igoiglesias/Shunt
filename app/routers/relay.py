"""Repasse de rotas desconhecidas ao host oficial (catch-all da T5).

O router abaixo e registrado por ultimo em `app/main.py`: todo path que
nao casou com `/v1`, `/health`, `/shunt.css`, `/openapi.json`, etc.
e repassado verbatim ao host oficial (`api.anthropic.com` ou
`api.openai.com`). O token do Shunt nunca viaja: sai nos headers
antes do repasse, e a query perde o `token`.

A ordem importa: este router e o ultimo. Tudo que o FastAPI nao casou
com rotas declaradas cai aqui -- inclusive um `/v1/v1/files` mal formado
que nao casou com a rota real de `/v1`.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from app.config.settings import ProviderConfig
from app.core.dispatcher import _exception_text, error_body
from app.core.observability import RelayLog, log_relay
from app.core.official_hosts import OFFICIAL_HOSTS
from app.core.relay import relay_headers, relay_params, relay_target
from app.core.token_auth import TokenRejected, require_shunt_token

router = APIRouter()

# Caminhos que nao sao repassados: respondem local com 404.
RESERVED_EXACT: frozenset[str] = frozenset(
    {
        "/",
        "/health",
        "/shunt.css",
        "/openapi.json",
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
    }
)

# Importado por lazy import dentro da funcao para evitar ciclo.


@router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    include_in_schema=False,
)
async def relay_endpoint(request: Request) -> Response:
    """Repassa a requisicao ao host oficial, byte a byte.

    O path que o cliente pediu e preservado na URL upstream; o proxy
    nao traduz nada aqui (o path do cliente e o path do provedor).
    """
    path = request.url.path

    # Caminhos reservados respondem local, mesmo que tenham caído aqui
    # por alguma rota mal formada.
    if path in RESERVED_EXACT or path.startswith("/admin/"):
        return JSONResponse(status_code=404, content={"detail": "Not Found"})

    # Um path decodificado com caractere nao imprimivel (ex.: null byte
    # vindo de %00) nunca vira URL upstream valida: o httpx levanta
    # InvalidURL ao montar o pedido. Sem rota declarada casar, a resposta
    # volta a ser a do framework em base: 404 local, sem tocar no pool.
    if any(not char.isprintable() for char in path):
        return JSONResponse(status_code=404, content={"detail": "Not Found"})

    # Nenhum path e bloqueado a partir daqui: o catch-all relaya
    # tudo que nao e reservado, inclusive /api/oauth/usage.

    # Token obrigatorio. O TokenRejected sobe ao handler de app/main.py.
    await require_shunt_token(request)

    # Determina o protocolo do host oficial a partir dos headers
    # (nao do path, que e do cliente).
    target_protocol = relay_target(request.headers)
    target_host = OFFICIAL_HOSTS[target_protocol]

    # Monta a URL upstream: origin + path do cliente + query limpa.
    query = relay_params(str(request.url.query))
    url_path = str(request.url).replace(
        f"{request.url.scheme}://{request.url.netloc}", ""
    )
    upstream_url = f"{target_host.origin}{url_path}"
    if query:
        upstream_url = f"{upstream_url}?{'&'.join(f'{k}={v}' for k, v in query)}"

    # Headers a enviar ao upstream: tudo menos os hopping e segredos.
    token_headers: frozenset = getattr(request.state, "token_headers", frozenset())
    headers = relay_headers(request.headers, token_headers)

    # ProviderConfig para o pool.client.
    config = ProviderConfig(
        base_url=target_host.origin,
        protocol=target_protocol,
        api_key=None,
    )

    start = __import__("time").monotonic()
    try:
        async with request.app.state.pool.client(
            target_protocol, config
        ) as client:
            req_kwargs: dict = {
                "url": upstream_url,
                "headers": headers,
                "follow_redirects": False,
            }
            if request.method == "GET":
                req_kwargs["params"] = dict(query) if query else None
            elif request.method in ("POST", "PUT", "PATCH"):
                req_kwargs["content"] = await request.body()

            resp = await client.request(request.method, **req_kwargs)
            elapsed = int((__import__("time").monotonic() - start) * 1000)

            # Corpo byte a byte, headers sobrevoantes.
            body = resp.content
            resp_headers = {
                k: v
                for k, v in resp.headers.items()
                if k.lower() not in {"transfer-encoding", "connection", "keep-alive"}
            }

            log_relay(
                RelayLog(
                    path=path,
                    target=target_protocol,
                    status=resp.status_code,
                    duration_ms=elapsed,
                )
            )

            return Response(
                content=body,
                status_code=resp.status_code,
                headers=resp_headers,
                media_type=resp.headers.get("content-type"),
            )
    except TokenRejected:
        raise
    except Exception as err:
        elapsed = int((__import__("time").monotonic() - start) * 1000)
        from httpx import ConnectError, ReadTimeout

        if isinstance(err, ConnectError):
            log_relay(
                RelayLog(
                    path=path,
                    target=target_protocol,
                    status=502,
                    error_type="upstream_unreachable",
                    duration_ms=elapsed,
                )
            )
            return JSONResponse(
                status_code=502,
                content=error_body(target_protocol, 502, _exception_text(err)),
            )
        if isinstance(err, ReadTimeout):
            log_relay(
                RelayLog(
                    path=path,
                    target=target_protocol,
                    status=504,
                    error_type="upstream_timeout",
                    duration_ms=elapsed,
                )
            )
            return JSONResponse(
                status_code=504,
                content=error_body(target_protocol, 504, _exception_text(err)),
            )
        raise
