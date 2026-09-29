"""Repasse de rotas desconhecidas ao host oficial (catch-all da T5).

O router abaixo e registrado por ultimo em `app/main.py`: todo path que
nao casou com `/v1`, `/health`, `/shunt.css`, `/openapi.json`, etc.
e repassado verbatim ao host oficial (`api.anthropic.com` ou
`api.openai.com`). O token do Shunt nunca viaja: sai nos headers
antes do repasse, e a query perde o `token`.

A ordem importa: este router e o ultimo. Tudo que o FastAPI nao casou
com rotas declaradas cai aqui -- inclusive um `/v1/v1/files` mal formado
que nao casou com a rota real de `/v1`.

Tres detalhes do repasse que so importam porque foram medidos:

- **Um catch-all registrado por ultimo vence o `Match.PARTIAL`.** O FastAPI
  responde 405 so quando nenhuma rota casa; com o catch-all presente, um
  `OPTIONS /v1/messages` (path conhecido, metodo sem rota propria) cai nele e
  e repassado, e um `DELETE /health` idem. A unica excecao e o PARTIAL de uma
  rota FORA de `/v1` (`_partial_local_match`): um `POST /api/stats` com token
  valido nao pode subir ao host oficial, e responde 405 local.
- **O gzip passa cru.** A resposta e lida com `aiter_raw()`, que NAO
  descomprime: os bytes sobem intactos e o `content-encoding`/`content-length`
  que os acompanham sao os certos para eles (`.content` descomprimiria e o
  cliente descomprimiria de novo, corrompendo o corpo).
- **A URL usa a ORIGEM do host, nao o `base_url`.** O `base_url` oficial da
  OpenAI ja termina em `/v1`, entao colar o path do cliente sobre ele produzia
  `/v1/v1/files`; `OfficialHost.origin` e so esquema+host, e o path que sobe e
  exatamente o que o cliente pediu.
"""

import contextlib
import time
from collections.abc import AsyncIterator

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from starlette.routing import Match

from app.config.settings import ProviderConfig
from app.core.dispatcher import _exception_text, error_body
from app.core.observability import RelayLog, log_relay
from app.core.official_hosts import OFFICIAL_HOSTS
from app.core.relay import RESPONSE_DROP, relay_headers, relay_params, relay_target
from app.core.token_auth import require_shunt_token

# Reaproveitado do router de /v1: a resposta em stream que SEMPRE fecha o
# gerador do corpo (libera a resposta upstream mesmo com o cliente
# desconectado), com fechamento blindado e teto. Importar em vez de
# duplicar: essa blindagem ja foi medida e testada no /v1.
from app.routers.v1 import ClosingStreamingResponse

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


def _partial_local_match(request: Request) -> bool:
    """Um path FORA de `/v1` casa com rota local mas o metodo nao?

    Desenho 1b do plano: `Match.PARTIAL` (path casa, metodo nao) numa rota
    declarada local fora da familia `/v1` responde 405 local ANTES de
    repassar -- senao um `POST /api/stats` com token valido subiria ate o
    host oficial. `/v1` e `/v1/*` com metodo sem rota propria NAO entram
    aqui: o plano manda repassar (nao existe 404 por falta de sinal).
    """
    path = request.url.path
    # `startswith("/v1")` sozinho casa "/v1x"/"/v10", que nao sao da familia
    # `/v1`. Como estes nao tem rota local declarada, o PARTIAL nao acontece
    # e eles sobem ao host oficial de qualquer jeito -- a precisao aqui e
    # para o dia em que uma rota `/v1x` existir e tiver que ser repassada.
    if path == "/v1" or path.startswith("/v1/"):
        return False
    for route in request.app.routes:
        matches = getattr(route, "matches", None)
        if matches is None:
            continue
        # Pula o catch-all deste proprio router: ele casa qualquer path, e
        # um metodo fora da lista dele nao deve virar 405 local (o plano
        # manda repassar). Identificar pelo endpoint e estavel: o path e
        # uma convencao de string que um dia pode mudar.
        if getattr(route, "endpoint", None) is relay_endpoint:
            continue
        match, _ = matches(request.scope)
        if match is Match.PARTIAL:
            return True
    return False


def _raw_response_headers(response: httpx.Response) -> list[tuple[bytes, bytes]]:
    """Headers da resposta upstream sem os hopping (que nao fazem sentido
    repassados), como LISTA DE DUPLAS preservando ordem e duplicatas.

    Por que lista e nao dict: dois `Set-Cookie` upstream sao dois headers
    separados, e `Headers.items()` os junta com ", " -- o que corrompe o
    `Expires`/`Date` de cada cookie (ambos tem virgula), virando um cookie
    so para o cliente. `multi_items()` entrega cada duplicata em separado.

    O `content-encoding` e o `content-length` sobem verbatim: sao os
    corretos para os bytes crus que o corpo carrega."""
    drop = RESPONSE_DROP
    return [
        (k.encode("latin-1"), v.encode("latin-1"))
        for k, v in response.headers.multi_items()
        if k.lower() not in drop
    ]


async def _close_on_error(
    response: httpx.Response | None, stack: contextlib.AsyncExitStack
) -> None:
    """Fecha resposta e stack no caminho de erro (antes do status)."""
    if response is not None:
        await response.aclose()
    await stack.aclose()


@router.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    include_in_schema=False,
)
async def relay_endpoint(request: Request) -> Response:
    """Repassa a requisicao ao host oficial, byte a byte.

    O path que o cliente pediu e preservado na URL upstream; o proxy
    nao traduz nada aqui (o path do cliente e o path do provedor). A query
    limpa vai em `params` (lista de pares, ordem e duplicatas intactas) e
    NUNCA colada na string da URL: medido que a URL bruta do pedido levava
    o `?token=...` original ao upstream quando so o token estava na query,
    e um segundo `?` em nao-GET montava URL mal formada.

    A resposta em stream vive MAIS que o handler: o `AsyncExitStack`
    segura o cliente do pool ate o `finally` do gerador do corpo (senao o
    cliente -- proprio, pois o host oficial nao e um provedor do catalogo
    -- fecharia antes de o primeiro byte subir).
    """
    path = request.url.path

    # Caminhos reservados respondem local, mesmo que tenham caido aqui
    # por alguma rota mal formada.
    if path in RESERVED_EXACT or path.startswith("/admin/"):
        return JSONResponse(status_code=404, content={"detail": "Not Found"})

    # Um path decodificado com caractere nao imprimivel (ex.: null byte
    # vindo de %00) nunca vira URL upstream valida: o httpx levanta
    # InvalidURL ao montar o pedido. Sem rota declarada casar, a resposta
    # volta a ser a do framework em base: 404 local, sem tocar no pool.
    if any(not char.isprintable() for char in path):
        return JSONResponse(status_code=404, content={"detail": "Not Found"})

    # Rota local declarada com o path mas sem o metodo (fora de /v1):
    # 405 local, o upstream nao e tocado (design 1b do plano).
    if _partial_local_match(request):
        return JSONResponse(status_code=405, content={"detail": "Method Not Allowed"})

    # Nenhum path e bloqueado a partir daqui: o catch-all repassa
    # tudo que nao e reservado, inclusive /api/oauth/usage.

    # Token obrigatorio. O TokenRejected (nao e httpx.HTTPError) sobe ao
    # handler de app/main.py.
    await require_shunt_token(request)

    # Determina o protocolo do host oficial a partir dos headers
    # (nao do path, que e do cliente).
    target_protocol = relay_target(request.headers)
    target_host = OFFICIAL_HOSTS[target_protocol]

    # Query limpa (sem `token`, com ordem, duplicatas e valores vazios
    # intactos). A URL upstream tem APENAS origin + path: a query entra
    # uma unica vez, em `params`, para todo metodo.
    query = relay_params(str(request.url.query))
    upstream_url = f"{target_host.origin}{path}"

    # Headers a enviar ao upstream: tudo menos os hopping e segredos.
    token_headers: frozenset[str] = getattr(
        request.state, "token_headers", frozenset()
    )
    headers = relay_headers(request.headers, token_headers)

    # ProviderConfig para o pool.client.
    config = ProviderConfig(
        base_url=target_host.origin,
        protocol=target_protocol,
        api_key=None,
    )

    start = time.monotonic()
    # O stack segura o cliente ate o finally do corpo: o host oficial nao e
    # um provedor do catalogo, entao o pool devolve um cliente PROPRIO que
    # fecha na saida -- e a saida so pode chegar depois do ultimo byte.
    stack = contextlib.AsyncExitStack()
    client: httpx.AsyncClient = await stack.enter_async_context(
        request.app.state.pool.client(target_protocol, config)
    )

    response: httpx.Response | None = None
    try:
        req_kwargs: dict = {
            "url": upstream_url,
            "params": query,
            "headers": headers,
        }
        if request.method in ("POST", "PUT", "PATCH"):
            req_kwargs["content"] = await request.body()

        built = client.build_request(request.method, **req_kwargs)
        # `stream=True`: so os headers descem; o corpo sobe a seguir e o
        # finally do gerador (abaixo) fecha a resposta.
        response = await client.send(built, stream=True)
        elapsed = int((time.monotonic() - start) * 1000)

        if request.method == "HEAD":
            # HEAD nao tem corpo: sem stream. O aclose e imediato (headers
            # so) e o stack libera o cliente antes do return.
            # `headers={}` (Mapping vazio) e a forma que o Starlette aceita
            # (`init_headers` chama `.items()`); os headers reais entram
            # depois, no atributo publico que o ASGI le no send.
            resp = Response(status_code=response.status_code, headers={})
            resp.raw_headers = _raw_response_headers(response)
            await response.aclose()
            await stack.aclose()
            return resp

        # O log do relay sai AGORA, quando os headers do upstream chegam
        # (o status ja e conhecido), medindo a duracao ate o primeiro
        # header: a resposta em stream e entregue ao cliente ANTES de o
        # corpo terminar, entao esperar o final aqui registraria depois do
        # `return` -- nunca chegaria.
        log_relay(
            RelayLog(
                path=path,
                target=target_protocol,
                status=response.status_code,
                duration_ms=elapsed,
            )
        )

        # Corpo cruso via aiter_raw(): httpx NAO descomprime, entao o
        # `content-encoding: gzip` e o `content-length` que sobem sao os
        # corretos para esses bytes. Medido: `.content` descomprime sozinho
        # e corrompia a resposta (o cliente descomprimia de novo). O
        # finally fecha a resposta upstream E o stack (o cliente proprio do
        # pool nao pode morrer antes do ultimo byte subir).
        async def body() -> AsyncIterator[bytes]:
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                try:
                    await response.aclose()
                finally:
                    await stack.aclose()

        # A stream precisa do `media_type` (vira content-type no cliente),
        # mas `init_headers` so aceita Mapping e colapsaria duplicatas --
        # dois `Set-Cookie` upstream tem de chegar separados. Sobe-se um
        # Mapping vazio e entao se atribui o atributo publico que o ASGI le
        # no send (`raw_headers`, medido em starlette/responses.py: e lido
        # por `stream_response` e nunca re-derivado). Por isso o
        # content-type entra na lista explicitamente: o que a `init_headers`
        # populou fica de fora dela.
        stream_response = ClosingStreamingResponse(
            body(),
            status_code=response.status_code,
            headers={},
            media_type=response.headers.get("content-type"),
        )
        stream_response.raw_headers = _raw_response_headers(response)
        return stream_response
    except httpx.TimeoutException as err:
        # Qualquer timeout (pool, connect, read, write) e 504. Medido que
        # `ReadTimeout` so cobria um deles: o pool expira antes de conectar
        # (PoolTimeout) e isso virava 500.
        await _close_on_error(response, stack)
        elapsed = int((time.monotonic() - start) * 1000)
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
    except httpx.HTTPError as err:
        # O resto (conexao recusada, reset, protocolo distante) e 502.
        await _close_on_error(response, stack)
        elapsed = int((time.monotonic() - start) * 1000)
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
    except BaseException:
        # Seja o que for que quebre entre o `enter_async_context` e o
        # `return` do stream (ex.: um erro de parsing do corpo do pedido),
        # o stack nao pode vazar um cliente proprio aberto.
        if response is not None:
            await response.aclose()
        await stack.aclose()
        raise
