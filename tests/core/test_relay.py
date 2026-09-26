"""Regras puras do relay: destino, headers e query da rota desconhecida (T5)."""

from app.core.relay import RELAY_DROP, RESPONSE_DROP, relay_headers, relay_params, relay_target


def test_anthropic_signal_headers_pick_anthropic():
    """Sinal forte da Anthropic (`anthropic-` no nome) vence o bearer: sem a
    regra anterior ao `detect_protocol`, o OAuth da Anthropic ia para a OpenAI."""
    assert relay_target(
        {
            "anthropic-beta": "oauth-2025-04-20",
            "authorization": "Bearer x",
            "user-agent": "Bun/1.4.3",
        }
    ) == "anthropic"


def test_claude_cli_user_agent_picks_anthropic():
    assert relay_target({"user-agent": "claude-cli/2.1.280 (external, sdk-cli)"}) == "anthropic"


def test_anthropic_oauth_bearer_picks_anthropic():
    """So o prefixo `sk-ant-` do valor Bearer identifica a Anthropic; outros
    valores seguem para a deteccao por dialeto."""
    assert relay_target({"authorization": "Bearer sk-ant-api03-ABC"}) == "anthropic"


def test_openai_signals_pick_openai():
    assert relay_target({"openai-organization": "org-1"}) == "openai"
    assert relay_target({"user-agent": "openai-python/6.0"}) == "openai"


def test_a_plain_bearer_without_signals_reaches_the_dialect_detection():
    """Bearer genérico: nenhum sinal forte, o dialeto (`authorization`) decide."""
    assert relay_target({"authorization": "Bearer sk-1"}) == "openai"


def test_stub_headers_and_no_headers_default_to_anthropic():
    """Headers exatos da linha 2 do stub do harness (sem `host`, que varia) e
    dict vazio: falta de sinal e Anthropic, o default do plano."""
    assert relay_target(
        {
            "connection": "keep-alive",
            "user-agent": "Bun/1.4.3",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
        }
    ) == "anthropic"
    assert relay_target({}) == "anthropic"


def test_relay_headers_drops_hopping_and_secret_headers_but_keeps_encoding():
    """`accept-encoding` PASSA para o gzip do upstream chegar verbatim; o token
    (`x-shunt-token`) e a `cookie` nunca viajam. Nomes sao ignorados em caixa."""
    out = relay_headers(
        {
            "Host": "api.anthropic.com",
            "Content-Length": "12",
            "Connection": "keep-alive",
            "x-shunt-token": "token-da-suite",
            "cookie": "s=1",
            "accept-encoding": "gzip, deflate, br",
            "anthropic-beta": "oauth-2025-04-20",
            "authorization": "Bearer x",
        },
        frozenset(),
    )
    assert out == {
        "accept-encoding": "gzip, deflate, br",
        "anthropic-beta": "oauth-2025-04-20",
        "authorization": "Bearer x",
    }


def test_relay_headers_removes_the_matched_token_header_only():
    """Com `x-api-key` em `token_headers` (o token casou la, T2), ele sai da
    requisicao e a `authorization` (credencial real) fica."""
    out = relay_headers({"x-api-key": "token-da-suite", "authorization": "Bearer sk-real"}, frozenset({"x-api-key"}))
    assert out == {"authorization": "Bearer sk-real"}


def test_relay_params_strips_the_token_keeps_blanks_and_order():
    """`token` sai da query (case-insensitive); valores vazios e a ordem ficam,
    para o upstream ver a mesma query que o cliente mandou."""
    assert relay_params("beta=true&token=t&x=") == [("beta", "true"), ("x", "")]
    assert relay_params("") == []
    assert relay_params("TOKEN=segredo") == []


def test_the_drop_constants_are_lowercase_frozensets():
    """O T5 e a resposta montam os sets de descartes a partir destas
    constantes; minúsculas por que nomes sao comparados normalizados."""
    assert RELAY_DROP == frozenset(
        {
            "host",
            "content-length",
            "connection",
            "keep-alive",
            "transfer-encoding",
            "te",
            "upgrade",
            "proxy-authorization",
            "proxy-connection",
            "x-shunt-token",
            "cookie",
        }
    )
    assert RESPONSE_DROP == frozenset({"transfer-encoding", "connection", "keep-alive"})
