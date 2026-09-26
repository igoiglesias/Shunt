"""Aceite do bug do harness: transparente para provedor nao declarado (T6).

E2E_SETTINGS (conftest.py) declara SO openrouter/anthropic/local -- NEM
"openai" NEM "anthropic" como provedor com o nome derivado de `claude-*`.
O fixture deep-copia, entao `del settings.providers["anthropic"]` remove o
provider fake e o `_transparent` cai no host oficial embutido (spec R1).

Cenarios cobertos:
- Sem 'anthropic' declarado, sem default, sem rota casando: `claude-*`
  vira transparente unico, servido pelo host oficial embutido.
- R2: credencial Anthropic pedindo gpt-* sem default -> 400, nada viaja.
- R2 simetrico: credencial OpenAI pedindo claude-* -> 400.
- Familia rota + no default: a cadeia da rota falha (sem chave), e o
  transparente oficial fecha com o model original.
- T4: token do Shunt como unica credencial -> pulado, 400.
"""

from tests.e2e.fake_provider import Scripted

OFFICIAL_ANSWER = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5",
    "stop_reason": "end_turn",
    "content": [{"type": "text", "text": "pelo host oficial"}],
    "usage": {"input_tokens": 3, "output_tokens": 2},
}

OPENAI_ANSWER = {
    "id": "chatcmpl_1",
    "object": "chat.completion",
    "model": "gpt-5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "pelo host oficial openai"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
}


def test_undeclared_anthropic_is_served_from_the_official_host(shunt, provider, settings):
    """Sem 'anthropic' no catalogo, sem default e sem rota casando: o nome
    `claude-*` vira o candidato transparente unico, servido pelo host oficial
    embutido (spec R1), em vez do 500 de `KeyError` do bug."""
    del settings.providers["anthropic"]
    settings.default_model = None

    provider.queue(Scripted(json_body=OFFICIAL_ANSWER))
    response = shunt.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "oi"}],
        },
        headers={
            "x-api-key": "sk-ant-do-harness",
            "anthropic-version": "2023-06-01",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["content"][0]["text"] == "pelo host oficial"
    assert body["model"] == "claude-sonnet-5"

    (call,) = provider.calls
    assert call.url.netloc == b"api.anthropic.com", call.url
    assert call.url.path == "/v1/messages"
    assert call.headers["x-api-key"] == "sk-ant-do-harness"
    assert provider.bodies[0]["model"] == "claude-sonnet-5"


def test_undeclared_openai_is_served_from_the_official_host(shunt, provider, settings):
    """gpt-* sem 'openai' declarado (NUNCA esteve no E2E_SETTINGS), sem default,
    e sem rota casando: transparente unico servido pelo host oficial openai."""
    settings.default_model = None

    provider.queue(Scripted(json_body=OPENAI_ANSWER))
    response = shunt.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "oi"}],
        },
        headers={"authorization": "Bearer sk-do-cliente-openai"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "pelo host oficial openai"
    assert body["model"] == "gpt-5"

    (call,) = provider.calls
    assert call.url.netloc == b"api.openai.com", call.url
    assert call.url.path == "/v1/chat/completions"
    assert call.headers["authorization"] == "Bearer sk-do-cliente-openai"
    assert provider.bodies[0]["model"] == "gpt-5"


def test_anthropic_credential_asking_gpt5_no_default_returns_400(shunt, provider, settings):
    """T5 E2E, o exemplo canonico: Claude Code (credencial Anthropic) pedindo
    gpt-5 sem default. O transparente openai NAO e elegivel para a credencial
    Anthropic (spec R2): candidato pulado, 400 no envelope Anthropic,
    NENHUM upstream chamado."""
    del settings.providers["anthropic"]
    settings.default_model = None

    response = shunt.post(
        "/v1/messages",
        json={
            "model": "gpt-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "oi"}],
        },
        headers={
            "x-api-key": "sk-ant-do-harness",
            "anthropic-version": "2023-06-01",
        },
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "gpt-5" in body["error"]["message"]
    assert provider.calls == []


def test_openai_credential_asking_claude_no_default_returns_400(shunt, provider, settings):
    """T5 simetrico: cliente OpenAI pedindo claude-* sem default. O transparente
    anthropic NAO e elegivel para a credencial OpenAI: 400 no envelope OpenAI,
    nenhum upstream chamado."""
    del settings.providers["anthropic"]
    settings.default_model = None

    response = shunt.post(
        "/v1/chat/completions",
        json={
            "model": "claude-sonnet-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "oi"}],
        },
        headers={"authorization": "Bearer sk-do-cliente-openai"},
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert "claude-sonnet-5" in body["error"]["message"]
    assert provider.calls == []


def test_family_route_fails_and_transparent_official_closes_the_chain(shunt, provider, settings):
    """O caso do usuario: rota de familia casa (haiku), a cadeia da rota falha
    (o modelo 'free' responde 500 no fake), e o default esta nulo. O ultimo
    degrau da cadeia e o transparente ao host oficial com o model original."""
    del settings.providers["anthropic"]
    settings.default_model = None
    # A rota "haiku" -> ["free"] casa com "claude-haiku-4-5-20251001";
    # sem chave para "free" (openrouter tem fake-or-key), ele tenta e falha;
    # o degrau de baixo transparente oficial e o ultimo.

    # 502 e retry (MAX_ATTEMPTS=3): o candidato da rota esgota as tres
    # tentativas antes de a cadeia cair no transparente oficial.
    haiku_answer = {**OFFICIAL_ANSWER, "model": "claude-haiku-4-5-20251001"}
    provider.queue(
        Scripted(status=502, json_body={"error": "upstream down"}),
        Scripted(status=502, json_body={"error": "upstream down"}),
        Scripted(status=502, json_body={"error": "upstream down"}),
        Scripted(json_body=haiku_answer),
    )
    response = shunt.post(
        "/v1/messages",
        json={
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "oi"}],
        },
        headers={
            "x-api-key": "sk-ant-do-harness",
            "anthropic-version": "2023-06-01",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["content"][0]["text"] == "pelo host oficial"
    assert body["model"] == "claude-haiku-4-5-20251001"

    (call,) = [c for c in provider.calls if c.url.netloc == b"api.anthropic.com"]
    assert call.url.path == "/v1/messages"
    assert call.headers["x-api-key"] == "sk-ant-do-harness"
    assert provider.bodies[-1]["model"] == "claude-haiku-4-5-20251001"


def test_only_the_shunt_token_as_credential_returns_400(shunt, provider, settings):
    """T4 E2E: a unica credencial apresentada e o proprio token do Shunt
    (que o proxy validou -- credential_is_token=True). Sem chave de provedor
    configurada para sair, o transparente e pulado: 400 no envelope do
    chamador, nenhum upstream, e o token NUNCA viaja como credencial."""
    del settings.providers["anthropic"]
    settings.default_model = None

    # O fixture `shunt` injeta o token do Shunt em todo pedido (autouse em
    # tests/conftest.py); aqui a x-api-key E esse token.
    response = shunt.post(
        "/v1/messages",
        json={
            "model": "claude-sonnet-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "oi"}],
        },
        headers={
            "x-api-key": "token-da-suite-de-testes",
            "anthropic-version": "2023-06-01",
        },
    )
    assert response.status_code == 400, response.text
    body = response.json()
    assert body["type"] == "error"
    assert "shunt token" in body["error"]["message"]
    assert provider.calls == []
