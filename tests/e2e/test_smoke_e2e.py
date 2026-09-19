"""Fumaca do E2E: o aplicativo inteiro, de ponta a ponta, contra um provedor falso.

Nada aqui usa dublê de tradutor ou de dispatcher. A requisicao entra pela rota
real, atravessa resolucao, traducao e transporte, e volta traduzida.
"""


def test_request_reaches_the_fake_provider_and_comes_back_translated(shunt, provider):
    provider.queue(
        json_body={
            "id": "chatcmpl-1",
            "model": "vendor/free",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "pronto"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }
    )
    response = shunt.post(
        "/v1/messages",
        json={
            "model": "claude-haiku-4-5",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "oi"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["content"] == [{"type": "text", "text": "pronto"}]
    assert provider.bodies[0]["model"] == "vendor/free"
    assert provider.calls[0].url.path == "/v1/chat/completions"
