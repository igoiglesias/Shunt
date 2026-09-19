"""O stream contra o mundo real: fragmentacao adversaria e lixo no meio.

Nenhum destes casos cabe num teste de unidade do tradutor: todos dependem de
como os bytes chegam pelo transporte, e nao de como o evento e traduzido.
"""

import json

from tests.e2e.fake_provider import Scripted, sse_chunks

ASK = {
    "model": "claude-opus-4-5",
    "max_tokens": 64,
    "stream": True,
    "messages": [{"role": "user", "content": "oi"}],
}


def texts_of(body):
    out = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = json.loads(line.removeprefix("data: "))
        if payload.get("delta", {}).get("type") == "text_delta":
            out.append(payload["delta"]["text"])
    return "".join(out)


def collect(shunt, payload=None):
    with shunt.stream("POST", "/v1/messages", json=payload or ASK) as response:
        return "".join(response.iter_text())


def test_keepalive_comments_before_the_first_event_do_not_trigger_fallback(shunt, provider):
    provider.queue(
        Scripted(
            sse=[
                b": OPENROUTER PROCESSING\n\n",
                b": OPENROUTER PROCESSING\n\n",
                b'data: {"choices": [{"delta": {"content": "chegou"}}]}\n\n',
                b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n',
                b"data: [DONE]\n\n",
            ]
        )
    )
    body = collect(shunt)
    assert len(provider.calls) == 1
    assert texts_of(body) == "chegou"


def test_one_byte_at_a_time_produces_the_same_events(shunt, provider):
    whole = b"".join(
        sse_chunks(
            {"choices": [{"delta": {"content": "um"}}]},
            {"choices": [{"delta": {"content": " dois"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        )
    )
    provider.queue(Scripted(sse=[whole[i : i + 1] for i in range(len(whole))]))
    assert texts_of(collect(shunt)) == "um dois"


def test_multibyte_character_split_across_chunks_is_not_corrupted(shunt, provider):
    payload = json.dumps({"choices": [{"delta": {"content": "✅ feito"}}]}, ensure_ascii=False)
    raw = f"data: {payload}\n\n".encode()
    cut = raw.index(b"\xe2") + 2
    provider.queue(
        Scripted(
            sse=[
                raw[:cut],
                raw[cut:],
                b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n',
                b"data: [DONE]\n\n",
            ]
        )
    )
    assert texts_of(collect(shunt)) == "✅ feito"


def test_malformed_json_inside_the_stream_is_skipped_not_fatal(shunt, provider):
    provider.queue(
        Scripted(
            sse=[
                b'data: {"choices": [{"delta": {"content": "antes"}}]}\n\n',
                b"data: {isso nao e json}\n\n",
                b'data: {"choices": [{"delta": {"content": " depois"}}]}\n\n',
                b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n',
                b"data: [DONE]\n\n",
            ]
        )
    )
    body = collect(shunt)
    assert texts_of(body) == "antes depois"
    assert body.rstrip().endswith("}")
    assert "message_stop" in body


def test_stream_that_ends_without_done_still_closes_the_message(shunt, provider):
    provider.queue(Scripted(sse=[b'data: {"choices": [{"delta": {"content": "cortado"}}]}\n\n']))
    body = collect(shunt)
    assert "event: message_stop" in body
    assert "event: content_block_stop" in body


def test_stream_with_nothing_but_done_hands_the_turn_to_the_next_candidate(shunt, provider):
    """200 que fecha sem um evento sequer e falha do candidato, nao resposta vazia.

    Medido: o plano previa que esta fosse a unica resposta e que o proxy a
    fechasse limpa. A Task 19 decidiu o contrario, e a decisao e melhor -- um
    provedor que abre o stream e nao diz nada ainda esta antes do primeiro
    evento, que e exatamente a janela em que trocar de candidato nao mente para
    o cliente. O que este teste prende e que a troca acontece e que o cliente
    recebe o conteudo do segundo, e nao um silencio.
    """
    provider.queue(
        Scripted(sse=[b"data: [DONE]\n\n"]),
        Scripted(
            sse=sse_chunks(
                {"choices": [{"delta": {"content": "do segundo"}}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            )
        ),
    )
    body = collect(shunt)
    assert len(provider.calls) == 2
    assert texts_of(body) == "do segundo"
    assert "event: message_stop" in body


def test_an_empty_stream_with_no_candidate_left_answers_an_error_not_a_hang(shunt, provider, settings):
    settings.routes = [("opus", ["free"])]
    provider.queue(Scripted(sse=[b"data: [DONE]\n\n"]))
    body = collect(shunt)
    assert "event: error" in body
    assert "stream ended before the first valid event" in body


def test_client_disconnect_releases_the_upstream_response(shunt, provider):
    provider.queue(
        Scripted(
            sse=sse_chunks(*[{"choices": [{"delta": {"content": f"{i}"}}]} for i in range(50)]),
            chunk_delay=0.01,
        )
    )
    with shunt.stream("POST", "/v1/messages", json=ASK) as response:
        next(response.iter_text())  # le um pedaco e abandona
    assert len(provider.calls) == 1
