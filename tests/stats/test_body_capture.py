"""A extracao do texto da conversa: os dois dialetos, o teto e a redacao."""

import pytest

from app.stats import bodies


@pytest.fixture(autouse=True)
def ligado(monkeypatch):
    monkeypatch.setenv("SHUNT_STORE_BODIES", "1")


def test_it_is_off_unless_asked_for(monkeypatch):
    """Um proxy que comeca gravando o que o operador digita e uma surpresa ruim."""
    monkeypatch.delenv("SHUNT_STORE_BODIES", raising=False)
    assert bodies.enabled() is False
    assert bodies.capture("oi", "ola") is None
    monkeypatch.setenv("SHUNT_STORE_BODIES", "0")
    assert bodies.enabled() is False
    monkeypatch.setenv("SHUNT_STORE_BODIES", "sim")
    assert bodies.enabled() is True


def test_the_anthropic_prompt_becomes_readable_text():
    text = bodies.prompt_text(
        {
            "system": [{"type": "text", "text": "seja breve"}],
            "messages": [
                {"role": "user", "content": "leia a.txt"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "vou ler"},
                        {"type": "tool_use", "name": "read", "input": {"path": "a.txt"}},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "content": "conteudo do arquivo"},
                        {"type": "image"},
                    ],
                },
            ],
        }
    )
    assert "system: seja breve" in text
    assert "user: leia a.txt" in text
    assert "[pensando] vou ler" in text
    assert '[ferramenta read] {"path": "a.txt"}' in text
    assert "[resultado de ferramenta] conteudo do arquivo" in text
    assert "[imagem]" in text


def test_the_openai_prompt_becomes_readable_text():
    text = bodies.prompt_text(
        {
            "messages": [
                {"role": "system", "content": "seja breve"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"function": {"name": "read", "arguments": '{"path": "a.txt"}'}}
                    ],
                },
            ]
        }
    )
    assert "system: seja breve" in text
    assert "[ferramenta read]" in text


@pytest.mark.parametrize(
    ("body", "esperado"),
    [
        ({"prompt": "complete isto"}, "complete isto"),
        ({"input": "texto para embutir"}, "texto para embutir"),
        ({"messages": ["isso nao e mensagem"]}, ""),
        ({}, ""),
    ],
)
def test_routes_without_messages_still_have_text(body, esperado):
    assert bodies.prompt_text(body) == esperado


def test_the_anthropic_answer_becomes_readable_text():
    text = bodies.answer_text(
        {
            "content": [
                {"type": "thinking", "thinking": "pensei"},
                {"type": "text", "text": "pronto"},
                {"type": "tool_use", "name": "write", "input": {"path": "b.txt"}},
            ]
        }
    )
    assert "[pensando] pensei" in text
    assert "pronto" in text
    assert "[ferramenta write]" in text


def test_the_openai_answer_becomes_readable_text():
    text = bodies.answer_text(
        {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "pensei",
                        "content": "pronto",
                        "tool_calls": [{"function": {"name": "write", "arguments": "{}"}}],
                    }
                },
                {"text": " completado"},
                "isso nao e escolha",
            ]
        }
    )
    assert "[pensando] pensei" in text
    assert "pronto" in text
    assert "[ferramenta write]" in text
    assert "completado" in text


@pytest.mark.parametrize(
    "segredo",
    [
        "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAA",
        "ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "xoxb-1111111111-abcdefghij",
        "AKIAIOSFODNN7EXAMPLE",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijk",
        "OPENROUTER_API_KEY=sk-or-v1-segredo",
    ],
)
def test_what_looks_like_a_credential_is_redacted(segredo):
    """Um agente que cola um `.env` num prompt nao pode gravar a chave para sempre."""
    limpo = bodies.redact(f"antes {segredo} depois")
    assert segredo not in limpo
    assert "[redigido]" in limpo
    assert "antes" in limpo and "depois" in limpo


def test_ordinary_text_survives_the_redaction():
    texto = "leia o arquivo sk.txt e rode make check"
    assert bodies.redact(texto) == texto


def test_the_cut_keeps_the_original_size(monkeypatch):
    monkeypatch.setenv("SHUNT_BODY_LIMIT", "300")
    captured = bodies.capture("x" * 5000, "y" * 10)
    assert len(captured["prompt"]) == 300
    assert captured["prompt_bytes"] == 5000
    assert captured["answer_bytes"] == 10
    assert captured["truncated"] is True


def test_a_limit_that_does_not_parse_falls_back(monkeypatch):
    monkeypatch.setenv("SHUNT_BODY_LIMIT", "muito")
    assert bodies.limit() == bodies.DEFAULT_LIMIT
    monkeypatch.setenv("SHUNT_BODY_LIMIT", "10")
    assert bodies.limit() == 200, "um teto minusculo esconderia toda conversa"


def test_nothing_to_store_stores_nothing():
    assert bodies.capture("", "") is None
    assert bodies.capture("oi", "")["answer"] == ""


@pytest.mark.parametrize(
    ("bloco", "esperado"),
    [
        ("texto solto", "texto solto"),
        (["isso nao e bloco"], ""),
        (7, ""),
        ({"type": "document"}, "[document]"),
        ({}, "[bloco]"),
    ],
)
def test_a_block_shape_never_seen_before_still_reads(bloco, esperado):
    """Provedor inventa tipo de bloco; a conversa nao pode virar excecao."""
    assert bodies._block_text(bloco) == esperado


def test_the_capture_redacts_before_storing():
    """Mutante B2: `capture` sem `redact` sobreviveu.

    Os testes de redacao chamavam `redact` direto; nenhum provava que o caminho
    que GRAVA passa por ela -- que e onde a chave chegaria ao banco.
    """
    capturado = bodies.capture(
        "user: minha chave e sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAA",
        "assistant: anotei ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    )
    assert "sk-ant-api03" not in capturado["prompt"]
    assert "ghp_" not in capturado["answer"]
    assert capturado["prompt"].count("[redigido]") == 1
    assert capturado["answer"].count("[redigido]") == 1
