"""O catalogo desta instalacao, conferido como configuracao e nao como codigo.

A resolucao por familia e um casamento de substring percorrido na ORDEM da
lista de rotas. Um alias cujo nome contem o padrao de uma rota anterior nunca
chega ao proprio provedor: foi o que aconteceu com um alias chamado
`free-cloudflare`, engolido pela rota `free`. Este teste existe para que a
proxima escotilha nao repita o erro em silencio.
"""

from app.config.settings import load_settings
from app.core.resolver import resolve

settings = load_settings()


def test_every_escape_hatch_route_reaches_the_provider_it_names():
    default = settings.default_model
    for alias, model in settings.models.items():
        chain = resolve(alias, settings).chain
        # O primeiro degrau e o proprio alias; so um alias SEMEANTE ao default
        # pode ver o default encerrar a cadeia antes dele.
        first = chain[0]
        assert first.alias == alias or (
            default is not None and alias.startswith(default)
        ), f"pedir {alias!r} pelo nome caiu em {first.alias!r}: alguma rota anterior casa como substring"
        if first.alias == alias:
            assert first.provider == model.provider
        # O default declarado e o ultimo degrau de toda a cadeia.
        assert default is not None
        assert chain[-1].alias == default
        assert chain[-1].provider == settings.models[default].provider


def test_the_claude_code_slots_all_end_on_a_remote_fallback():
    """Os tres nomes que o harness pede nao podem morrer se o local cair.
    O ultimo degrau da cadeia e o `default_model` declarado."""
    default = settings.default_model
    assert default is not None
    for slot in ("haiku", "sonnet", "opus"):
        chain = resolve(slot, settings).chain
        assert len(chain) >= 2, slot
        assert {c.provider for c in chain} != {"local"}, slot
        assert chain[-1].alias == default, slot


def test_the_legacy_free_name_lands_on_the_remote_chain_head():
    """`model: "free"` nao e mais um alias nem uma escotilha: o nome antigo
    casa na familia `free` (substring) e sobe no primeiro remoto, em vez de
    forcar um unico provedor como a rota `("free", ["free"])` que existiu. Fixa
    o mapeamento que a renomeacao `free` -> `open-free` mudou para quem ja
    mandava o nome antigo no harness."""
    chain = resolve("free", settings).chain
    assert chain[0].alias == "open-free"
    assert chain[0].provider == "openrouter"
    default = settings.default_model
    assert default is not None
    assert chain[-1].alias == default
