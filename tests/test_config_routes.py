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
    for alias, model in settings.models.items():
        first = resolve(alias, settings).chain[0]
        assert first.alias == alias, (
            f"pedir {alias!r} pelo nome caiu em {first.alias!r}: "
            f"alguma rota anterior casa como substring"
        )
        assert first.provider == model.provider


def test_the_claude_code_slots_all_end_on_a_remote_fallback():
    """Os tres nomes que o harness pede nao podem morrer se o local cair."""
    for slot in ("haiku", "sonnet", "opus"):
        chain = resolve(slot, settings).chain
        assert len(chain) >= 2, slot
        assert {c.provider for c in chain} != {"local"}, slot
