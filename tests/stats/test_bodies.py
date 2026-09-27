"""O texto da conversa: tabela separada, teto de tamanho, e desligavel."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.stats.models import RequestBody


def test_a_body_round_trips_with_its_original_size(make_engine):
    engine = make_engine()
    with Session(engine) as session:
        session.add(
            RequestBody(
                request_id="r1",
                prompt="leia a.txt",
                answer="pronto",
                prompt_bytes=10,
                answer_bytes=6,
                truncated=False,
            )
        )
        session.commit()
    with Session(engine) as session:
        stored = session.scalars(select(RequestBody)).one()
    assert stored.prompt == "leia a.txt"
    assert stored.answer == "pronto"
    assert stored.truncated is False


def test_the_size_before_the_cut_is_kept(make_engine):
    """A tela precisa dizer "mostrando 64 KB de 380 KB", e nao fingir que acabou."""
    engine = make_engine()
    with Session(engine) as session:
        session.add(
            RequestBody(
                request_id="grande",
                prompt="x" * 100,
                answer=None,
                prompt_bytes=380_000,
                answer_bytes=0,
                truncated=True,
            )
        )
        session.commit()
    with Session(engine) as session:
        stored = session.scalars(select(RequestBody)).one()
    assert len(stored.prompt) == 100
    assert stored.prompt_bytes == 380_000
    assert stored.truncated is True
