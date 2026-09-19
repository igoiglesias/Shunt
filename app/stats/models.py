"""Uma linha por requisicao, e todo grafico do painel sai de agregacao disso.

Por que gravar o evento cru em vez de contadores: a pergunta que o painel vai
fazer daqui a um mes nao esta escrita hoje. Contador responde so o que foi
previsto; a linha crua responde tambem "o que aconteceu naquela requisicao que
demorou 40 segundos". No volume de um proxy pessoal, a diferenca de espaco nao
paga a informacao perdida.

Nada aqui e NOT NULL alem do que existe em toda requisicao. Requisicao recusada
-- corpo malformado, modelo sem provedor, candidato sem credencial -- nao tem
provedor nem candidato, e e justamente a que mais se quer contar: obrigar a
coluna seria perder a linha ou inventar um valor.
"""

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RequestEvent(Base):
    __tablename__ = "request_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    # Qual superficie foi usada, e em que modo. `route` e o path do cliente, e
    # nao o do provedor: sao diferentes sempre que o proxy traduz.
    route: Mapped[str] = mapped_column(String(64))
    dialect: Mapped[str] = mapped_column(String(16))
    stream: Mapped[bool] = mapped_column(default=False)

    # O que o cliente pediu e que regra do catalogo casou.
    requested_model: Mapped[str] = mapped_column(String(128))
    rule: Mapped[str] = mapped_column(String(16))
    matched: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Quem de fato respondeu. Nulo quando ninguem respondeu.
    provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    candidate_model: Mapped[str | None] = mapped_column(String(128), nullable=True)

    status: Mapped[int] = mapped_column(Integer)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)

    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)

    # `ttft_ms` so existe em streaming, onde ha um primeiro evento a cronometrar.
    ttft_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)

    # A cadeia: quem falhou antes de alguem responder. Fica como JSON porque a
    # pergunta que se faz dela e "quantas vezes o candidato X foi pulado", que
    # se responde lendo a lista inteira, e nunca "junte com outra tabela".
    attempts: Mapped[list] = mapped_column(JSON, default=list)
    fell_back: Mapped[bool] = mapped_column(default=False)

    tools_offered: Mapped[list] = mapped_column(JSON, default=list)
    tools_called: Mapped[list] = mapped_column(JSON, default=list)
    thinking_blocks: Mapped[int] = mapped_column(Integer, default=0)

    # Os tres eixos de toda consulta do painel: a janela de tempo, e o
    # agrupamento por provedor ou por modelo dentro dela. Indice composto e nao
    # dois indices separados porque o filtro de tempo entra em todas elas.
    __table_args__ = (
        Index("ix_request_events_started_at", "started_at"),
        Index("ix_request_events_provider_started_at", "provider", "started_at"),
        Index("ix_request_events_candidate_started_at", "candidate_model", "started_at"),
    )


class RequestBody(Base):
    """O texto que entrou e o texto que saiu, numa tabela SEPARADA.

    Separada de proposito: a listagem e a busca da tela de auditoria leem
    `request_events` centenas de linhas por vez, e um `SELECT *` que arrastasse
    megabytes de conversa junto tornaria a busca lenta para responder uma
    pergunta que ela nem faz. O texto so e lido quando alguem abre UMA
    requisicao.

    Guardar conversa e escolha do operador, e vem desligada por variavel de
    ambiente: e o dado mais sensivel que passa por este proxy.
    """

    __tablename__ = "request_bodies"

    request_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("request_events.request_id"), primary_key=True
    )
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Tamanho ANTES do corte, para que a tela possa dizer "mostrando 64 KB de
    # 380 KB" em vez de fingir que a conversa acabou ali.
    # O corpo CRU da requisicao, redigido e cortado. O texto responde "o que
    # foi dito"; isto responde "como reproduzir" -- os parametros, as
    # ferramentas declaradas, o que o cliente mandou de fato.
    request_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_bytes: Mapped[int] = mapped_column(Integer, default=0)
    answer_bytes: Mapped[int] = mapped_column(Integer, default=0)
    truncated: Mapped[bool] = mapped_column(default=False)
