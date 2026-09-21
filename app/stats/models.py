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

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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

    # O que o provedor disse sobre cache. NULO quando ele nao disse nada -- que
    # e diferente de zero: medido, o Groq nao manda o campo e o OpenRouter manda
    # zero. Guardar os dois como 0 faria a tela afirmar "0% de cache" sobre quem
    # apenas nao informa.
    cached_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cache_write_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # De qual projeto veio, lido do corpo na ingestao (veja `app/core/project.py`).
    # Nulo e o caso comum e legitimo: cliente que nao e o Claude Code, ou versao
    # que mudou o rotulo do bloco de ambiente. A tela mostra "sem projeto" com a
    # contagem, para a perda aparecer em vez de sumir.
    project: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # A sessao do harness. Serve para herdar o projeto nas requisicoes seguintes
    # da mesma conversa, que nao repetem o bloco.
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

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
        Index("ix_request_events_project_started_at", "project", "started_at"),
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


class Analysis(Base):
    """A leitura de um periodo por um modelo, guardada com o dossie que a gerou.

    Tabela propria porque a analise custa token: pedir duas vezes a mesma janela
    e pagar duas vezes pela mesma resposta. A chave `fingerprint` e o periodo
    mais os filtros, que e o que define "a mesma janela".

    O dossie fica junto do texto de proposito. Uma recomendacao sem o numero que
    a sustenta e conselho generico, e o numero so pode ser conferido se ele for
    guardado -- o periodo ja passou, e a mesma consulta amanha da outro valor.
    """

    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    filters_json: Mapped[str] = mapped_column(Text, default="{}")
    dossier_json: Mapped[str] = mapped_column(Text, default="{}")

    requested_model: Mapped[str] = mapped_column(String(200))
    candidate_model: Mapped[str | None] = mapped_column(String(200), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    status: Mapped[int] = mapped_column(Integer, default=200)
    text: Mapped[str] = mapped_column(Text, default="")
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)


class Provider(Base):
    """Provedor upstream (local, openrouter, groq, anthropic, etc.)."""

    __tablename__ = "providers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    base_url: Mapped[str] = mapped_column(String(512), nullable=False)
    protocol: Mapped[str] = mapped_column(String(16), nullable=False)  # "openai" or "anthropic"
    api_key_env: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    models: Mapped[list[Model]] = relationship(back_populates="provider", lazy="selectin")


class Model(Base):
    """Modelo do catalogo: alias local -> provedor + nome upstream."""

    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alias: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    provider_id: Mapped[int] = mapped_column(ForeignKey("providers.id"), nullable=False)
    upstream_model: Mapped[str] = mapped_column(String(256), nullable=False)
    supports_tools: Mapped[bool] = mapped_column(Boolean, default=True)
    supports_streaming: Mapped[bool] = mapped_column(Boolean, default=True)
    supports_vision: Mapped[bool] = mapped_column(Boolean, default=False)
    context_window: Mapped[int] = mapped_column(Integer, nullable=False)
    max_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    provider: Mapped[Provider] = relationship(back_populates="models", lazy="selectin")

    __table_args__ = ()


class Route(Base):
    """Rota: padrao de substring -> lista ordenada de candidatos (Model.alias)."""

    __tablename__ = "routes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pattern: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    candidates: Mapped[list[RouteCandidate]] = relationship(back_populates="route", lazy="selectin", order_by="RouteCandidate.order_index")

    __table_args__ = ()


class RouteCandidate(Base):
    """Um candidato dentro de uma rota (ordem importa)."""

    __tablename__ = "route_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    route_id: Mapped[int] = mapped_column(ForeignKey("routes.id"), nullable=False)
    model_alias: Mapped[str] = mapped_column(String(128), nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    route: Mapped[Route] = relationship(back_populates="candidates", lazy="selectin")

    __table_args__ = ()


class ConfigVersion(Base):
    """Snapshot da configuracao completa para auditoria/rollback."""

    __tablename__ = "config_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    snapshot_json: Mapped[str] = mapped_column(Text, default="{}")

    __table_args__ = ()
