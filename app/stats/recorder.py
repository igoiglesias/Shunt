"""A fila entre a requisicao e o banco, e o worker que a esvazia.

Este modulo existe por uma restricao unica: **o banco nao pode custar latencia
a API**. Tudo aqui decorre dela.

- `record()` e SINCRONO e nao levanta. Ele empilha e volta. Nao ha `await` de
  rede, nao ha sessao de ORM, nao ha `commit` entre o cliente e a resposta.
- A fila tem teto. Fila cheia significa banco lento ou fora do ar, e a resposta
  certa nessa hora e perder estatistica -- nunca segurar a requisicao que o
  usuario esta esperando. O descarte e contado e aparece no proprio painel,
  para que a perda seja visivel em vez de silenciosa.
- O worker grava em LOTE: um round-trip ao banco por lote, e nao por
  requisicao. Ele acorda por tempo ou por volume, o que vier primeiro.
- A gravacao roda em thread separada (`asyncio.to_thread`), porque o dialeto do
  libsql e sincrono. Bloquear ali nao custa nada: e fora do caminho quente.
- Toda excecao de gravacao e capturada e contada. Banco fora do ar deixa o
  painel velho; o proxy nao percebe.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime

from sqlalchemy import Engine, update
from sqlalchemy.orm import Session

from app.config.config import (
    BATCH_SIZE,
    DRAIN_TIMEOUT,
    INTERVAL,
    MAX_QUEUE,
    RECONNECT_SECONDS,
    SUBSCRIBER_QUEUE,
)
from app.stats.models import ApiToken, RequestBody, RequestEvent

logger = logging.getLogger("shunt")


class Recorder:
    def __init__(
        self,
        engine: Engine | None,
        max_queue: int = MAX_QUEUE,
        batch_size: int = BATCH_SIZE,
        interval: float = INTERVAL,
        reconnect: Callable[[], Engine | None] | None = None,
        reconnect_seconds: float = RECONNECT_SECONDS,
    ) -> None:
        self._engine = engine
        # Como reabrir o banco, e quando tentar de novo. Sem `reconnect` o
        # gravador se comporta como antes: o que veio no boot e o que ha.
        self._reconnect = reconnect
        self._reconnect_seconds = reconnect_seconds
        self._next_attempt = 0.0
        self._batch_size = batch_size
        self._interval = interval
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task | None = None
        self._subscribers: list[asyncio.Queue] = []
        self.dropped = 0
        self.failures = 0
        self.commits = 0
        self.reconnects = 0

    @property
    def enabled(self) -> bool:
        return self._engine is not None

    @property
    def engine(self) -> Engine | None:
        return self._engine

    @property
    def queued(self) -> int:
        return self._queue.qsize()

    def subscribe(self) -> asyncio.Queue:
        """Uma fila por navegador aberto no painel.

        O SSE do painel escuta AQUI, e nao o banco: um navegador aberto nao
        pode virar consulta repetida. Cada assinante tem fila propria e
        pequena, porque um leitor lento nao pode segurar o proxy -- quando
        enche, o evento passa direto e aquele navegador perde a linha ao vivo,
        que a proxima atualizacao do resumo corrige.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def _publish(self, event: dict) -> None:
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    @property
    def configured(self) -> bool:
        """Existe um banco declarado, mesmo que agora ele nao abra?"""
        return self._engine is not None or self._reconnect is not None

    def _try_reconnect(self) -> None:
        """Uma tentativa por intervalo, e so no worker de fundo.

        Nunca no caminho da requisicao: reabrir banco fala com a rede.
        """
        if self._engine is not None or self._reconnect is None:
            return
        now = time.monotonic()
        if now < self._next_attempt:
            return
        self._next_attempt = now + self._reconnect_seconds
        try:
            engine = self._reconnect()
        except Exception as err:  # noqa: BLE001 - tentar de novo depois e a
            # resposta certa para qualquer falha aqui.
            logger.warning("stats: reconexao falhou (%s: %s)", type(err).__name__, err)
            return
        if engine is not None:
            self._engine = engine
            self.reconnects += 1
            logger.info("stats: banco reaberto depois de %d tentativa(s)", self.reconnects)

    def _enqueue(self, event: dict) -> None:
        """O lado que escreve no banco. Nunca espera, nunca levanta.

        A unica coisa que pode acontecer de ruim aqui e o evento ser descartado,
        e isso e deliberado: ver o contador de descarte subir e melhor do que
        ver a latencia da API subir.
        """
        if self._engine is None and self._reconnect is None:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped += 1

    def record(self, event: dict) -> None:
        """Anuncia o evento ao painel e o empilha para o banco.

        O barramento do painel e servido mesmo sem banco: quem so quer olhar o
        proxy correndo nao precisa configurar Turso nenhum.
        """
        self._publish(event)
        self._enqueue(event)

    def store(self, event: dict) -> None:
        """Grava no banco SEM anunciar no barramento do painel.

        E a escrita do repasse de rotas desconhecidas: trafego de passagem nao
        e requisicao de modelo, e publicar o faria aparecer no SSE e em
        `recent` como se o proxy tivesse chamado um modelo. O caminho de
        enfileirar e o mesmo de `record` -- nunca espera, nunca levanta.
        """
        self._enqueue(event)

    def touch_token(self, token_id: int) -> None:
        """`last_used_at` do token vai pela fila: nunca uma escrita no caminho da requisicao."""
        self._enqueue({"kind": "token_used", "token_id": token_id, "at": datetime.now(UTC)})

    async def start(self) -> None:
        if not self.configured or self._task is not None:
            return
        self._task = asyncio.create_task(self._run())

    async def aclose(self) -> None:
        """Para o worker depois de tentar gravar o que ainda esta na fila."""
        if self._task is None:
            return
        task, self._task = self._task, None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        try:
            await asyncio.wait_for(self._drain(), timeout=DRAIN_TIMEOUT)
        except TimeoutError:
            logger.warning("stats: dreno final expirou com %d eventos na fila", self.queued)

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            if self._engine is None:
                await asyncio.to_thread(self._try_reconnect)
            await self._drain()

    async def _drain(self) -> None:
        while not self._queue.empty():
            batch: list[dict] = []
            while len(batch) < self._batch_size and not self._queue.empty():
                batch.append(self._queue.get_nowait())
            await asyncio.to_thread(self._write, batch)

    def _write(self, batch: list[dict]) -> None:
        """Um `commit` por lote. Falha nao sobe: ela e contada e o worker segue."""
        if self._engine is None:
            return
        try:
            with Session(self._engine) as session:
                for event in batch:
                    # Evento interno: atualiza o ultimo uso do token sem passar
                    # pela tabela de eventos -- o painel de uso da API nao precisa
                    # ver cada toque, e um UPDATE nao custa linha nova.
                    if event.get("kind") == "token_used":
                        session.execute(
                            update(ApiToken)
                            .where(ApiToken.id == event["token_id"])
                            .values(last_used_at=event["at"])
                        )
                        continue
                    # O texto da conversa viaja junto do evento e vai para a
                    # OUTRA tabela: a busca da auditoria le centenas de linhas
                    # por vez e nao pode arrastar megabytes atras.
                    body = event.pop("body", None)
                    session.add(RequestEvent(**event))
                    if body:
                        session.add(RequestBody(request_id=event["request_id"], **body))
                session.commit()
            self.commits += 1
        except Exception as err:  # noqa: BLE001 - um lote perdido nao pode
            # derrubar a gravacao dos proximos, nem escapar para a aplicacao.
            self.failures += 1
            logger.warning(
                "stats: lote de %d eventos perdido (%s: %s)", len(batch), type(err).__name__, err
            )
