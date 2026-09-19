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

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from app.stats.models import RequestEvent

logger = logging.getLogger("shunt")

MAX_QUEUE = 10_000
BATCH_SIZE = 200
INTERVAL = 1.0
# Prazo do dreno final. Um `aclose` que espera o banco indefinidamente trocaria
# um desligamento limpo por um processo pendurado.
DRAIN_TIMEOUT = 5.0


class Recorder:
    def __init__(
        self,
        engine: Engine | None,
        max_queue: int = MAX_QUEUE,
        batch_size: int = BATCH_SIZE,
        interval: float = INTERVAL,
    ) -> None:
        self._engine = engine
        self._batch_size = batch_size
        self._interval = interval
        self._queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task | None = None
        self.dropped = 0
        self.failures = 0
        self.commits = 0

    @property
    def enabled(self) -> bool:
        return self._engine is not None

    @property
    def queued(self) -> int:
        return self._queue.qsize()

    def record(self, event: dict) -> None:
        """Empilha o evento. Nunca espera, nunca levanta.

        A unica coisa que pode acontecer de ruim aqui e o evento ser descartado,
        e isso e deliberado: ver o contador de descarte subir e melhor do que
        ver a latencia da API subir.
        """
        if self._engine is None:
            return
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self.dropped += 1

    async def start(self) -> None:
        if self._engine is None or self._task is not None:
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
                session.add_all([RequestEvent(**event) for event in batch])
                session.commit()
            self.commits += 1
        except Exception as err:  # noqa: BLE001 - um lote perdido nao pode
            # derrubar a gravacao dos proximos, nem escapar para a aplicacao.
            self.failures += 1
            logger.warning(
                "stats: lote de %d eventos perdido (%s: %s)", len(batch), type(err).__name__, err
            )
