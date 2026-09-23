import asyncio
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Self

import httpx

from app.config.config import TIMEOUT_CONNECT, TIMEOUT_POOL, TIMEOUT_READ, TIMEOUT_WRITE
from app.config.settings import Settings

TIMEOUT = httpx.Timeout(
    connect=TIMEOUT_CONNECT, read=TIMEOUT_READ, write=TIMEOUT_WRITE, pool=TIMEOUT_POOL
)


class _Gate:
    """Semaforo de um provedor com limite, com tentativa que nao espera.

    O `asyncio.Semaphore` nao tem `try_acquire`: a unica forma de pegar um
    lugar e `await`, e o caminho normal do dispatcher precisa saber NA HORA se
    o provedor esta cheio para seguir ao proximo candidato. Aqui o lugar vago
    e passado direto para quem esta na fila (FIFO), entao quem esperou no
    ultimo recurso nao perde a vez para um pedido novo que chega no mesmo
    instante.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.in_use = 0
        self._waiters: deque[asyncio.Future[None]] = deque()

    def try_take(self) -> bool:
        # Sem checar a fila de espera: como `give_back` transfere o lugar
        # direto para quem espera, `in_use < limit` so acontece com a fila
        # vazia. Um pedido novo nunca fura a vez de quem ja esperava.
        if self.in_use >= self.limit:
            return False
        self.in_use += 1
        return True

    def enqueue(self) -> asyncio.Future[None]:
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        return waiter

    def abandon(self, waiter: asyncio.Future[None]) -> None:
        """Quem esperava desistiu (prazo, cancelamento, cliente foi embora).

        O lugar pode ter sido transferido para esta espera no mesmo tique em
        que ela desistiu: nesse caso ele ja e dela e tem de voltar ao pool,
        senao o slot fica preso para sempre.
        """
        if waiter.done():
            self.give_back()
        # Fora da fila, a espera nunca mais recebe lugar: `give_back` so
        # entrega a quem esta na fila.
        if waiter in self._waiters:
            self._waiters.remove(waiter)

    def give_back(self) -> None:
        # Toda espera na fila esta pendente: so esta funcao conclui uma (e a
        # tira da fila), e `abandon` tira da fila quem desistiu. O lugar so
        # passa direto a quem espera quando cabe no limite ATUAL: depois de um
        # `resize` para baixo `in_use` pode estar acima do limite, e entregar
        # o lugar manteria o excesso para sempre. Nesse caso ele some ate a
        # contagem voltar ao limite.
        if self._waiters and self.in_use <= self.limit:
            # O lugar passa direto para quem esperava: `in_use` nao muda.
            self._waiters.popleft().set_result(None)
            return
        self.in_use -= 1

    def resize(self, limit: int) -> None:
        """Limite novo, no lugar. Subiu: acorda quem espera ate encher a
        capacidade nova. Desceu: quem esta em voo continua, e `give_back`
        segura o handoff ate a contagem caber."""
        self.limit = limit
        while self._waiters and self.in_use < self.limit:
            self.in_use += 1
            self._waiters.popleft().set_result(None)

    def release_all(self) -> None:
        """Gate que deixou de existir (limite ou provedor removido): toda
        espera vira um lugar agora. `in_use` sobe junto para o `give_back`
        de cada um devolver simetricamente; ninguem mais le este gate."""
        while self._waiters:
            self.in_use += 1
            self._waiters.popleft().set_result(None)


class Slot:
    """Um lugar ocupado num provedor. `release()` e idempotente, para que todo
    caminho de saida possa liberar sem contar quem ja liberou."""

    def __init__(self, gate: _Gate | None) -> None:
        self._gate = gate
        self._held = True

    def release(self) -> None:
        if self._held and self._gate is not None:
            self._gate.give_back()
        self._held = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()


class SlotRequest:
    """Um lugar na FILA de um provedor, mantido entre varias esperas curtas.

    O streaming espera o slot em fatias de PING_INTERVAL para mandar
    keep-alive entre elas. Refazer `wait_slot` a cada fatia punha a espera no
    FIM da fila, e durante o `yield` do keep-alive ela nem estava na fila --
    um pedido que chegasse depois levava o slot. Aqui a espera entra na fila
    uma vez so e `wait()` so observa; sair do `with` sem o slot desiste dela.
    """

    def __init__(self, gate: _Gate | None) -> None:
        self._gate = gate
        self._waiter: asyncio.Future[None] | None = None
        # `_ready`: o lugar ja e deste pedido e ainda nao virou `Slot`.
        self._ready = gate is None or gate.try_take()
        if not self._ready and gate is not None:
            self._waiter = gate.enqueue()

    async def wait(self, timeout: float) -> Slot | None:
        """O slot, se chegou em ate `timeout` segundos; senao None, e a
        espera continua na fila. Um pedido entrega UM slot: depois dele,
        `wait()` devolve None -- dois `Slot` para um lugar liberariam dois."""
        if self._waiter is not None and not self._ready:
            await asyncio.wait({self._waiter}, timeout=max(timeout, 0.0))
            self._ready = self._waiter.done()
        if not self._ready:
            return None
        # Entregue: o `Slot` passa a ser o dono do lugar.
        self._waiter = None
        self._ready = False
        return Slot(self._gate)

    def cancel(self) -> None:
        if self._ready and self._gate is not None:
            # O lugar veio (no construtor) mas nunca virou `Slot`: devolve.
            self._gate.give_back()
            self._ready = False
        if self._waiter is not None and self._gate is not None:
            self._gate.abandon(self._waiter)
            self._waiter = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.cancel()


class UpstreamPool:
    """Clientes httpx e slots de concorrencia, por provedor.

    O pool nunca e recriado: `update` reconcilia os gates no lugar a cada
    edicao de configuracao, preservando `in_use` e as esperas em fila (ver
    `update`/`_reconcile_gates`). O limite e por processo: com varios workers
    do uvicorn, cada um conta os seus.
    """

    def __init__(
        self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._closed = False
        self._gates: dict[str, _Gate] = {
            name: _Gate(config.max_concurrency)
            for name, config in settings.providers.items()
            if config.max_concurrency is not None
        }

    async def update(self, settings: Settings) -> None:
        """Reconcilia o pool com um catalogo novo, sem trocar de identidade.

        Chamado por `apply_settings` a cada edicao (e pelo vigia de versao):
        recriar o pool cortava streams em voo e zerava `in_use`.
        """
        self._settings = settings
        self._reconcile_gates(settings)

    def _reconcile_gates(self, settings: Settings) -> None:
        for name, config in settings.providers.items():
            gate = self._gates.get(name)
            if config.max_concurrency is None:
                if gate is not None:
                    self._gates.pop(name).release_all()
            elif gate is None:
                self._gates[name] = _Gate(config.max_concurrency)
            elif gate.limit != config.max_concurrency:
                gate.resize(config.max_concurrency)
        for name in [name for name in self._gates if name not in settings.providers]:
            self._gates.pop(name).release_all()

    def get(self, provider: str) -> httpx.AsyncClient:
        if provider not in self._clients:
            self._clients[provider] = self._new_client(provider)
        return self._clients[provider]

    def _new_client(self, provider: str) -> httpx.AsyncClient:
        config = self._settings.providers[provider]
        return httpx.AsyncClient(
            base_url=config.base_url, timeout=TIMEOUT, transport=self._transport
        )

    @asynccontextmanager
    async def client(self, provider: str) -> AsyncIterator[httpx.AsyncClient]:
        """O cliente do provedor pelo tempo de um candidato.

        Com o pool aberto e o cliente compartilhado de sempre. Com o pool ja
        fechado -- um `apply_settings` trocou o pool enquanto este pedido ainda
        esperava um slot nele -- e um cliente so deste uso, fechado na saida:
        `get()` o criaria dentro de um pool que ninguem mais vai fechar.
        """
        if not self._closed:
            yield self.get(provider)
            return
        own = self._new_client(provider)
        try:
            yield own
        finally:
            await own.aclose()

    def limit(self, provider: str) -> int | None:
        gate = self._gates.get(provider)
        return gate.limit if gate is not None else None

    def in_use(self, provider: str) -> int:
        gate = self._gates.get(provider)
        return gate.in_use if gate is not None else 0

    def try_slot(self, provider: str) -> Slot | None:
        """Um lugar agora, ou None se o provedor esta cheio. Nunca espera."""
        gate = self._gates.get(provider)
        if gate is None:
            return Slot(None)
        return Slot(gate) if gate.try_take() else None

    def request_slot(self, provider: str) -> SlotRequest:
        """Entra na fila do provedor agora; `wait()` observa sem sair dela."""
        return SlotRequest(self._gates.get(provider))

    async def wait_slot(self, provider: str, timeout: float) -> Slot | None:
        """Espera um lugar por ate `timeout` segundos; None se nao vagou."""
        with self.request_slot(provider) as request:
            return await request.wait(timeout)

    async def aclose(self) -> None:
        self._closed = True
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()
