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


class _Held:
    """Um cliente httpx e quantos candidatos o usam agora.

    `retired`: saiu do dicionario ativo num `update` (base_url mudou ou o
    provedor saiu) e fecha quando o ultimo uso termina -- nunca embaixo de um
    stream que ainda le por ele.
    """

    def __init__(self, client: httpx.AsyncClient, base_url: str) -> None:
        self.client = client
        # O base_url com que ESTE cliente foi criado -- nao o do `Settings`
        # de quando `update` rodou. Dois `update()` concorrentes cada um
        # captura o `Settings` "antigo" no seu proprio inicio; com pools
        # compartilhados, esse "antigo" pode ja estar defasado quando a
        # reconciliacao roda (ver `_reconcile_clients`). Comparar contra o
        # base_url gravado no proprio Held e o unico jeito de decidir certo
        # sem depender de quem correu primeiro.
        self.base_url = base_url
        self.uses = 0
        self.retired = False


class UpstreamPool:
    """Clientes httpx e slots de concorrencia, por provedor.

    O pool nunca e recriado: `update` reconcilia os gates e os clientes no
    lugar a cada edicao de configuracao, preservando `in_use` e as esperas em
    fila (ver `update`/`_reconcile_gates`/`_reconcile_clients`), e aposentando
    clientes cujo `base_url` mudou ou cujo provedor saiu do catalogo. O limite
    e por processo: com varios workers do uvicorn, cada um conta os seus.
    """

    def __init__(
        self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._clients: dict[str, _Held] = {}
        self._retired: list[_Held] = []
        self._closed = False
        self._gates: dict[str, _Gate] = {
            name: _Gate(config.max_concurrency)
            for name, config in settings.providers.items()
            if config.max_concurrency is not None
        }

    async def update(self, settings: Settings) -> None:
        """Reconcilia o pool com um catalogo novo, sem trocar de identidade.

        Chamado por `apply_settings` a cada edicao (e pelo vigia de versao) --
        de dois lugares do mesmo worker, entao dois `update()` podem correr
        concorrentes no mesmo pool. Recriar o pool cortava streams em voo e
        zerava `in_use`.
        """
        self._settings = settings
        self._reconcile_gates(settings)
        await self._reconcile_clients(settings)

    async def _reconcile_clients(self, new: Settings) -> None:
        """Aposenta clientes cujo provedor saiu ou cujo `base_url` mudou.

        Duas fases, sem nenhum `await` na primeira: com dois `update()`
        concorrentes, cada um so cede o loop dentro do `aclose` (fase 2) --
        nunca no meio de decidir/mutar `_clients` (fase 1). Assim a fase 1 de
        uma chamada roda do inicio ao fim sem outra intercalar no meio dela.

        A decisao compara contra `held.base_url` (gravado no proprio Held
        quando foi criado), nao contra um `Settings` "antigo" capturado no
        inicio deste `update()`: esse "antigo" e uma leitura de um atributo
        compartilhado (`self._settings`) e pode ja estar defasado se outro
        `update()` correu por cima antes deste comecar. O guard `is not held`
        cobre o caso em que, mesmo assim, o nome já aponta para outro Held
        quando a fase 2 roda (por exemplo um `client()` concorrente recriou o
        provedor enquanto este `update()` esperava a vez).
        """
        to_close: list[_Held] = []
        for name, held in list(self._clients.items()):
            after = new.providers.get(name)
            if after is not None and held.base_url == after.base_url:
                continue
            if self._clients.get(name) is not held:
                continue
            del self._clients[name]
            held.retired = True
            if held.uses == 0:
                to_close.append(held)
            else:
                self._retired.append(held)
        for held in to_close:
            await held.client.aclose()

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
        """O cliente compartilhado do provedor, SEM contagem de uso: quem
        chama `get()` pode ver o cliente fechar num `update` que troque o
        `base_url`. O caminho de requisicao usa `client()`; `get()` fica
        para os testes."""
        return self._held(provider).client

    def _held(self, provider: str) -> _Held:
        if provider not in self._clients:
            base_url = self._settings.providers[provider].base_url
            self._clients[provider] = _Held(self._new_client(provider), base_url)
        return self._clients[provider]

    def _new_client(self, provider: str) -> httpx.AsyncClient:
        config = self._settings.providers[provider]
        return httpx.AsyncClient(
            base_url=config.base_url, timeout=TIMEOUT, transport=self._transport
        )

    @asynccontextmanager
    async def client(self, provider: str) -> AsyncIterator[httpx.AsyncClient]:
        """O cliente do provedor pelo tempo de um candidato, com uso contado.

        Com o pool aberto e o cliente compartilhado; se um `update` o
        aposentar no meio do uso, ele so fecha quando este uso (e os outros
        em voo) terminarem. Com o pool ja fechado -- o shutdown chegou com
        este pedido ainda esperando um slot -- e um cliente so deste uso,
        fechado na saida: `get()` o criaria dentro de um pool que ninguem
        mais vai fechar.
        """
        if self._closed:
            own = self._new_client(provider)
            try:
                yield own
            finally:
                await own.aclose()
            return
        held = self._held(provider)
        held.uses += 1
        try:
            yield held.client
        finally:
            held.uses -= 1
            if held.retired and held.uses == 0:
                await self._close_retired(held)

    async def _close_retired(self, held: _Held) -> None:
        if held in self._retired:
            self._retired.remove(held)
        if not held.client.is_closed:
            await held.client.aclose()

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
        for held in (*self._clients.values(), *self._retired):
            await held.client.aclose()
        self._clients.clear()
        self._retired.clear()
