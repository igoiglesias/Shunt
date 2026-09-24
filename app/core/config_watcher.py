"""Vigia de versao do catalogo, um por worker.

`make prod` sobe um processo por nucleo e cada um guarda o catalogo em
memoria (`app.state.settings`). Uma edicao no admin so chegava ao worker que
a atendeu; os outros seguiam com o catalogo velho ate o restart. O vigia
consulta `max(id)` de `config_versions` num intervalo curto -- toda escrita
do admin insere uma linha (`create_config_version`, em
`app/routers/admin_config.py`; o seed nao insere, entao banco novo comeca em
NULL) -- e recarrega o catalogo quando o numero muda. Medido antes do design:
a consulta custa 0,11 ms (mediana) e a recarga completa 11 ms num sqlite
local. O worker que atendeu a edicao recarrega uma vez a mais no ciclo
seguinte; custo aceito para nao acoplar o admin ao vigia.

Falha em qualquer passo mantem o catalogo atual e tenta de novo no proximo
ciclo: catalogo velho ainda atende; catalogo vazio ou parcial nao. Por isso
`seen` so avanca depois de `apply` terminar.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from app.config.config import CONFIG_POLL_SECONDS
from app.config.settings import Settings, load_settings_from_db
from app.stats.engine import reachable

logger = logging.getLogger("shunt")

VERSION_SQL = text("SELECT max(id) FROM config_versions")


class ConfigWatcher:
    def __init__(
        self,
        engine_of: Callable[[], Engine | None],
        apply: Callable[[Settings], Awaitable[None]],
        url: str,
        interval: float = CONFIG_POLL_SECONDS,
    ) -> None:
        # `engine_of` e chamado a cada ciclo: o gravador reabre o banco quando
        # ele volta, e a engine nova tem de ser a usada.
        self._engine_of = engine_of
        self._apply = apply
        self._url = url
        self._interval = interval
        self._task: asyncio.Task[None] | None = None
        self.seen: int | None = None
        self.reloads = 0
        self.failures = 0

    async def start(self) -> None:
        if self._task is not None:
            return
        # Versao inicial junto com o boot: o primeiro ciclo nao recarrega a
        # toa. Sem ela (banco lento no boot) o primeiro ciclo recarrega uma
        # vez, o que e o custo certo.
        engine = self._engine_of()
        if engine is not None:
            try:
                self.seen = await asyncio.to_thread(self._read_version, engine)
            except Exception as err:  # noqa: BLE001 - ver docstring do modulo.
                logger.warning(
                    "config: versao inicial nao lida (%s: %s)", type(err).__name__, err
                )
        self._task = asyncio.create_task(self._run())

    async def aclose(self) -> None:
        if self._task is None:
            return
        task, self._task = self._task, None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            await self.check_once()

    async def check_once(self) -> bool:
        """Um ciclo: recarregou o catalogo? Nunca levanta (cancelamento passa:
        `CancelledError` nao e `Exception`)."""
        try:
            engine = self._engine_of()
            if engine is None:
                return False
            version = await asyncio.to_thread(self._read_version, engine)
            if version == self.seen:
                return False
            settings = await asyncio.to_thread(self._load, engine)
            await self._apply(settings)
        except Exception as err:  # noqa: BLE001 - ver docstring do modulo.
            self.failures += 1
            logger.warning(
                "config: vigia falhou, catalogo mantido (%s: %s)", type(err).__name__, err
            )
            return False
        self.seen = version
        self.reloads += 1
        logger.info("config: catalogo recarregado (versao %s)", version)
        return True

    def _read_version(self, engine: Engine) -> int | None:
        # O driver do libsql trava o processo num host inalcancavel (ver
        # `REACH_TIMEOUT` em app/config/config.py); o socket com relogio
        # decide antes. Banco em arquivo responde True sem tocar a rede.
        if not reachable(self._url):
            raise ConnectionError("banco inalcancavel neste ciclo")
        with engine.connect() as connection:
            return connection.execute(VERSION_SQL).scalar()

    def _load(self, engine: Engine) -> Settings:
        with Session(engine) as session:
            return load_settings_from_db(session)
