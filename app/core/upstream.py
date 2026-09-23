import httpx

from app.config.config import TIMEOUT_CONNECT, TIMEOUT_POOL, TIMEOUT_READ, TIMEOUT_WRITE
from app.config.settings import Settings

TIMEOUT = httpx.Timeout(
    connect=TIMEOUT_CONNECT, read=TIMEOUT_READ, write=TIMEOUT_WRITE, pool=TIMEOUT_POOL
)


class UpstreamPool:
    def __init__(
        self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._clients: dict[str, httpx.AsyncClient] = {}

    def get(self, provider: str) -> httpx.AsyncClient:
        if provider not in self._clients:
            config = self._settings.providers[provider]
            self._clients[provider] = httpx.AsyncClient(
                base_url=config.base_url, timeout=TIMEOUT, transport=self._transport
            )
        return self._clients[provider]

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()
