import httpx

from app.config.settings import Settings

TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0)


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
