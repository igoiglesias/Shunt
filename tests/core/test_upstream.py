from app.config.settings import ProviderConfig, Settings
from app.core.upstream import UpstreamPool

SETTINGS = Settings(
    providers={
        "openrouter": ProviderConfig(base_url="https://openrouter.ai/api/v1",
                                     protocol="openai", api_key_env=None),
        "local": ProviderConfig(base_url="http://localhost:8080/v1",
                                protocol="openai", api_key_env=None),
    },
    models={}, routes=[], default_model=None,
)


async def test_same_provider_reuses_the_same_client():
    pool = UpstreamPool(SETTINGS)
    try:
        assert pool.get("openrouter") is pool.get("openrouter")
        assert pool.get("openrouter") is not pool.get("local")
    finally:
        await pool.aclose()


async def test_client_carries_the_provider_base_url_and_split_timeouts():
    pool = UpstreamPool(SETTINGS)
    try:
        client = pool.get("local")
        assert str(client.base_url).rstrip("/") == "http://localhost:8080/v1"
        assert client.timeout.connect == 10.0
        assert client.timeout.read == 60.0
    finally:
        await pool.aclose()


async def test_aclose_closes_the_clients_and_empties_the_pool():
    pool = UpstreamPool(SETTINGS)
    first = pool.get("openrouter")
    await pool.aclose()

    assert first.is_closed
    # A fresh get() after aclose must build a brand-new client: proof the
    # internal map was actually emptied, not just left with closed entries.
    second = pool.get("openrouter")
    assert second is not first
    await pool.aclose()
