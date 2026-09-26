from dataclasses import FrozenInstanceError

from app.core.official_hosts import OFFICIAL_HOSTS, OfficialHost


def test_official_hosts_carry_base_url_and_protocol():
    # Medido (httpx 0.28): o path do provedor ANTHROPIC ja carrega o /v1
    # (`PATHS: ("anthropic", "messages") -> "/v1/messages"`); com um base_url
    # terminado em /v1 o join duplicava o prefixo (`/v1/v1/messages`). O base
    # do Anthropic fica SEM /v1; o OpenAI ja tem (os paths dele nao levam).
    assert OFFICIAL_HOSTS["anthropic"] == OfficialHost(
        "https://api.anthropic.com", "anthropic"
    )
    assert OFFICIAL_HOSTS["openai"] == OfficialHost("https://api.openai.com/v1", "openai")


def test_only_the_official_targets_are_embedded():
    assert set(OFFICIAL_HOSTS) == {"anthropic", "openai"}
    # "chatgpt" pertence ao plano 2026-09-23 (Tasks 2.x/3.x): fora de escopo.
    assert "chatgpt" not in OFFICIAL_HOSTS
    assert "openrouter" not in OFFICIAL_HOSTS


def test_entries_are_frozen():
    host = OFFICIAL_HOSTS["anthropic"]
    try:
        host.base_url = "https://outro.example"  # type: ignore[misc]
    except FrozenInstanceError:
        return
    raise AssertionError("OfficialHost deveria ser frozen")
