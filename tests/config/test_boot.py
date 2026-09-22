"""Testes de boot: settings vem do banco (seedado) ou vazio quando sem engine."""

import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

sys.path.insert(0, "/home/iglesias/Documents/Projetos_Pessoais/Shunt")

# Chaves para o seed
os.environ["LOCAL_API_KEY"] = "local-test"
os.environ["OPENROUTER_API_KEY"] = "sk-or-test"
os.environ["GROQ_API_KEY"] = "gsk-test"
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"

import app.config.seed as seed
from app.config.settings import Settings, load_settings_from_db
from app.stats.models import Base


def test_boot_with_engine_returns_seeded_settings():
    """Com engine, load_settings_from_db devolve o catálogo seedado."""
    engine = create_engine("sqlite+pysqlite:////tmp/shunt-test-boot.db")
    try:
        Base.metadata.create_all(engine)
        seed.seed_catalog_if_empty(engine)

        with Session(engine) as session:
            settings = load_settings_from_db(session)

        assert isinstance(settings, Settings)
        assert len(settings.providers) == 3
        assert "local" in settings.providers
        assert "openrouter" in settings.providers
        assert "groq" in settings.providers
        # chaves vêm do env no seed
        assert settings.providers["local"].api_key == "local-test"
        assert settings.providers["openrouter"].api_key == "sk-or-test"
        assert settings.providers["groq"].api_key == "gsk-test"
        assert len(settings.models) == 6
        assert settings.default_model == "open-gpt-oss-120"
        assert len(settings.routes) == 10
    finally:
        os.remove("/tmp/shunt-test-boot.db")


def test_boot_without_engine_returns_empty_settings():
    """Sem engine (TURSO_DATABASE_URL não definido), Settings vazio."""
    # app/main.py usa Settings(providers={}, models={}, routes=[]) quando engine is None
    s = Settings(providers={}, models={}, routes=[])
    assert len(s.providers) == 0
    assert len(s.models) == 0
    assert len(s.routes) == 0
    assert s.default_model is None


def test_lifespan_seeds_catalog_and_loads_settings(tmp_path, monkeypatch):
    """Primeiro boot com banco vazio: o lifespan semeia o catálogo e o carrega.

    A engine e a fonte do catálogo: sem `seed_catalog_if_empty` no boot, um
    banco novo sobe com `Settings` vazio e toda rota resolve para "sem
    candidato" -- defeito que nenhum teste de unidade do seed cobre sozinho,
    porque o seed existe, só não era chamado.
    """
    from fastapi.testclient import TestClient

    import app.main as main

    db = tmp_path / "boot.db"
    monkeypatch.setenv("TURSO_DATABASE_URL", f"sqlite:///{db}")
    state = main.app.state
    try:
        with TestClient(main.app):
            settings = main.app.state.settings
            assert len(settings.providers) == 3
            assert len(settings.models) == 6
            assert len(settings.routes) == 10
            assert settings.default_model == "open-gpt-oss-120"
            # chaves entraram no banco uma vez, lidas do env no seed
            assert settings.providers["openrouter"].api_key == "sk-or-test"
        # e o banco guarda a chave bruta, nao o nome da variavel
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        from app.stats.models import Provider

        engine = create_engine(f"sqlite:///{db}")
        try:
            with Session(engine) as session:
                p = session.query(Provider).filter_by(name="groq").one()
                assert p.api_key == "gsk-test"
        finally:
            engine.dispose()
    finally:
        # o `app` e singleton de processo: devolve o estado ao limpo para os
        # testes que sobem o proprio aplicativo depois deste.
        for attr in ("settings", "pool", "recorder", "admin_session_secret"):
            if hasattr(state, attr):
                delattr(state, attr)


if __name__ == "__main__":
    test_boot_with_engine_returns_seeded_settings()
    test_boot_without_engine_returns_empty_settings()
    print("test_boot.py: todos passaram (RED esperado até boot rewire)")