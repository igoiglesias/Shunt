"""Testes do seed do catálogo: insere se vazio, idempotente, chaves vêm do env."""

import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

# Define chaves de teste ANTES de importar seed
os.environ["LOCAL_API_KEY"] = "local-test"
os.environ["OPENROUTER_API_KEY"] = "sk-or-test"
os.environ["GROQ_API_KEY"] = "gsk-test"
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"

from app.config import seed
from app.stats.models import Model, Provider, Route, RouteCandidate


def test_seed_creates_catalog_from_env():
    engine = create_engine("sqlite+pysqlite:////tmp/shunt-test-seed.db")
    try:
        from app.stats.models import Base
        Base.metadata.create_all(engine)
        seed.seed_catalog_if_empty(engine)

        with Session(engine) as session:
            # providers
            providers = session.query(Provider).order_by(Provider.name).all()
            assert len(providers) == 3
            names = {p.name for p in providers}
            assert names == {"local", "openrouter", "groq"}

            # local sem chave
            p = session.query(Provider).filter_by(name="local").one()
            assert p.base_url == "http://127.0.0.1:8080/v1"
            assert p.protocol == "openai"
            assert p.api_key == "local-test"  # vem do env

            # openrouter
            p = session.query(Provider).filter_by(name="openrouter").one()
            assert p.api_key == "sk-or-test"

            # groq
            p = session.query(Provider).filter_by(name="groq").one()
            assert p.api_key == "gsk-test"

            # models (6 no catálogo atual)
            models = session.query(Model).order_by(Model.alias).all()
            assert len(models) == 6
            aliases = [m.alias for m in models]
            expected = [
                "groq-free", "open-deepseek-v4.1-flash", "open-free",
                "open-gpt-oss-120", "open-nemotron-ultra", "qwen-local"
            ]
            assert sorted(aliases) == sorted(expected)

            # default_model
            defaults = [m for m in models if m.is_default]
            assert len(defaults) == 1
            assert defaults[0].alias == "open-gpt-oss-120"

            # routes
            routes = session.query(Route).order_by(Route.order_index).all()
            assert len(routes) == 10
            patterns = [r.pattern for r in routes]
            expected_patterns = [
                "groq", "local", "nemotron", "deepseek", "gpt-oss",
                "free", "fable", "haiku", "sonnet", "opus"
            ]
            assert patterns == expected_patterns

            # candidates order
            for r in routes:
                candidates = session.query(RouteCandidate).filter_by(route_id=r.id).order_by(RouteCandidate.order_index).all()
                assert len(candidates) > 0

    finally:
        import os as _os
        _os.remove("/tmp/shunt-test-seed.db")


def test_seed_idempotent():
    """Chamada repetida não duplica."""
    engine = create_engine("sqlite+pysqlite:////tmp/shunt-test-seed2.db")
    try:
        from app.stats.models import Base
        Base.metadata.create_all(engine)
        seed.seed_catalog_if_empty(engine)
        seed.seed_catalog_if_empty(engine)
        seed.seed_catalog_if_empty(engine)

        with Session(engine) as session:
            assert session.query(Provider).count() == 3
            assert session.query(Model).count() == 6
            assert session.query(Route).count() == 10
    finally:
        import os as _os
        _os.remove("/tmp/shunt-test-seed2.db")


def test_seed_noop_when_catalog_complete():
    """Catalogo completo (3 tabelas nao vazias) e dono do operador: o seed nao toca."""
    engine = create_engine("sqlite+pysqlite:////tmp/shunt-test-seed3.db")
    try:
        from app.stats.models import Base
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            p = Provider(name="preexistente", base_url="http://x", protocol="openai")
            session.add(p)
            session.flush()
            session.add(Model(alias="preex", provider_id=p.id, upstream_model="x", is_default=True,
                              context_window=8192, max_output_tokens=1024))
            session.add(Route(pattern="preex", order_index=0))
            session.commit()

        seed.seed_catalog_if_empty(engine)

        with Session(engine) as session:
            assert session.query(Provider).count() == 1
            assert session.query(Model).count() == 1
            assert session.query(Route).count() == 1
            p = session.query(Provider).one()
            assert p.name == "preexistente"
    finally:
        import os as _os
        _os.remove("/tmp/shunt-test-seed3.db")


def test_seed_repairs_catalog_missing_routes():
    """DB com resto de QA (provider/model sem rotas): o seed completa o catalogo.

    Defeito medido no banco real: um provider criado por QA deixava
    `count(Provider) > 0` e o seed pulava -- zero rotas, todo pedido 400.
    Regra: se qualquer tabela do catalogo estiver vazia, as partes
    padrao ausentes entram; linhas existentes sobrevivem.
    """
    engine = create_engine("sqlite+pysqlite:////tmp/shunt-test-seed4.db")
    try:
        from app.stats.models import Base
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            p = Provider(name="QA", base_url="http://qa", protocol="openai")
            session.add(p)
            session.flush()
            session.add(Model(alias="qa-test", provider_id=p.id, upstream_model="vendor/qa",
                              context_window=8192, max_output_tokens=1024))
            session.commit()

        seed.seed_catalog_if_empty(engine)

        with Session(engine) as session:
            names = {p.name for p in session.query(Provider)}
            assert names == {"QA", "local", "openrouter", "groq"}
            aliases = {m.alias for m in session.query(Model)}
            expected = {"qa-test", "qwen-local", "open-free", "open-nemotron-ultra",
                        "open-deepseek-v4.1-flash", "open-gpt-oss-120", "groq-free"}
            assert aliases == expected
            assert session.query(Route).count() == 10
            orp = session.query(Provider).filter_by(name="openrouter").one()
            assert orp.api_key == "sk-or-test"
    finally:
        import os as _os
        _os.remove("/tmp/shunt-test-seed4.db")


def test_seed_migrates_api_key_env_to_api_key():
    """Linha antiga com o nome da variavel vira chave bruta no boot."""
    os.environ["LEGACY_KEY"] = "legacy-value"
    engine = create_engine("sqlite+pysqlite:////tmp/shunt-test-seed5.db")
    try:
        from app.stats.models import Base
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            p = Provider(name="legado", base_url="http://x", protocol="openai")
            session.add(p)
            session.flush()
            session.add(Provider(name="ausente", base_url="http://y", protocol="openai"))
            session.flush()
            # Schema legacy: a coluna `api_key_env` so existia em bancos antigos.
            session.execute(text("ALTER TABLE providers ADD COLUMN api_key_env VARCHAR(128)"))
            session.execute(text("UPDATE providers SET api_key_env = 'LEGACY_KEY' WHERE name = 'legado'"))
            session.execute(text("UPDATE providers SET api_key_env = 'SHUNT_VAR_AUSENTE_XYZ' WHERE name = 'ausente'"))
            session.add(Model(alias="preex", provider_id=p.id, upstream_model="x", is_default=True,
                              context_window=8192, max_output_tokens=1024))
            session.add(Route(pattern="preex", order_index=0))
            session.commit()

        seed.seed_catalog_if_empty(engine)

        with Session(engine) as session:
            legado = session.query(Provider).filter_by(name="legado").one()
            assert legado.api_key == "legacy-value"
            # variavel ausente no ambiente: nao inventa chave
            ausente = session.query(Provider).filter_by(name="ausente").one()
            assert ausente.api_key is None
    finally:
        os.environ.pop("LEGACY_KEY", None)
        import os as _os
        _os.remove("/tmp/shunt-test-seed5.db")


def test_catalog_settings_builds():
    """catalog_settings() retorna Settings com chaves do env."""
    from app.config import seed
    from app.config.settings import Settings

    s = seed.catalog_settings()
    assert isinstance(s, Settings)
    assert len(s.providers) == 3
    assert s.providers["local"].api_key == "local-test"
    assert s.providers["openrouter"].api_key == "sk-or-test"
    assert s.providers["groq"].api_key == "gsk-test"
    assert len(s.models) == 6
    assert s.default_model == "open-gpt-oss-120"
    assert len(s.routes) == 10
