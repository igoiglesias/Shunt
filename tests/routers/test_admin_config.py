"""Tests for admin_config router + load_settings_from_db."""

from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config.settings import (
    ModelCaps,
    ModelConfig,
    ProviderConfig,
    Settings,
    load_settings_from_db,
)
from app.core.security import issue_jwt
from app.core.upstream import UpstreamPool
from app.main import app
from app.routers.admin_config import create_config_version
from app.stats.models import Base, ConfigVersion, Model, Provider, Route, RouteCandidate
from app.stats.recorder import Recorder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_engine(tmp_path: Path):
    db = tmp_path / "admin_test.db"
    engine = create_engine(f"sqlite:///{db}")
    Base.metadata.create_all(engine)
    return engine


ADMIN_SESSION_SECRET = "segredo-de-teste-do-admin"


def _reset_app_state():
    app.state.settings = Settings(
        providers={}, models={}, routes=[], default_model=None
    )
    app.state.pool = UpstreamPool(app.state.settings)
    # Sessao JWT validada em toda rota admin; o segredo precisa estar no
    # app.state para o cookie de teste ser aceito.
    app.state.admin_session_secret = ADMIN_SESSION_SECRET


def client(tmp_path: Path, *, no_engine: bool = False):
    """Return a TestClient backed by a per-test temp SQLite DB and a Recorder.

    `no_engine` builds a client whose Recorder has no engine, so the 503
    path can be exercised without poking global state outside the context.
    """
    engine = None if no_engine else _make_engine(tmp_path)
    _reset_app_state()
    app.state.recorder = Recorder(engine)
    return TestClient(app)


def seed_db(session: Session):
    """Populate one provider + two models + one route + one real version."""
    p = Provider(name="openrouter", base_url="https://or.test", protocol="openai", api_key_env=None)
    session.add(p)
    session.flush()
    m1 = Model(alias="free", provider_id=p.id, upstream_model="vendor/free", supports_tools=True, supports_streaming=True, supports_vision=False, context_window=64000, max_output_tokens=8192, is_default=True)
    m2 = Model(alias="cheap", provider_id=p.id, upstream_model="vendor/cheap", supports_tools=False, supports_streaming=True, supports_vision=False, context_window=32000, max_output_tokens=4096, is_default=False)
    session.add_all([m1, m2])
    session.flush()
    r = Route(pattern="opus", order_index=0)
    session.add(r)
    session.flush()
    session.add(RouteCandidate(route_id=r.id, model_alias="free", order_index=0))
    session.add(RouteCandidate(route_id=r.id, model_alias="cheap", order_index=1))
    session.flush()
    cv = create_config_version(session)
    return cv.id


# Cookie de sessao JWT valido, aceito pela dependencia `require_admin`.
# O segredo e o mesmo injetado em `_reset_app_state` acima.
COOKIE = {"shunt_admin": issue_jwt(user_id=1, secret=ADMIN_SESSION_SECRET, ttl_seconds=3600)}


# ---- Auth ------------------------------------------------------------------


def test_auth_no_engine_returns_503(tmp_path):
    """Sessao valida mas sem banco: a pagina responde 503, nao segue para render."""
    with client(tmp_path, no_engine=True) as c:
        r = c.get("/admin/config", cookies=COOKIE)
    assert r.status_code == 503
    assert r.json()["detail"] == "Persistencia desligada"


# ---- Page ------------------------------------------------------------------


def test_config_page_renders_seeded_names(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            seed_db(s)
        r = c.get("/admin/config", cookies=COOKIE)
    assert r.status_code == 200
    assert "openrouter" in r.text
    assert "free" in r.text
    assert "opus" in r.text


# ---- Providers -------------------------------------------------------------


def test_create_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.post("/admin/config/providers", data={"name": "anthropic", "base_url": "https://ai.test", "protocol": "anthropic"}, cookies=COOKIE, follow_redirects=False)
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        rows = s.execute(select(Provider).where(Provider.name == "anthropic")).scalars().all()
    assert len(rows) == 1
    assert rows[0].base_url == "https://ai.test"


def test_create_provider_duplicate_400(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="x", base_url="u", protocol="openai"))
            s.commit()
        r = c.post("/admin/config/providers", data={"name": "x", "base_url": "u", "protocol": "openai"}, cookies=COOKIE)
    assert r.status_code == 400
    assert r.json()["detail"] == "Provider ja existe"


def test_patch_provider_unknown_404(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.patch("/admin/config/providers/missing", data={"name": "x", "base_url": "u", "protocol": "openai"}, cookies=COOKIE)
    assert r.status_code == 404
    assert r.json()["detail"] == "Provider nao encontrado"


def test_delete_provider_unknown_404(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.delete("/admin/config/providers/missing", cookies=COOKIE)
    assert r.status_code == 404
    assert r.json()["detail"] == "Provider nao encontrado"


def test_delete_provider_referenced_by_model_400(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="p1", base_url="u", protocol="openai")
            s.add(p); s.flush()
            m = Model(alias="m1", provider_id=p.id, upstream_model="x", context_window=1024, max_output_tokens=128)
            s.add(m); s.commit()
        r = c.delete("/admin/config/providers/p1", cookies=COOKIE)
    assert r.status_code == 400
    assert r.json()["detail"] == "Provider em uso por models"


def test_patch_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="u", protocol="openai"))
            s.commit()
        r = c.patch("/admin/config/providers/p1", data={"base_url": "u2", "protocol": "anthropic"}, cookies=COOKIE, follow_redirects=False)
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        p = s.execute(select(Provider).where(Provider.name == "p1")).scalar_one()
    assert p.base_url == "u2"
    assert p.protocol == "anthropic"


def test_create_provider_invalid_protocol_400(monkeypatch, tmp_path):
    """Sem validacao o valor invalido commita e load_settings_from_db levanta
    depois -> 500 em toda mutacao ate a linha sair do banco."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.post("/admin/config/providers", data={"name": "bad", "base_url": "u", "protocol": "grpc"}, cookies=COOKIE)
    assert r.status_code == 400
    assert r.json()["detail"] == "Protocol deve ser openai ou anthropic"
    with Session(app.state.recorder.engine) as s:
        assert s.execute(select(Provider).where(Provider.name == "bad")).scalar_one_or_none() is None


def test_patch_provider_invalid_protocol_400(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="u", protocol="openai"))
            s.commit()
        r = c.patch("/admin/config/providers/p1", data={"base_url": "u2", "protocol": "grpc"}, cookies=COOKIE)
    assert r.status_code == 400
    with Session(app.state.recorder.engine) as s:
        p = s.execute(select(Provider).where(Provider.name == "p1")).scalar_one()
    assert p.protocol == "openai"  # nao mutou


def test_delete_provider_ok(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="gone", base_url="u", protocol="openai"))
            s.commit()
        r = c.delete("/admin/config/providers/gone", cookies=COOKIE, follow_redirects=False)
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        assert s.get(Provider, "gone") is None


def test_new_provider_form_200(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.get("/admin/config/providers/new", cookies=COOKIE)
    assert r.status_code == 200


def test_edit_provider_form_200(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="u", protocol="openai"))
            s.commit()
        r = c.get("/admin/config/providers/p1/edit", cookies=COOKIE)
    assert r.status_code == 200


# ---- Models ----------------------------------------------------------------


def test_create_model_clears_prior_default(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="u", protocol="openai"))
            s.commit()
        r = c.post("/admin/config/models", data={
            "alias": "default1", "provider_id": "1", "upstream_model": "x",
            "is_default": "true", "supports_tools": "false",
            "supports_streaming": "false", "supports_vision": "false",
            "context_window": "4096", "max_output_tokens": "512",
        }, cookies=COOKIE, follow_redirects=False)
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        default_rows = s.execute(select(Model).where(Model.is_default == True)).scalars().all()
        m = s.execute(select(Model).where(Model.alias == "default1")).scalar_one()
    assert [row_.alias for row_ in default_rows] == ["default1"]
    # Colunas persistidas: so a flag default nao segura um cruzamento de campos.
    assert m.upstream_model == "x"
    assert m.supports_tools is False
    assert m.supports_streaming is False
    assert m.supports_vision is False
    assert m.context_window == 4096
    assert m.max_output_tokens == 512


def test_create_model_duplicate_alias_400(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="p1", base_url="u", protocol="openai"); s.add(p); s.flush()
            s.add(Model(alias="dup", provider_id=p.id, upstream_model="x", context_window=1024, max_output_tokens=128))
            s.commit()
        r = c.post("/admin/config/models", data={"alias": "dup", "provider_id": "1", "upstream_model": "x"}, cookies=COOKIE)
    assert r.status_code == 400
    assert r.json()["detail"] == "Model alias ja existe"


def test_create_model_unknown_provider_400(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.post("/admin/config/models", data={"alias": "m1", "provider_id": "999", "upstream_model": "x"}, cookies=COOKIE)
    assert r.status_code == 400
    assert r.json()["detail"] == "Provider invalido"


def test_patch_model_unknown_alias_404(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.patch("/admin/config/models/missing", data={"provider_id": "1", "upstream_model": "x"}, cookies=COOKIE)
    assert r.status_code == 404
    assert r.json()["detail"] == "Model nao encontrado"


def test_patch_model_to_unknown_provider_400(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="p1", base_url="u", protocol="openai"); s.add(p); s.flush()
            s.add(Model(alias="m1", provider_id=p.id, upstream_model="x", context_window=1024, max_output_tokens=128))
            s.commit()
        r = c.patch("/admin/config/models/m1", data={"provider_id": "999", "upstream_model": "x"}, cookies=COOKIE)
    assert r.status_code == 400
    assert r.json()["detail"] == "Provider invalido"


def test_patch_model_unset_only_default_400(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="p1", base_url="u", protocol="openai"); s.add(p); s.flush()
            m = Model(alias="m1", provider_id=p.id, upstream_model="x", context_window=1024, max_output_tokens=128, is_default=True)
            s.add(m); s.commit()
        r = c.patch("/admin/config/models/m1", data={"provider_id": "1", "upstream_model": "x", "is_default": "false"}, cookies=COOKIE)
    assert r.status_code == 400
    assert r.json()["detail"] == "Nao pode remover o unico default"


def test_patch_model_success_persists_fields_and_flips_default(monkeypatch, tmp_path):
    """Caminho de sucesso do PATCH: campos persistem e o default anterior cede.
    Sem isso so os erros (404/400) estao cobertos e um cruzamento de campo passa."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="p1", base_url="u", protocol="openai"); s.add(p); s.flush()
            old = Model(alias="old", provider_id=p.id, upstream_model="o", context_window=1024, max_output_tokens=128, is_default=True)
            new = Model(alias="new", provider_id=p.id, upstream_model="n", context_window=2048, max_output_tokens=256)
            s.add_all([old, new]); s.commit()
        r = c.patch("/admin/config/models/new", data={
            "provider_id": "1", "upstream_model": "n2", "is_default": "true",
            "supports_tools": "false", "supports_streaming": "false",
            "context_window": "8192", "max_output_tokens": "1024",
        }, cookies=COOKIE, follow_redirects=False)
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        m = s.execute(select(Model).where(Model.alias == "new")).scalar_one()
        defaults = [row_.alias for row_ in s.execute(select(Model).where(Model.is_default == True)).scalars().all()]
    assert m.upstream_model == "n2"
    assert m.supports_tools is False
    assert m.context_window == 8192
    assert m.max_output_tokens == 1024
    assert defaults == ["new"]  # old perdeu o default


def test_patch_model_omitted_ints_keep_current_value(monkeypatch, tmp_path):
    """Cliente direto omite context_window/max_output_tokens: o valor atual
    preserva, nao reseta para o default do endpoint (mesma regra dos flags)."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="p1", base_url="u", protocol="openai"); s.add(p); s.flush()
            m = Model(alias="m1", provider_id=p.id, upstream_model="x", context_window=64000, max_output_tokens=2048)
            s.add(m); s.commit()
        r = c.patch("/admin/config/models/m1", data={"provider_id": "1", "upstream_model": "y"}, cookies=COOKIE, follow_redirects=False)
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        m = s.execute(select(Model).where(Model.alias == "m1")).scalar_one()
    assert m.context_window == 64000
    assert m.max_output_tokens == 2048


def test_patch_model_real_htmx_payload_with_duplicate_keys(monkeypatch, tmp_path):
    """O navegador envia o hidden companion ANTES do checkbox: duas chaves iguais.
    Starlette devolve a ultima (values[-1]), entao `true` so vale se vem depois.
    Este teste segura o payload real de navegador -- todos os outros enviam valor
    unico e nao provariam nada sobre a ordem das chaves."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="p1", base_url="u", protocol="openai"); s.add(p); s.flush()
            m = Model(alias="m1", provider_id=p.id, upstream_model="x", context_window=1024, max_output_tokens=128)
            s.add(m); s.commit()
        # Desmarca tools (hidden=false so), deixa streaming marcado
        # (false entao true). Chave duplicada = a que o htmx envia.
        r = c.patch(
            "/admin/config/models/m1",
            data={
                "provider_id": "1",
                "upstream_model": "x",
                "supports_tools": ["false"],
                "supports_streaming": ["false", "true"],
                "context_window": "1024",
                "max_output_tokens": "128",
                "is_default": ["false"],
            },
            cookies=COOKIE,
            follow_redirects=False,
        )
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        m = s.execute(select(Model).where(Model.alias == "m1")).scalar_one()
    assert m.supports_tools is False
    assert m.supports_streaming is True


def test_delete_model_used_by_route_400(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="p1", base_url="u", protocol="openai"); s.add(p); s.flush()
            m = Model(alias="m1", provider_id=p.id, upstream_model="x", context_window=1024, max_output_tokens=128)
            s.add(m); s.flush()
            rt = Route(pattern="pat", order_index=0); s.add(rt); s.flush()
            s.add(RouteCandidate(route_id=rt.id, model_alias="m1", order_index=0))
            s.commit()
        r = c.delete("/admin/config/models/m1", cookies=COOKIE)
    assert r.status_code == 400
    assert r.json()["detail"] == "Model em uso por routes"


def test_delete_model_default_400(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="p1", base_url="u", protocol="openai"); s.add(p); s.flush()
            m = Model(alias="m1", provider_id=p.id, upstream_model="x", context_window=1024, max_output_tokens=128, is_default=True)
            s.add(m); s.commit()
        r = c.delete("/admin/config/models/m1", cookies=COOKIE)
    assert r.status_code == 400
    assert r.json()["detail"] == "Nao pode deletar default model"


def test_new_model_form_200(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.get("/admin/config/models/new", cookies=COOKIE)
    assert r.status_code == 200


def test_edit_model_form_200(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="p1", base_url="u", protocol="openai"); s.add(p); s.flush()
            s.add(Model(alias="m1", provider_id=p.id, upstream_model="x", context_window=1024, max_output_tokens=128))
            s.commit()
        r = c.get("/admin/config/models/m1/edit", cookies=COOKIE)
    assert r.status_code == 200


# ---- Routes ----------------------------------------------------------------


def test_create_route_empty_pattern_400(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.post("/admin/config/routes", data={"pattern": "", "candidates": ["free"]}, cookies=COOKIE)
    assert r.status_code == 400
    assert r.json()["detail"] == "Pattern obrigatorio"


def test_create_route_no_candidates_400(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.post("/admin/config/routes", data={"pattern": "x", "candidates": []}, cookies=COOKIE)
    assert r.status_code == 400
    assert r.json()["detail"] == "Pelo menos um candidato obrigatorio"


def test_create_route_unknown_candidate_400(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.post("/admin/config/routes", data={"pattern": "x", "candidates": "nonexistent"}, cookies=COOKIE)
    assert r.status_code == 400
    assert r.json()["detail"] == "Model nonexistent nao existe"


def test_create_route_gets_order_max_plus_one(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            seed_db(s)
        r = c.post("/admin/config/routes", data={"pattern": "nova", "candidates": ["free"]}, cookies=COOKIE, follow_redirects=False)
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        route = s.execute(select(Route).where(Route.pattern == "nova")).scalars().first()
    assert route is not None
    assert route.order_index == 1  # seed has order_index=0


def test_patch_route_unknown_id_404(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.patch("/admin/config/routes/999", data={"pattern": "x", "candidates": "free"}, cookies=COOKIE)
    assert r.status_code == 404
    assert r.json()["detail"] == "Route nao encontrado"


def test_patch_route_replaces_candidates(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            seed_db(s)
        route_id = 1
        r = c.patch(f"/admin/config/routes/{route_id}", data={"pattern": "opus2", "candidates": "cheap"}, cookies=COOKIE, follow_redirects=False)
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        olds = s.execute(select(RouteCandidate).where(RouteCandidate.route_id == route_id)).scalars().all()
        aliases = [c.model_alias for c in olds]
    assert aliases == ["cheap"]


def test_create_route_drops_blank_and_dup_candidates(monkeypatch, tmp_path):
    """<select> em "— selecione —" manda ""; duplicatas de fallback sem sentido."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        engine = c.app.state.recorder.engine
        with Session(engine) as s:
            seed_db(s)  # modelos free + cheap
        # "" (option vazio) e duplicata de "free" devem sumir; ordem preservada
        r = c.post("/admin/config/routes", data={"pattern": "nova", "candidates": ["", "free", "cheap", "free"]}, cookies=COOKIE)
    assert r.status_code == 200
    with Session(engine) as s:
        route = s.execute(select(Route).where(Route.pattern == "nova")).scalar_one()
        cands = s.execute(
            select(RouteCandidate).where(RouteCandidate.route_id == route.id).order_by(RouteCandidate.order_index)
        ).scalars().all()
    assert [c.model_alias for c in cands] == ["free", "cheap"]


def test_reorder_routes(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        engine = c.app.state.recorder.engine
        with Session(engine) as s:
            seed_db(s)  # rota id=1 (opus), order_index=0
            r2 = Route(pattern="segunda", order_index=1)
            s.add(r2)
            s.flush()
            s.add(RouteCandidate(route_id=r2.id, model_alias="cheap", order_index=0))
            s.commit()  # rota id=2 com candidato; load_settings_from_db valida
        # Inverte: id=2 vai primeiro, id=1 fica em segundo
        r = c.post("/admin/config/routes/reorder", data={"order": ["2", "1"]}, cookies=COOKIE, follow_redirects=False)
    assert r.status_code == 200
    with Session(engine) as s:
        first = s.execute(select(Route).where(Route.id == 2)).scalar_one()
        second = s.execute(select(Route).where(Route.id == 1)).scalar_one()
    assert first.order_index == 0
    assert second.order_index == 1


def test_reorder_partial_order_400(monkeypatch, tmp_path):
    """Payload parcial deixaria a rota fora da lista com order_index antigo ->
    prioridade de fallback indeterminada."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        engine = c.app.state.recorder.engine
        with Session(engine) as s:
            seed_db(s)  # rota id=1
            r2 = Route(pattern="segunda", order_index=1); s.add(r2); s.flush()
            s.add(RouteCandidate(route_id=r2.id, model_alias="cheap", order_index=0))
            s.commit()
        r = c.post("/admin/config/routes/reorder", data={"order": ["1"]}, cookies=COOKIE)
    assert r.status_code == 400
    with Session(engine) as s:
        # Nada mudou: o commit nao aconteceu.
        first = s.execute(select(Route).where(Route.id == 1)).scalar_one()
    assert first.order_index == 0


def test_reorder_duplicate_ids_400(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        engine = c.app.state.recorder.engine
        with Session(engine) as s:
            seed_db(s)  # rota id=1
            r2 = Route(pattern="segunda", order_index=1); s.add(r2); s.flush()
            s.add(RouteCandidate(route_id=r2.id, model_alias="cheap", order_index=0))
            s.commit()
        r = c.post("/admin/config/routes/reorder", data={"order": ["1", "1"]}, cookies=COOKIE)
    assert r.status_code == 400


def test_delete_route_unknown_404(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.delete("/admin/config/routes/999", cookies=COOKIE)
    assert r.status_code == 404
    assert r.json()["detail"] == "Route nao encontrado"


def test_delete_route_with_candidates_ok(monkeypatch, tmp_path):
    """Regressao: session.delete(route) com candidatos quebrou o commit
    (route_candidates.route_id nullable=False, sem cascade) -> IntegrityError."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        engine = c.app.state.recorder.engine
        with Session(engine) as s:
            seed_db(s)  # rota id=1 com 2 candidatos
        r = c.delete("/admin/config/routes/1", cookies=COOKIE)
    assert r.status_code == 200
    with Session(engine) as s:
        route = s.execute(select(Route).where(Route.id == 1)).scalar_one_or_none()
        candidates = s.execute(
            select(RouteCandidate).where(RouteCandidate.route_id == 1)
        ).scalars().all()
    assert route is None
    assert candidates == []


def test_create_provider_applies_to_pool_without_reload(monkeypatch, tmp_path):
    """Regressao: CRUD devia exigir /reload para entrar no pool. Agora cada
    mutacao aplica settings na hora."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.post(
            "/admin/config/providers",
            data={"name": "newprov", "base_url": "https://np.test", "protocol": "openai"},
            cookies=COOKIE,
        )
    assert r.status_code == 200
    assert app.state.pool.get("newprov") is not None


def test_new_route_form_200(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.get("/admin/config/routes/new", cookies=COOKIE)
    assert r.status_code == 200


def test_edit_route_form_200(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            seed_db(s)
        r = c.get("/admin/config/routes/1/edit", cookies=COOKIE)
    assert r.status_code == 200


# ---- Default model ---------------------------------------------------------


def test_default_model_form_200(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            seed_db(s)
        r = c.get("/admin/config/default-model", cookies=COOKIE)
    assert r.status_code == 200


def test_set_default_unknown_alias_404(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.post("/admin/config/default-model", data={"alias": "no-such"}, cookies=COOKIE)
    assert r.status_code == 404
    assert r.json()["detail"] == "Model nao encontrado"


def test_set_default_flips_is_default_in_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            seed_db(s)
        r = c.post("/admin/config/default-model", data={"alias": "cheap"}, cookies=COOKIE, follow_redirects=False)
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        default = s.execute(select(Model).where(Model.is_default == True)).scalars().all()
    assert [m.alias for m in default] == ["cheap"]


# ---- Reload ----------------------------------------------------------------


def test_reload_reflects_db_mutation(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            seed_db(s)
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="newprov", base_url="https://np.test", protocol="openai", api_key_env=None))
            s.commit()
        r = c.post("/admin/config/reload", cookies=COOKIE)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "reloaded"
    assert body["default_model"] == "free"
    assert "newprov" in app.state.settings.providers
    # Regression: the pool was rebuilt and can resolve the new provider
    # without raising KeyError.
    pool_client = app.state.pool.get("newprov")
    assert pool_client is not None


def test_reload_pool_no_keyerror(monkeypatch, tmp_path):
    """Regression for the pool-rebuild fix: after reload, get() on a
    provider that was added directly to the DB does not raise."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="newprov", base_url="https://np.test", protocol="openai", api_key_env=None))
            s.commit()
        r = c.post("/admin/config/reload", cookies=COOKIE)
    assert r.status_code == 200
    assert "newprov" in app.state.settings.providers
    assert app.state.pool.get("newprov") is not None


# ---- Rollback --------------------------------------------------------------


def test_rollback_restores_snapshot(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            cv_id = seed_db(s)
            r = Route(pattern="mutated", order_index=99)
            s.add(r); s.flush()
            mut_id = r.id
            s.commit()
        r = c.post(f"/admin/config/rollback/{cv_id}", cookies=COOKIE, follow_redirects=False)
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        assert s.get(Route, mut_id) is None
        routes = s.execute(select(Route).order_by(Route.order_index)).scalars().all()
        assert [rt.pattern for rt in routes] == ["opus"]
        versions = s.execute(select(ConfigVersion)).scalars().all()
    assert len(versions) >= 2
    # Regression: rollback rebuilt settings, so the mutated route is gone
    # from app.state.settings.routes and the default matches the snapshot.
    assert ("mutated",) not in app.state.settings.routes
    assert app.state.settings.default_model == "free"


def test_rollback_unknown_id_404(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.post("/admin/config/rollback/99999", cookies=COOKIE)
    assert r.status_code == 404
    assert r.json()["detail"] == "Versao nao encontrada"


# ---- History ---------------------------------------------------------------


def test_history_200_with_versions(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            seed_db(s)
        r = c.get("/admin/config/history", cookies=COOKIE)
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        versions = s.execute(select(ConfigVersion)).scalars().all()
    assert len(versions) > 0


# ---- load_settings_from_db -------------------------------------------------


def test_load_settings_from_db(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path):
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="groq", base_url="https://g.test", protocol="openai", api_key_env=None)
            s.add(p); s.flush()
            m1 = Model(alias="g-free", provider_id=p.id, upstream_model="groq/compound", supports_tools=True, supports_streaming=False, supports_vision=False, context_window=8192, max_output_tokens=1024, is_default=True)
            m2 = Model(alias="g-fast", provider_id=p.id, upstream_model="groq/llama", supports_tools=False, supports_streaming=True, supports_vision=False, context_window=4096, max_output_tokens=512, is_default=False)
            s.add_all([m1, m2])
            s.flush()
            r1 = Route(pattern="chat", order_index=0)
            s.add(r1); s.flush()
            c1 = RouteCandidate(route_id=r1.id, model_alias="g-free", order_index=0)
            c2 = RouteCandidate(route_id=r1.id, model_alias="g-fast", order_index=1)
            s.add_all([c1, c2])
            s.commit()

        with Session(app.state.recorder.engine) as s:
            settings = load_settings_from_db(s)

    assert settings.providers["groq"] == ProviderConfig(base_url="https://g.test", protocol="openai", api_key_env=None)
    assert settings.models["g-free"] == ModelConfig(provider="groq", model="groq/compound", supports=ModelCaps(tools=True, streaming=False, vision=False), context_window=8192, max_output_tokens=1024)
    assert settings.models["g-fast"] == ModelConfig(provider="groq", model="groq/llama", supports=ModelCaps(tools=False, streaming=True, vision=False), context_window=4096, max_output_tokens=512)
    assert settings.routes == [("chat", ["g-free", "g-fast"])]
    assert settings.default_model == "g-free"
