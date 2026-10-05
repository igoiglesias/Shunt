"""Tests for admin_config router + load_settings_from_db."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
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
from app.routers.admin_config import apply_settings, create_config_version
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
    p = Provider(name="openrouter", base_url="https://or.test", protocol="openai", api_key=None)
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


# ---------------------------------------------------------------------------
# D1: os chips das tabelas de config sao ROTULOS, nao controles. Sem handler
# de clique e sem data-filter, `aria-pressed` e um atributo mentiroso -- o
# axe-core relata aria-allowed-attr. NAO viram <button>: um botao sem acao e
# um defeito pior. Os chips da audit/dashboard sao <button> de verdade e
# continuam fora disto.
# ---------------------------------------------------------------------------


def test_provider_chips_carry_no_aria_pressed(monkeypatch, tmp_path):
    """O chip de protocolo do provedor nao declara estado de botao."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(
                Provider(name="p1", base_url="https://p1.test", protocol="openai")
            )
            s.commit()
        r = c.get("/admin/config/providers", cookies=COOKIE)
    assert r.status_code == 200
    assert 'class="chip"' in r.text
    # A presenca do atributo e o defeito; ele mente interatividade.
    assert 'aria-pressed' not in r.text


def test_model_chips_carry_no_aria_pressed(monkeypatch, tmp_path):
    """Os 4 chips de capacidade/default da linha do modelo nao declaram
    estado de botao -- inclui o chip .bad, que pinta o vermelho."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="p1", base_url="u", protocol="openai")
            s.add(p)
            s.flush()
            # supports_tools False -> o chip vem com a classe .bad
            s.add(
                Model(
                    alias="m1",
                    provider_id=p.id,
                    upstream_model="x",
                    supports_tools=False,
                    supports_streaming=True,
                    supports_vision=False,
                    is_default=True,
                    context_window=8192,
                    max_output_tokens=4096,
                )
            )
            s.commit()
        r = c.get("/admin/config/models", cookies=COOKIE)
    assert r.status_code == 200
    assert 'class="chip bad"' in r.text
    assert 'class="chip"' in r.text
    assert "aria-pressed" not in r.text


# ---------------------------------------------------------------------------
# D3/D5: no mobile a tabela de config e mais larga que a tela e a coluna de
# acoes (Editar/Excluir) caia alem da dobra do scroll. A coluna fica colada a
# direita; os botoes tambem crescem para o dedo (44px). Estas sao regras de
# CSS dentro de @media (max-width: 760px) -- mesmo padrao que o arquivo ja
# usava. Sem navegador aqui, a assercao e sobre a regra existir ligada a
# essa media query; a prova de layout fica no suite de browser.
# ---------------------------------------------------------------------------


def test_mobile_media_query_glues_the_actions_column_to_the_right_edge(monkeypatch, tmp_path):
    """D3: a coluna de acoes e sticky a direita dentro do media query de
    760px, com fundo opaco para o conteudo nao mostrar por baixo."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="https://p1.test", protocol="openai"))
            s.commit()
        r = c.get("/admin/config", cookies=COOKIE)
    source = r.text
    media = source.split("@media (max-width: 760px)", 1)[1]
    # A regra so existe DENTRO do media query (nao vira sticky no desktop).
    # Os seletores casam a ULTIMA celula: as quatro tabelas da config pem as
    # acoes em <td class="n"><div class="actions"> -- a classe .actions esta
    # no div de dentro, entao td.actions/th.actions nao casavam nada.
    for needle in ("position: sticky", "right: 0",
                   "td:last-child", "th:last-child"):
        assert needle in media, f"{needle} precisa estar no media query de 760px"
    # O seletor morto nao pode sobrar como regra. Assercao no seletor, nao
    # em substring solto: o comentario que documenta o bug cita as palavras.
    assert "td.actions," not in media
    assert "td.actions {" not in media
    # Fundo opaco: transparent deixa a linha anterior aparecer na rolagem.
    assert "background: var(--panel)" in media


def test_mobile_media_query_raises_the_button_height_to_44px(monkeypatch, tmp_path):
    """D5: .btn e .btn-sm ganham min-height de 44px no media query de 760px
    (o piso para toque WCAG)."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            seed_db(s)
        r = c.get("/admin/config", cookies=COOKIE)
    media = r.text.split("@media (max-width: 760px)", 1)[1]
    assert ".btn { min-height: 44px; }" in media
    assert ".btn-sm" in media and "44px" in media


def test_providers_list_never_renders_the_full_key(monkeypatch, tmp_path):
    """A chave completa nunca entra no HTML; so os 4 ultimos digitos aparecem
    depois da mascara."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(
                Provider(
                    name="p1",
                    base_url="https://p1.test",
                    protocol="openai",
                    api_key="sk-chave-secreta-1234",
                )
            )
            s.commit()
        r = c.get("/admin/config/providers", cookies=COOKIE)
    assert r.status_code == 200
    assert "sk-chave-secreta-1234" not in r.text
    assert "1234" in r.text


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


def test_create_provider_persists_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.post(
            "/admin/config/providers",
            data={"name": "p1", "base_url": "u", "protocol": "openai", "api_key": "sk-nova"},
            cookies=COOKIE,
        )
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        p = s.execute(select(Provider).where(Provider.name == "p1")).scalar_one()
    assert p.api_key == "sk-nova"
    assert app.state.settings.providers["p1"].api_key == "sk-nova"


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
            s.add(Provider(name="newprov", base_url="https://np.test", protocol="openai", api_key=None))
            s.commit()
        r = c.post("/admin/config/reload", cookies=COOKIE)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "reloaded"
    assert body["default_model"] == "free"
    assert "newprov" in app.state.settings.providers
    # Regressao: o pool (o mesmo objeto) ja resolve o provedor novo sem KeyError.
    pool_client = app.state.pool.get("newprov")
    assert pool_client is not None


def test_reload_pool_no_keyerror(monkeypatch, tmp_path):
    """Regressao: o pool (o mesmo objeto) ja resolve o provedor novo sem
    KeyError -- pool reconciliado."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="newprov", base_url="https://np.test", protocol="openai", api_key=None))
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


def test_rollback_preserves_provider_api_key_by_name(monkeypatch, tmp_path):
    """O snapshot nao grava a chave bruta (so o bool de existencia), entao o
    rollback tem de carregar a chave atual do banco para o provider de mesmo
    nome. Sem isso `Provider(**snapshot)` grava o bool na coluna String."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="u", protocol="openai", api_key="sk-real-chave"))
            s.commit()
            cv_id = create_config_version(s).id
            s.commit()
        c.patch(
            "/admin/config/providers/p1",
            data={"base_url": "u2", "protocol": "openai"},
            cookies=COOKIE,
        )
        r = c.post(f"/admin/config/rollback/{cv_id}", cookies=COOKIE, follow_redirects=False)
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        p = s.execute(select(Provider).where(Provider.name == "p1")).scalar_one()
    assert p.base_url == "u"
    assert p.api_key == "sk-real-chave"
    # apply_settings recarrega do banco: a chave carregada do snapshot final
    # nao pode ser o bool gravado no historico.
    assert app.state.settings.providers["p1"].api_key == "sk-real-chave"


def test_edit_provider_form_does_not_prefill_the_key_with_bullets(monkeypatch, tmp_path):
    """Formulario pre-preenchido com •••••••• faz o Salvar enviar os bullet
    e o update sobrescrever a chave real com a string de mascaramento."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="u", protocol="openai", api_key="sk-real-chave"))
            s.commit()
        r = c.get("/admin/config/providers/p1/edit", cookies=COOKIE)
    assert r.status_code == 200
    assert 'value="••••••••"' not in r.text


def test_patch_provider_empty_api_key_keeps_current_key(monkeypatch, tmp_path):
    """Campo vazio no PATCH significa 'nao mude a chave'."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="u", protocol="openai", api_key="sk-real-chave"))
            s.commit()
        r = c.patch(
            "/admin/config/providers/p1",
            data={"base_url": "u2", "protocol": "openai", "api_key": ""},
            cookies=COOKIE,
        )
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        p = s.execute(select(Provider).where(Provider.name == "p1")).scalar_one()
    assert p.api_key == "sk-real-chave"
    # apply_settings recarrega do banco: a chave sobrevive as settings vivas.
    assert app.state.settings.providers["p1"].api_key == "sk-real-chave"


def test_patch_provider_with_key_overwrites_current_key(monkeypatch, tmp_path):
    """Valor nao vazio no campo sobrescreve a chave gravada."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="u", protocol="openai", api_key="sk-antiga"))
            s.commit()
        r = c.patch(
            "/admin/config/providers/p1",
            data={"base_url": "u", "protocol": "openai", "api_key": "sk-nova"},
            cookies=COOKIE,
        )
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        p = s.execute(select(Provider).where(Provider.name == "p1")).scalar_one()
    assert p.api_key == "sk-nova"
    # apply_settings recarrega do banco: a chave nova tambem entra nas settings
    # vivas, de onde o dispatcher lera em runtime.
    assert app.state.settings.providers["p1"].api_key == "sk-nova"


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
            p = Provider(name="groq", base_url="https://g.test", protocol="openai", api_key=None)
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

    assert settings.providers["groq"] == ProviderConfig(base_url="https://g.test", protocol="openai", api_key=None)
    assert settings.models["g-free"] == ModelConfig(provider="groq", model="groq/compound", supports=ModelCaps(tools=True, streaming=False, vision=False), context_window=8192, max_output_tokens=1024)
    assert settings.models["g-fast"] == ModelConfig(provider="groq", model="groq/llama", supports=ModelCaps(tools=False, streaming=True, vision=False), context_window=4096, max_output_tokens=512)
    assert settings.routes == [("chat", ["g-free", "g-fast"])]
    assert settings.default_model == "g-free"


# ---- Limite de concorrencia por provedor -----------------------------------


def test_provider_config_defaults_to_no_concurrency_limit():
    assert ProviderConfig(base_url="u", protocol="openai").max_concurrency is None


def test_provider_config_refuses_a_limit_below_one():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ProviderConfig(base_url="u", protocol="openai", max_concurrency=0)
    assert ProviderConfig(base_url="u", protocol="openai", max_concurrency=1).max_concurrency == 1


def test_load_settings_from_db_carries_the_concurrency_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path):
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="local", base_url="http://l", protocol="openai", max_concurrency=2))
            s.add(Provider(name="cloud", base_url="http://c", protocol="openai"))
            s.commit()
        with Session(app.state.recorder.engine) as s:
            settings = load_settings_from_db(s)
    assert settings.providers["local"].max_concurrency == 2
    assert settings.providers["cloud"].max_concurrency is None


MAX_CONCURRENCY_ERROR = "Maximo de requisicoes simultaneas deve ser um inteiro entre 1 e 10000"


def test_create_provider_with_a_concurrency_limit_reaches_db_settings_and_pool(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.post(
            "/admin/config/providers",
            data={"name": "local", "base_url": "http://l", "protocol": "openai", "max_concurrency": "2"},
            cookies=COOKIE,
        )
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        p = s.execute(select(Provider).where(Provider.name == "local")).scalar_one()
    assert p.max_concurrency == 2
    assert app.state.settings.providers["local"].max_concurrency == 2
    # O pool reconciliado ja conhece o limite novo.
    assert app.state.pool.limit("local") == 2


@pytest.mark.parametrize("raw", ["", "   "])
def test_create_provider_with_an_empty_limit_is_unlimited(monkeypatch, tmp_path, raw):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.post(
            "/admin/config/providers",
            data={"name": "cloud", "base_url": "http://c", "protocol": "openai", "max_concurrency": raw},
            cookies=COOKIE,
        )
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        p = s.execute(select(Provider).where(Provider.name == "cloud")).scalar_one()
    assert p.max_concurrency is None
    assert app.state.pool.limit("cloud") is None


@pytest.mark.parametrize(
    "raw",
    # "99999999999999999999" estourava o INTEGER do SQLite no commit (500);
    # "1_000" e digitos arabe-indicos passavam pelo `int()` do Python.
    # "9" * 5000 passa no isdigit() e o int() do Python recusa com ValueError
    # (limite de 4300 digitos) -> era 500.
    ["abc", "0", "-3", "1.5", "99999999999999999999", "10001", "1_000", "\u0661\u0662", "+5", pytest.param("9" * 5000, id="5000-digitos")],
)
def test_create_provider_with_an_invalid_limit_is_refused(monkeypatch, tmp_path, raw):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.post(
            "/admin/config/providers",
            data={"name": "bad", "base_url": "u", "protocol": "openai", "max_concurrency": raw},
            cookies=COOKIE,
        )
    assert r.status_code == 400
    assert r.json()["detail"] == MAX_CONCURRENCY_ERROR
    with Session(app.state.recorder.engine) as s:
        assert s.execute(select(Provider).where(Provider.name == "bad")).scalar_one_or_none() is None


def test_patch_provider_sets_then_clears_the_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="u", protocol="openai"))
            s.commit()
        r = c.patch(
            "/admin/config/providers/p1",
            data={"base_url": "u", "protocol": "openai", "max_concurrency": "3"},
            cookies=COOKIE,
        )
        assert r.status_code == 200
        with Session(app.state.recorder.engine) as s:
            assert s.execute(select(Provider)).scalar_one().max_concurrency == 3
        assert app.state.pool.limit("p1") == 3
        r = c.patch(
            "/admin/config/providers/p1",
            data={"base_url": "u", "protocol": "openai", "max_concurrency": ""},
            cookies=COOKIE,
        )
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        assert s.execute(select(Provider)).scalar_one().max_concurrency is None
    assert app.state.pool.limit("p1") is None


def test_patch_provider_with_an_invalid_limit_is_refused_and_changes_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="u", protocol="openai", max_concurrency=2))
            s.commit()
        r = c.patch(
            "/admin/config/providers/p1",
            data={"base_url": "u2", "protocol": "openai", "max_concurrency": "0"},
            cookies=COOKIE,
        )
    assert r.status_code == 400
    assert r.json()["detail"] == MAX_CONCURRENCY_ERROR
    with Session(app.state.recorder.engine) as s:
        p = s.execute(select(Provider)).scalar_one()
    assert (p.base_url, p.max_concurrency) == ("u", 2)


def test_snapshot_and_rollback_carry_the_limit(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="u", protocol="openai", max_concurrency=2))
            s.commit()
            cv = create_config_version(s)
            cv_id = cv.id
            assert json.loads(cv.snapshot_json)["providers"][0]["max_concurrency"] == 2
        c.patch(
            "/admin/config/providers/p1",
            data={"base_url": "u", "protocol": "openai", "max_concurrency": "5"},
            cookies=COOKIE,
        )
        r = c.post(f"/admin/config/rollback/{cv_id}", cookies=COOKIE)
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        assert s.execute(select(Provider)).scalar_one().max_concurrency == 2
    assert app.state.settings.providers["p1"].max_concurrency == 2


def test_rollback_of_an_old_snapshot_without_the_limit_restores_unlimited(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="u", protocol="openai", max_concurrency=4))
            s.flush()
            old = {
                "providers": [{"name": "p1", "base_url": "u", "protocol": "openai", "api_key": False}],
                "models": [],
                "routes": [],
                "default_model": None,
            }
            cv = ConfigVersion(snapshot_json=json.dumps(old))
            s.add(cv)
            s.commit()
            cv_id = cv.id
        r = c.post(f"/admin/config/rollback/{cv_id}", cookies=COOKIE)
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        assert s.execute(select(Provider)).scalar_one().max_concurrency is None


def test_the_provider_form_offers_the_limit_field(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        new = c.get("/admin/config/providers/new", cookies=COOKIE)
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="u", protocol="openai", max_concurrency=2))
            s.commit()
        edit = c.get("/admin/config/providers/p1/edit", cookies=COOKIE)
    assert 'name="max_concurrency"' in new.text
    assert 'type="number"' in new.text and 'min="1"' in new.text
    assert "Máximo de requisições simultâneas" in new.text
    assert "Vazio = sem limite" in new.text
    assert 'value="2"' in edit.text


def test_the_providers_list_shows_the_limit_or_a_dash(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="lim", base_url="u", protocol="openai", api_key="sk-abcd", max_concurrency=7))
            s.add(Provider(name="sem", base_url="u", protocol="openai", api_key="sk-wxyz"))
            s.commit()
        r = c.get("/admin/config/providers", cookies=COOKIE)
    assert "Simultâneas" in r.text
    rows = {m.group(1): m.group(0) for m in re.finditer(r"<tr>\s*<td><b>(\w+)</b>.*?</tr>", r.text, re.DOTALL)}
    assert re.search(r'<td class="n">7</td>', rows["lim"])
    assert re.search(r'<td class="n">—</td>', rows["sem"])


def test_the_limit_accepts_the_ceiling_and_surrounding_spaces(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.post(
            "/admin/config/providers",
            data={"name": "big", "base_url": "u", "protocol": "openai", "max_concurrency": " 10000 "},
            cookies=COOKIE,
        )
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        assert s.execute(select(Provider)).scalar_one().max_concurrency == 10000


def test_patch_provider_that_omits_the_limit_keeps_it(monkeypatch, tmp_path):
    """Cliente direto que manda so base_url/protocol nao pode apagar o limite
    -- o mesmo contrato da chave. Presente e vazio (o form) ainda limpa."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="u", protocol="openai", max_concurrency=2))
            s.commit()
        r = c.patch(
            "/admin/config/providers/p1",
            data={"base_url": "u2", "protocol": "openai"},
            cookies=COOKIE,
        )
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        p = s.execute(select(Provider)).scalar_one()
    assert (p.base_url, p.max_concurrency) == ("u2", 2)
    assert app.state.pool.limit("p1") == 2


def test_patch_unknown_provider_with_an_invalid_limit_is_404(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.patch(
            "/admin/config/providers/nao-existe",
            data={"base_url": "u", "protocol": "openai", "max_concurrency": "0"},
            cookies=COOKIE,
        )
    assert r.status_code == 404


def test_an_edit_keeps_the_pool_identity_and_retires_only_the_changed_client(monkeypatch, tmp_path):
    """O pool nunca e recriado: um stream em voo segue no cliente que ja usa e
    `in_use` nao zera. So o cliente do provedor cujo `base_url` mudou e
    aposentado (fechado por estar ocioso aqui); o do outro provedor e o mesmo."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="http://u1", protocol="openai"))
            s.add(Provider(name="p2", base_url="http://u2", protocol="openai"))
            s.commit()
        assert c.post("/admin/config/reload", cookies=COOKIE).status_code == 200
        pool = app.state.pool
        p1 = pool.get("p1")
        p2 = pool.get("p2")
        r = c.patch(
            "/admin/config/providers/p1",
            data={"base_url": "http://u1-novo", "protocol": "openai"},
            cookies=COOKIE,
        )
        # Dentro do `with`: o shutdown do TestClient fecha o pool inteiro
        # (`aclose`), que limparia `_clients` e derrubaria estas asserções
        # por um motivo que nada tem a ver com `update()`.
        assert r.status_code == 200
        assert app.state.pool is pool
        assert p1.is_closed
        assert pool.get("p1") is not p1
        assert str(pool.get("p1").base_url).rstrip("/") == "http://u1-novo"
        assert pool.get("p2") is p2 and not p2.is_closed


async def test_apply_settings_without_a_prior_pool_creates_one_instead_of_updating():
    """Defensivo: se `app.state.pool` ainda nao existe (antes do lifespan
    terminar de subir), `apply_settings` cria o pool em vez de chamar
    `update()` num `None`, que estouraria `AttributeError`."""
    settings = Settings(providers={}, models={}, routes=[], default_model=None)
    fake_app = SimpleNamespace(state=SimpleNamespace())
    await apply_settings(fake_app, settings)
    assert fake_app.state.settings is settings
    assert isinstance(fake_app.state.pool, UpstreamPool)


# ---------------------------------------------------------------------------
# Historico: prune, exclusao e limpeza do modelo padrao
# ---------------------------------------------------------------------------


def test_history_fragment_self_anchors_so_outerHTML_keeps_the_target_alive(monkeypatch, tmp_path):
    """O fragmento _history.html precisa carregar a propria ancora
    #history-content: Restaurar/Excluir/Atualizar usam hx-swap="outerHTML",
    e sem a ancora no fragmento o alvo some apos o primeiro swap."""
    from pathlib import Path as _Path

    frag = _Path("app/templates/_history.html").read_text()
    assert 'id="history-content"' in frag


def _seed_versions(session: Session, n: int) -> None:
    for _ in range(n):
        session.add(ConfigVersion(snapshot_json="{}"))
    session.commit()


def test_prune_history_keeps_the_last_n_and_deletes_older_ones(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _seed_versions(s, 7)
        r = c.post("/admin/config/history/prune", data={"keep": "3"}, cookies=COOKIE)
        assert r.status_code == 200
        with Session(app.state.recorder.engine) as s:
            restantes = s.execute(select(ConfigVersion).order_by(ConfigVersion.id)).scalars().all()
        assert len(restantes) == 3
        # Mantem as mais recentes (ids maiores), nao as mais antigas
        assert [v.id for v in restantes] == [5, 6, 7]


def test_prune_history_never_deletes_everything(monkeypatch, tmp_path):
    """keep=0 ou negativo nao esvazia o historico; floor e 1 versao."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _seed_versions(s, 3)
        r = c.post("/admin/config/history/prune", data={"keep": "0"}, cookies=COOKIE)
        assert r.status_code == 200
        with Session(app.state.recorder.engine) as s:
            assert len(s.execute(select(ConfigVersion)).scalars().all()) == 1


def test_delete_one_history_version_removes_only_it(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            _seed_versions(s, 3)
        r = c.delete("/admin/config/history/2", cookies=COOKIE)
        assert r.status_code == 200
        with Session(app.state.recorder.engine) as s:
            ids = [v.id for v in s.execute(select(ConfigVersion).order_by(ConfigVersion.id)).scalars().all()]
        assert ids == [1, 3]


def test_delete_unknown_history_version_is_404(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.delete("/admin/config/history/999", cookies=COOKIE)
        assert r.status_code == 404


def test_clearing_the_default_model_unsets_it_without_deleting_any_model(monkeypatch, tmp_path):
    """alias vazio = limpar o padrao, nao 404. Nenhum modelo e apagado e a
    config ao vivo passa a nao ter default_model."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p", base_url="http://u", protocol="openai"))
            s.commit()
            prov = s.execute(select(Provider)).scalar_one()
            s.add(
                Model(
                    alias="a",
                    provider_id=prov.id,
                    upstream_model="m",
                    context_window=128000,
                    max_output_tokens=8192,
                    is_default=True,
                )
            )
            s.commit()
        assert c.post("/admin/config/reload", cookies=COOKIE).status_code == 200
        assert app.state.settings.default_model == "a"

        r = c.post("/admin/config/default-model", data={"alias": ""}, cookies=COOKIE)

        assert r.status_code == 200
        with Session(app.state.recorder.engine) as s:
            assert s.execute(select(Model).where(Model.is_default == True)).scalars().all() == []
            # O modelo em si continua existindo
            assert s.execute(select(Model).where(Model.alias == "a")).scalar_one() is not None
        assert app.state.settings.default_model is None


def test_routes_fragment_balances_its_own_anchor_div():
    """O fragmento _routes.html abre `<div id="routes-list">` e tem de fecha-lo.

    O alvo dos botoes Editar/Excluir/Reordenar e esse div com hx-swap
    "outerHTML". Sem o fechamento, o navegador aninha tudo o que vem depois
    (o #route-form e as secoes Modelos, Padrao e Historico) DENTRO do alvo,
    e o primeiro swap os engole junto -- os botoes de toda a tela param de
    responder ate um refresh manual.
    """
    source = Path("app/templates/_routes.html").read_text()
    depth = 0
    for line in source.splitlines():
        depth += line.count("<div") - line.count("</div")
    assert depth == 0, f"_routes.html nao fecha todas as divs (depth {depth})"


def test_the_routes_list_anchor_does_not_swallow_the_edit_form(monkeypatch, tmp_path):
    """O ponto onde a div da ancora fecha e o que mantem a tela viva.

    O botao Editar carrega o formulario em #route-form; Salvar faz
    outerHTML em #routes-list. Se a ancora engolir #route-form, o swap o
    apaga, e o botao Nova Rota (e todo Editar da tabela) fica sem alvo.
    """
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            seed_db(s)
        page = c.get("/admin/config", cookies=COOKIE).text

    # O div da ancora abre antes da tabela e fecha ANTES do #route-form.
    ancora = page.index('id="routes-list"')
    formulario = page.index('id="route-form"')
    assert ancora < formulario
    # Conta as divs do ponto em que a ancora abre ate o formulario: o
    # balanceamento tem que voltar a zero, provando que a ancora fechou.
    fatia = page[ancora:formulario]
    depth = fatia.count("<div") - fatia.count("</div")
    assert depth == 0, f"a ancora de #routes-list engole #route-form (depth {depth})"


# ---- Effort por modelo ------------------------------------------------------


def test_model_config_defaults_to_no_effort():
    config = ModelConfig(provider="p", model="m", context_window=1, max_output_tokens=1)
    assert config.effort is None


def test_load_settings_from_db_carries_the_effort(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path):
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="p1", base_url="u", protocol="openai")
            s.add(p)
            s.flush()
            s.add(
                Model(
                    alias="com", provider_id=p.id, upstream_model="x",
                    context_window=1024, max_output_tokens=128, effort="high",
                )
            )
            s.add(
                Model(
                    alias="sem", provider_id=p.id, upstream_model="y",
                    context_window=1024, max_output_tokens=128,
                )
            )
            s.commit()
        with Session(app.state.recorder.engine) as s:
            settings = load_settings_from_db(s)
    assert settings.models["com"].effort == "high"
    assert settings.models["sem"].effort is None


EFFORT_ERROR = "Effort deve ser vazio ou um de: low, medium, high, xhigh"


@pytest.mark.parametrize("value", ["low", "medium", "high", "xhigh"])
def test_create_model_with_each_canonical_effort(monkeypatch, tmp_path, value):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="u", protocol="openai"))
            s.commit()
        r = c.post(
            "/admin/config/models",
            data={"alias": "m1", "provider_id": "1", "upstream_model": "x", "effort": value},
            cookies=COOKIE,
        )
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        m = s.execute(select(Model).where(Model.alias == "m1")).scalar_one()
    assert m.effort == value
    assert app.state.settings.models["m1"].effort == value


@pytest.mark.parametrize(
    "extra", [{}, {"effort": ""}, {"effort": "   "}], ids=["ausente", "vazio", "espacos"]
)
def test_create_model_without_effort_stores_null(monkeypatch, tmp_path, extra):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="u", protocol="openai"))
            s.commit()
        r = c.post(
            "/admin/config/models",
            data={"alias": "m1", "provider_id": "1", "upstream_model": "x", **extra},
            cookies=COOKIE,
        )
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        m = s.execute(select(Model).where(Model.alias == "m1")).scalar_one()
    assert m.effort is None
    assert app.state.settings.models["m1"].effort is None


def test_create_model_effort_accepts_surrounding_spaces(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="u", protocol="openai"))
            s.commit()
        r = c.post(
            "/admin/config/models",
            data={"alias": "m1", "provider_id": "1", "upstream_model": "x", "effort": " high "},
            cookies=COOKIE,
        )
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        assert s.execute(select(Model)).scalar_one().effort == "high"


@pytest.mark.parametrize("raw", ["max", "none", "minimal", "HIGH", "extreme", "0"])
def test_create_model_with_an_invalid_effort_is_refused(monkeypatch, tmp_path, raw):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            s.add(Provider(name="p1", base_url="u", protocol="openai"))
            s.commit()
        r = c.post(
            "/admin/config/models",
            data={"alias": "m1", "provider_id": "1", "upstream_model": "x", "effort": raw},
            cookies=COOKIE,
        )
    assert r.status_code == 400
    assert r.json()["detail"] == EFFORT_ERROR
    with Session(app.state.recorder.engine) as s:
        assert s.execute(select(Model).where(Model.alias == "m1")).scalar_one_or_none() is None
        # Nada gravado: nem o modelo, nem uma versao de config.
        assert s.execute(select(ConfigVersion)).scalars().all() == []


def test_patch_model_changes_then_clears_the_effort(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="p1", base_url="u", protocol="openai"); s.add(p); s.flush()
            s.add(
                Model(
                    alias="m1", provider_id=p.id, upstream_model="x",
                    context_window=1024, max_output_tokens=128, effort="low",
                )
            )
            s.commit()
        r = c.patch(
            "/admin/config/models/m1",
            data={"provider_id": "1", "upstream_model": "x", "effort": "xhigh"},
            cookies=COOKIE,
        )
        assert r.status_code == 200
        with Session(app.state.recorder.engine) as s:
            assert s.execute(select(Model)).scalar_one().effort == "xhigh"
        assert app.state.settings.models["m1"].effort == "xhigh"
        # Presente e vazio e o que o select "Usar o do harness" envia: limpa.
        r = c.patch(
            "/admin/config/models/m1",
            data={"provider_id": "1", "upstream_model": "x", "effort": ""},
            cookies=COOKIE,
        )
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        assert s.execute(select(Model)).scalar_one().effort is None
    assert app.state.settings.models["m1"].effort is None


def test_patch_model_that_omits_the_effort_keeps_it(monkeypatch, tmp_path):
    """Cliente direto que nao manda o campo nao apaga o effort -- mesmo
    contrato do limite de concorrencia do provedor."""
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="p1", base_url="u", protocol="openai"); s.add(p); s.flush()
            s.add(
                Model(
                    alias="m1", provider_id=p.id, upstream_model="x",
                    context_window=1024, max_output_tokens=128, effort="high",
                )
            )
            s.commit()
        r = c.patch(
            "/admin/config/models/m1",
            data={"provider_id": "1", "upstream_model": "y"},
            cookies=COOKIE,
        )
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        m = s.execute(select(Model)).scalar_one()
    assert (m.upstream_model, m.effort) == ("y", "high")


def test_patch_model_with_an_invalid_effort_is_refused_and_changes_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="p1", base_url="u", protocol="openai"); s.add(p); s.flush()
            s.add(
                Model(
                    alias="m1", provider_id=p.id, upstream_model="x",
                    context_window=1024, max_output_tokens=128, effort="low",
                )
            )
            s.commit()
        r = c.patch(
            "/admin/config/models/m1",
            data={"provider_id": "1", "upstream_model": "y", "effort": "max"},
            cookies=COOKIE,
        )
    assert r.status_code == 400
    assert r.json()["detail"] == EFFORT_ERROR
    with Session(app.state.recorder.engine) as s:
        m = s.execute(select(Model)).scalar_one()
    assert (m.upstream_model, m.effort) == ("x", "low")


def test_patch_unknown_model_with_an_invalid_effort_is_404(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        r = c.patch(
            "/admin/config/models/nao-existe",
            data={"provider_id": "1", "upstream_model": "x", "effort": "max"},
            cookies=COOKIE,
        )
    assert r.status_code == 404


def test_the_model_form_offers_the_effort_select(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="p1", base_url="u", protocol="openai"); s.add(p); s.flush()
            s.add(
                Model(
                    alias="m1", provider_id=p.id, upstream_model="x",
                    context_window=1024, max_output_tokens=128, effort="medium",
                )
            )
            s.commit()
        new = c.get("/admin/config/models/new", cookies=COOKIE)
        edit = c.get("/admin/config/models/m1/edit", cookies=COOKIE)
    assert '<select id="effort" name="effort"' in new.text
    assert '<option value="" selected>Vazio = usar o do harness</option>' in new.text
    for value in ("low", "medium", "high", "xhigh"):
        assert f'<option value="{value}">{value}</option>' in new.text
    assert '<option value="medium" selected>medium</option>' in edit.text
    assert '<option value="" selected>' not in edit.text


def test_the_models_list_shows_the_effort_or_harness(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="p1", base_url="u", protocol="openai"); s.add(p); s.flush()
            s.add(
                Model(
                    alias="com", provider_id=p.id, upstream_model="x",
                    context_window=1024, max_output_tokens=128, effort="high",
                )
            )
            s.add(
                Model(
                    alias="sem", provider_id=p.id, upstream_model="y",
                    context_window=1024, max_output_tokens=128,
                )
            )
            s.commit()
        r = c.get("/admin/config/models", cookies=COOKIE)
    assert "<th>Effort</th>" in r.text
    rows = {
        m.group(1): m.group(0)
        for m in re.finditer(r"<tr>\s*<td><b>(\w+)</b>.*?</tr>", r.text, re.DOTALL)
    }
    assert '<td class="mono effort" title="high">high</td>' in rows["com"]
    assert '<td class="mono effort" title="Usando o effort do harness"><span class="chip">—</span></td>' in rows["sem"]


def test_snapshot_and_rollback_carry_the_effort(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="p1", base_url="u", protocol="openai"); s.add(p); s.flush()
            s.add(
                Model(
                    alias="m1", provider_id=p.id, upstream_model="x",
                    context_window=1024, max_output_tokens=128, effort="high",
                )
            )
            s.commit()
            cv = create_config_version(s)
            cv_id = cv.id
            assert json.loads(cv.snapshot_json)["models"][0]["effort"] == "high"
        r = c.patch(
            "/admin/config/models/m1",
            data={"provider_id": "1", "upstream_model": "x", "effort": "low"},
            cookies=COOKIE,
        )
        assert r.status_code == 200
        r = c.post(f"/admin/config/rollback/{cv_id}", cookies=COOKIE)
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        assert s.execute(select(Model)).scalar_one().effort == "high"
    assert app.state.settings.models["m1"].effort == "high"


def test_rollback_of_an_old_snapshot_without_the_effort_restores_null(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "t")
    with client(tmp_path) as c:
        with Session(app.state.recorder.engine) as s:
            p = Provider(name="p1", base_url="u", protocol="openai"); s.add(p); s.flush()
            s.add(
                Model(
                    alias="m1", provider_id=p.id, upstream_model="x",
                    context_window=1024, max_output_tokens=128, effort="high",
                )
            )
            old = {
                "providers": [
                    {"name": "p1", "base_url": "u", "protocol": "openai", "api_key": False}
                ],
                "models": [
                    {
                        "alias": "m1", "provider": "p1", "upstream_model": "x",
                        "supports_tools": True, "supports_streaming": True,
                        "supports_vision": False, "context_window": 1024,
                        "max_output_tokens": 128, "is_default": False,
                    }
                ],
                "routes": [],
                "default_model": None,
            }
            cv = ConfigVersion(snapshot_json=json.dumps(old))
            s.add(cv)
            s.commit()
            cv_id = cv.id
        r = c.post(f"/admin/config/rollback/{cv_id}", cookies=COOKIE)
    assert r.status_code == 200
    with Session(app.state.recorder.engine) as s:
        assert s.execute(select(Model)).scalar_one().effort is None
    assert app.state.settings.models["m1"].effort is None
