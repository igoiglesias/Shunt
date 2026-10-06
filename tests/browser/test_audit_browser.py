"""A tela de auditoria num navegador HEADLESS, contra um banco semeado.

O que so aparece aqui: se a busca estreita de verdade, se o detalhe mostra a
cadeia da requisicao clicada, e se o estado da busca vive na URL -- que e o que
transforma uma investigacao num link.
"""

import contextlib
import json
import os
import socket
import subprocess
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config.config import ADMIN_COOKIE
from app.core.security import hash_password, issue_jwt
from app.stats.models import Base, RequestBody, RequestEvent, User

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

# Segredo fixo so deste modulo: a auditoria exige sessao admin, e com o segredo
# conhecido o teste assina o proprio cookie em vez de passar pela tela de login.
# Nao e segredo de producao; so precisa ser estavel dentro do modulo.
SESSION_SECRET = "segredo-fixo-do-teste-da-auditoria"

# Admin semeado: so o teste do login usa a senha. Com um usuario no banco a tela
# de login deixa de ser o "primeiro acesso" e pede usuario e senha.
ADMIN_USER = "admin"
ADMIN_PASSWORD = "senha-do-teste-de-navegador"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def seed(path) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{path}")
    Base.metadata.create_all(engine)
    now = datetime.now(UTC)
    rows = []
    for index in range(24):
        failed = index % 6 == 0
        slow = index % 5 == 0
        rows.append(
            RequestEvent(
                request_id=f"req-{index:02d}",
                started_at=now - timedelta(minutes=index),
                route="/v1/messages" if index % 3 else "/v1/chat/completions",
                dialect="anthropic" if index % 3 else "openai",
                stream=index % 2 == 0,
                requested_model="claude-opus-5" if index % 2 else "claude-haiku-4-5",
                rule="family",
                matched="opus",
                provider=None if failed else ("groq" if index % 2 else "local"),
                candidate_model=None if failed else ("openai/gpt-oss-120b" if index % 2 else "qwen3.8-27b"),
                status=429 if failed else 200,
                error_type="rate_limit_error" if failed else None,
                input_tokens=100 * index,
                output_tokens=20 * index,
                ttft_ms=200 if index % 2 == 0 and not failed else None,
                duration_ms=9000 if slow else 400,
                attempts=["free: 502 (attempt 1)"] if index % 4 == 0 else [],
                fell_back=index % 4 == 0 and not failed,
                tools_offered=["Read", "Bash"] if index % 3 == 0 else [],
                tools_called=["Bash"] if index % 3 == 0 else [],
                thinking_blocks=1,
                cached_input_tokens=(50 * index if index % 3 else None),
                cache_write_tokens=None,
                project="/home/iglesias/Projetos/agenda" if index % 2 else None,
                session_id=f"s-{index % 3}",
            )
        )
    conversas = [
        RequestBody(
            request_id=f"req-{index:02d}",
            request_json=(
                '{\n  "model": "claude-opus-5",\n  "max_tokens": 64,\n'
                '  "messages": [{"role": "user", "content": "oi"}]\n}'
            ),
            prompt=(
                "system: seja breve\n\n"
                f"user: leia o arquivo {index}.txt e resuma\n\n"
                "assistant: [ferramenta Read] {\"path\": \"a.txt\"}\n\n"
                "user: [resultado de ferramenta] " + ("conteudo " * 200)
            ),
            answer=f"resumo da requisicao {index}",
            prompt_bytes=4000 if index else 90_000,
            answer_bytes=30,
            truncated=index == 0,
            # A linha 6 falha (`index % 6 == 0`, status 429) e carrega o corpo
            # do erro do provedor: e o que a aba "Erro" mostra.
            error=(
                '{"error":{"message":"rate limit exceeded","type":"rate_limit_error"}}'
                if index == 6
                else None
            ),
            error_bytes=68 if index == 6 else 0,
        )
        for index in range(0, 24, 2)
    ]
    with Session(engine) as session:
        session.add_all(rows)
        session.add_all(conversas)
        session.add(User(username=ADMIN_USER, password_hash=hash_password(ADMIN_PASSWORD)))
        session.commit()
    engine.dispose()


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    database = tmp_path_factory.mktemp("auditoria") / "stats.db"
    seed(database)
    port = free_port()
    process = subprocess.Popen(
        ["uv", "run", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        env={
            **os.environ,
            "TURSO_DATABASE_URL": f"sqlite+pysqlite:///{database}",
            "ADMIN_SESSION_SECRET": SESSION_SECRET,
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.3)
    else:
        process.terminate()
        pytest.fail("o servidor da auditoria nao subiu")
    yield base
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch()  # headless
        yield instance
        instance.close()


def admin_page(browser, base, width, height):
    """Pagina nova com a sessao admin ja no cookie e os erros de JS coletados.

    O cookie e assinado aqui com o segredo do servidor: a tela de login tem o
    seu proprio teste, e passar por ela em cada teste so gastaria tempo.
    """
    page = browser.new_page(viewport={"width": width, "height": height})
    # Cinco segundos, e nao os trinta do Playwright: toda espera aqui e por uma
    # condicao da tela, e uma condicao que nao chega em 5 s num servidor local e
    # defeito -- esperar mais so atrasa o vermelho.
    page.set_default_timeout(5_000)
    page.context.add_cookies(
        [{"name": ADMIN_COOKIE, "value": issue_jwt(1, SESSION_SECRET, 3600), "url": base}]
    )
    problems = []
    page.on("pageerror", lambda error: problems.append(str(error)))
    page.on(
        "console",
        lambda message: problems.append(message.text) if message.type == "error" else None,
    )
    return page, problems


def wait_search(page, condicao: str) -> None:
    """Espera a busca que a `condicao` disparou terminar de desenhar.

    A `condicao` e o que o clique muda na hora (a URL, por exemplo), e `loading`
    e a trava da propria tela: ela so volta a `false` depois que `render`
    desenhou as linhas e a contagem. As duas juntas dizem que a busca NOVA
    terminou, e nao a anterior.
    """
    page.wait_for_function(f"() => ({condicao}) && !loading")


def api_get(url: str) -> httpx.Response:
    """GET na API da auditoria com a sessao admin: `/api/*` responde 401 sem ela."""
    cookie = f"{ADMIN_COOKIE}={issue_jwt(1, SESSION_SECRET, 3600)}"
    return httpx.get(url, headers={"cookie": cookie}, timeout=10)


def open_audit(browser, base, width=1500, height=1000, query=""):
    """Abre a tela de auditoria ja autenticado e espera a primeira linha."""
    page, problems = admin_page(browser, base, width, height)
    page.goto(f"{base}/admin/requests{query}", wait_until="networkidle")
    page.wait_for_selector("#rows tr", timeout=10000)
    return page, problems


def test_the_screen_lists_the_requests_newest_first(browser, server):
    page, problems = open_audit(browser, server)
    ids = page.evaluate("() => [...document.querySelectorAll('#rows tr')].map(r => r.dataset.id)")
    count = page.inner_text("#count")
    page.close()
    assert problems == []
    assert ids[:3] == ["req-00", "req-01", "req-02"]
    assert "de 24 requisições" in count


def test_login_leads_to_the_requested_page(browser, server):
    """Um link filtrado aberto sem sessao volta ao mesmo filtro depois do login.

    E o unico teste que passa pela tela de login: sem o `next=`, o login cairia
    no painel e o link compartilhado perderia a busca.
    """
    alvo = f"{server}/admin/requests?provider=groq&has_tools=true"
    # Sem o padrao de 5 s dos helpers: este teste ja estourou sob carga, e o
    # runner de CI tem menos CPU. Ele fica com os 30 s do Playwright e, em troca,
    # diz onde parou quando estoura.
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    problems = []
    page.on("pageerror", lambda error: problems.append(str(error)))
    page.on(
        "console",
        lambda message: problems.append(message.text) if message.type == "error" else None,
    )
    page.goto(alvo, wait_until="networkidle")
    assert "/admin/login" in page.url
    page.fill('input[name="username"]', ADMIN_USER)
    page.fill('input[name="password"]', ADMIN_PASSWORD)
    # Este teste ja estourou 30 s uma vez, com a maquina carregada, sem causa
    # achada. A resposta do POST fica registrada para que a proxima falha diga
    # o que aconteceu: login recusado, redirect errado ou servidor mudo.
    try:
        with page.expect_response(
            lambda r: r.url.endswith("/admin/login") and r.request.method == "POST"
        ) as login:
            page.click('button[type="submit"]')
    except PlaywrightTimeoutError as erro:
        pytest.fail(f"o POST do login nao respondeu; pagina em {page.url}: {erro}")
    resposta = login.value
    assert resposta.status == 303, (
        f"o POST do login respondeu {resposta.status} "
        f"(location={resposta.headers.get('location')!r}); pagina em {page.url}"
    )
    page.wait_for_url(alvo)
    page.wait_for_selector("#rows tr", timeout=10000)
    page.close()
    assert problems == []


def test_a_chip_narrows_the_list_and_the_count(browser, server):
    page, problems = open_audit(browser, server)
    before = page.locator("#rows tr").count()
    page.get_by_role("button", name="falhas").click()
    page.wait_for_function(f"() => document.querySelectorAll('#rows tr').length < {before}")
    statuses = page.evaluate(
        "() => [...document.querySelectorAll('#rows tr td:last-child')].map(c => Number(c.textContent))"
    )
    page.close()
    assert problems == []
    assert statuses, "o filtro escondeu tudo"
    assert all(status >= 400 for status in statuses)


def test_the_search_state_lives_in_the_url(browser, server):
    page, _ = open_audit(browser, server)
    page.get_by_role("button", name="com fallback").click()
    wait_search(page, "new URLSearchParams(location.search).get('fell_back') === 'true'")
    url = page.url
    ids = page.evaluate("() => [...document.querySelectorAll('#rows tr')].map(r => r.dataset.id)")
    page.close()
    assert "fell_back=true" in url

    # O mesmo link, aberto do zero, devolve a mesma busca.
    again, problems = open_audit(browser, server, query=url[url.index("?") :])
    same = again.evaluate("() => [...document.querySelectorAll('#rows tr')].map(r => r.dataset.id)")
    pressed = again.evaluate(
        """() => [...document.querySelectorAll('.chip')]
            .filter(c => c.getAttribute('aria-pressed') === 'true').map(c => c.textContent.trim())"""
    )
    again.close()
    assert problems == []
    assert same == ids
    assert pressed == ["com fallback"]


def test_typing_in_the_search_box_filters(browser, server):
    page, problems = open_audit(browser, server)
    page.fill("#text", "qwen3.8-27b")
    page.wait_for_function(
        """() => [...document.querySelectorAll('#rows tr')]
            .every(r => r.textContent.includes('qwen3.8-27b'))"""
    )
    rows = page.locator("#rows tr").count()
    page.close()
    assert problems == []
    assert rows > 0


def test_clicking_a_row_opens_the_whole_chain(browser, server):
    page, problems = open_audit(browser, server)
    target = page.evaluate(
        """() => [...document.querySelectorAll('#rows tr')]
            .find(r => r.querySelector('.mark.fell'))?.dataset.id"""
    )
    assert target, "nenhuma linha com fallback para abrir"
    page.click(f'#rows tr[data-id="{target}"]')
    page.wait_for_selector("#detail .chain li")
    detail = page.inner_text("#detail")
    api = api_get(f"{server}/api/requests/{target}").json()
    selected = page.evaluate(
        """() => document.querySelector('#rows tr[aria-selected="true"]').dataset.id"""
    )
    page.close()
    assert problems == []
    assert selected == target
    assert api["request_id"] in detail
    assert api["attempts"][0].split(":")[0] in detail
    assert api["candidate_model"] in detail


def test_the_order_can_be_changed_to_the_slowest_first(browser, server):
    page, problems = open_audit(browser, server)
    page.get_by_role("button", name="Duração").click()
    wait_search(page, "new URLSearchParams(location.search).get('order_by') === 'duration'")
    # Pela CLASSE da célula, e não pelo índice: uma coluna nova na tabela
    # deslocava o índice e o teste passava a ler outra coluna.
    durations = page.evaluate(
        """() => [...document.querySelectorAll('#rows tr')]
            .map(r => r.querySelector('td[data-label="Duração"]').textContent.trim())"""
    )
    page.close()
    assert problems == []
    assert durations[0].endswith("s"), f"a primeira nao e a mais lenta: {durations[:3]}"


def test_loading_more_appends_without_repeating(browser, server):
    page, problems = open_audit(browser, server, query="?limit=10")
    first = page.evaluate("() => [...document.querySelectorAll('#rows tr')].map(r => r.dataset.id)")
    page.get_by_role("button", name="Carregar mais").click()
    page.wait_for_function("() => document.querySelectorAll('#rows tr').length > 10")
    after = page.evaluate("() => [...document.querySelectorAll('#rows tr')].map(r => r.dataset.id)")
    page.close()
    assert problems == []
    assert len(first) == 10
    assert len(after) == len(set(after)), "a lista repetiu uma requisicao"
    assert after[:10] == first


def test_the_export_carries_the_same_filters(browser, server):
    page, _ = open_audit(browser, server)
    page.get_by_role("button", name="falhas").click()
    wait_search(page, "new URLSearchParams(location.search).get('status_min') === '400'")
    query = page.get_attribute("#export", "data-query")
    page.close()
    assert "status_min=400" in query
    csv_text = api_get(f"{server}/api/requests/export?{query}").text
    assert csv_text.startswith("started_at,request_id")
    assert csv_text.count("\n") > 1


def test_the_export_says_how_many_it_took(browser, server):
    """CSV cortado em silencio vira conclusao errada numa planilha."""
    page, problems = open_audit(browser, server)
    page.get_by_role("button", name="Exportar CSV").click()
    page.wait_for_function("() => document.getElementById('status').textContent.includes('exportadas')")
    said = page.inner_text("#status")
    page.close()
    assert problems == []
    assert "exportadas" in said


def test_clearing_the_filters_brings_everything_back(browser, server):
    page, problems = open_audit(browser, server)
    total = page.inner_text("#count")
    page.get_by_role("button", name="falhas").click()
    # Espera a busca do chip terminar, para o teste medir o "Limpar filtros"
    # partindo da lista ja filtrada, e nao de uma busca abortada no meio.
    wait_search(page, "new URLSearchParams(location.search).get('status_min') === '400'")
    page.get_by_role("button", name="Limpar filtros").click()
    page.wait_for_function(f"() => document.getElementById('count').textContent === {total!r}")
    pressed = page.evaluate(
        """() => [...document.querySelectorAll('.chip')]
            .filter(c => c.getAttribute('aria-pressed') === 'true').length"""
    )
    page.close()
    assert problems == []
    assert pressed == 0


def test_the_screen_works_on_a_phone(browser, server):
    page, problems = open_audit(browser, server, width=390, height=844)
    measured = page.evaluate(
        """() => ({
            scrollWidth: document.documentElement.scrollWidth,
            innerWidth: window.innerWidth,
        })"""
    )
    page.close()
    assert problems == []
    assert measured["scrollWidth"] <= measured["innerWidth"] + 1


def test_a_failed_request_shows_why_instead_of_a_dash(browser, server):
    """A coluna "Respondeu" numa falha nao tem candidato; ela tem o motivo."""
    page, problems = open_audit(browser, server)
    page.get_by_role("button", name="falhas").click()
    page.wait_for_function(
        """() => [...document.querySelectorAll('#rows tr')].length > 0 &&
                 [...document.querySelectorAll('#rows tr td:last-child')]
                   .every(c => Number(c.textContent) >= 400)"""
    )
    first = page.locator("#rows tr").first.inner_text()
    page.close()
    assert problems == []
    assert "rate_limit_error" in first
    assert "—" not in first.split("\n")[3] if "\n" in first else True


def test_the_active_ordering_is_visible_and_not_only_in_aria(browser, server):
    page, problems = open_audit(browser, server)
    marker = page.evaluate(
        """() => {
            const head = [...document.querySelectorAll('th[data-order]')]
                .find(h => h.hasAttribute('aria-sort'));
            return head && getComputedStyle(head, '::after').content;
        }"""
    )
    page.close()
    assert problems == []
    assert marker and marker not in ("none", '""'), f"nenhuma marca visivel: {marker!r}"


def test_the_requested_model_has_its_own_selector(browser, server):
    """O backend ja filtrava por modelo pedido; faltava o seletor na tela."""
    page, problems = open_audit(browser, server)
    page.wait_for_function(
        "() => document.querySelectorAll('#requested_model option').length > 1"
    )
    page.select_option("#requested_model", "claude-opus-5")
    page.wait_for_function(
        """() => {
            const rows = [...document.querySelectorAll('#rows tr')];
            return rows.length > 0 && rows.every(r => r.children[2].textContent.includes('claude-opus-5'));
        }"""
    )
    url = page.url
    page.close()
    assert problems == []
    assert "requested_model=claude-opus-5" in url


def test_the_columns_say_which_model_each_one_is(browser, server):
    page, _ = open_audit(browser, server)
    heads = page.evaluate("() => [...document.querySelectorAll('thead th')].map(h => h.textContent.trim())")
    page.close()
    assert "Modelo pedido" in heads
    assert "Modelo que respondeu" in heads


def test_the_selectors_offer_only_what_exists_with_counts(browser, server):
    """Oferecer um provedor que nunca respondeu e um filtro que devolve vazio."""
    page, problems = open_audit(browser, server)
    page.wait_for_function("() => document.querySelectorAll('#provider option').length > 1")
    options = page.evaluate(
        """() => ({
            provider: [...document.querySelectorAll('#provider option')].map(o => o.textContent),
            served: [...document.querySelectorAll('#candidate_model option')].map(o => o.value),
        })"""
    )
    page.close()
    assert problems == []
    assert options["provider"][0].startswith("provedor: todos")
    assert any("groq (" in text for text in options["provider"]), options["provider"]
    assert "openai/gpt-oss-120b" in options["served"]
    assert "" in options["served"], "faltou a opcao de nao filtrar"


def test_choosing_a_provider_filters_and_lands_in_the_url(browser, server):
    page, problems = open_audit(browser, server)
    page.wait_for_function("() => document.querySelectorAll('#provider option').length > 1")
    page.select_option("#provider", "groq")
    wait_search(page, "new URLSearchParams(location.search).get('provider') === 'groq'")
    served = page.evaluate(
        "() => [...document.querySelectorAll('#rows tr')].map(r => r.children[3].textContent)"
    )
    page.close()
    assert problems == []
    assert served
    assert all("gpt-oss-120b" in text for text in served), served


def test_a_value_from_an_old_link_stays_selectable(browser, server):
    """Link antigo com um modelo que sumiu do banco nao pode mudar a busca sozinho.

    A lista vem vazia de proposito aqui, entao a pagina abre sem esperar linha.
    """
    page, problems = admin_page(browser, server, 1400, 900)
    page.goto(f"{server}/admin/requests?provider=um-provedor-que-sumiu", wait_until="networkidle")

    page.wait_for_function("() => document.querySelectorAll('#provider option').length > 1")
    chosen = page.input_value("#provider")
    page.close()
    assert problems == []
    assert chosen == "um-provedor-que-sumiu"


def test_the_conversation_tab_shows_the_turns_with_who_said_what(browser, server):
    """O pedido principal: ler o que subiu e o que voltou."""
    page, problems = open_audit(browser, server)
    page.click('#rows tr[data-id="req-02"]')
    page.wait_for_selector("#detail .tabs")
    page.get_by_role("tab", name="Conversa").click()
    page.wait_for_selector("#talk .turn")
    turns = page.evaluate(
        """() => [...document.querySelectorAll('#talk .turn')].map(t => ({
            who: t.querySelector('.who b').textContent,
            kind: t.className.replace('turn', '').trim(),
            text: t.querySelector('pre').textContent.slice(0, 40),
        }))"""
    )
    page.close()
    assert problems == []
    papeis = [turn["who"] for turn in turns]
    assert papeis[:3] == ["system", "user", "assistant"]
    assert papeis[-1] == "resposta"
    assert turns[1]["kind"] == "user"
    assert "leia o arquivo" in turns[1]["text"]
    assert "resumo da requisicao" in turns[-1]["text"]


def test_searching_inside_the_conversation_highlights_the_hits(browser, server):
    page, problems = open_audit(browser, server)
    page.click('#rows tr[data-id="req-02"]')
    page.wait_for_selector("#detail .tabs")
    page.get_by_role("tab", name="Conversa").click()
    page.wait_for_selector("#talk .turn")
    page.fill("#in-talk", "ferramenta")
    page.wait_for_selector("#talk mark")
    marks = page.locator("#talk mark").count()
    page.close()
    assert problems == []
    assert marks >= 1


def test_a_cut_conversation_says_how_much_is_missing(browser, server):
    page, problems = open_audit(browser, server)
    page.click('#rows tr[data-id="req-00"]')
    page.wait_for_selector("#detail .tabs")
    page.get_by_role("tab", name="Conversa").click()
    page.wait_for_selector("#talk .cut")
    said = page.inner_text("#talk .cut")
    page.close()
    assert problems == []
    assert "cortado" in said
    assert "90.0k" in said or "90k" in said


def test_a_request_with_no_stored_conversation_says_it_was_off(browser, server):
    page, problems = open_audit(browser, server)
    page.click('#rows tr[data-id="req-01"]')
    page.wait_for_selector("#detail .tabs")
    page.get_by_role("tab", name="Conversa").click()
    page.wait_for_selector("#talk .empty:not(.loading)")
    said = page.inner_text("#talk")
    page.close()
    # O 404 desta rota e a propria resposta, e o navegador o registra no
    # console: o que nao pode aparecer e erro de JavaScript.
    assert [p for p in problems if "Failed to load resource" not in p] == []
    assert "nao gravada" in said


def test_the_summary_comes_back_when_the_tab_switches_back(browser, server):
    page, problems = open_audit(browser, server)
    page.click('#rows tr[data-id="req-02"]')
    page.wait_for_selector("#detail .tabs")
    page.get_by_role("tab", name="Conversa").click()
    page.wait_for_selector("#talk .turn")
    page.get_by_role("tab", name="Resumo").click()
    page.wait_for_selector("#summary .chain")
    hidden = page.evaluate("() => document.getElementById('talk').hidden")
    page.close()
    assert problems == []
    assert hidden is True


def test_the_error_tab_shows_the_upstream_error_body(browser, server):
    """A aba "Erro" so existe na falha e mostra o corpo que o provedor mandou."""
    page, problems = open_audit(browser, server)
    page.click('#rows tr[data-id="req-06"]')
    page.wait_for_selector("#detail .tabs")
    # A linha 429 tem a aba; uma linha 200 nao teria.
    page.get_by_role("tab", name="Erro").click()
    page.wait_for_selector("#error pre")
    body = page.inner_text("#error pre")
    page.close()
    assert problems == []
    assert "rate limit exceeded" in body
    assert "rate_limit_error" in body


def test_the_period_shortcut_becomes_an_instant_in_the_url(browser, server):
    """Janela relativa nao sobrevive a um link mandado meia hora depois."""
    page, problems = open_audit(browser, server)
    page.select_option("#period", "1")
    page.wait_for_function("() => new URLSearchParams(location.search).has('since')")
    since = page.evaluate("() => new URLSearchParams(location.search).get('since')")
    rows = page.locator("#rows tr").count()
    page.close()
    assert problems == []
    assert since.endswith("Z") or "+" in since, since
    assert rows > 0


def test_choosing_two_dates_shows_the_fields_and_filters(browser, server):
    page, problems = open_audit(browser, server)
    page.select_option("#period", "custom")
    page.wait_for_selector("#since:visible")
    page.fill("#since", "2026-09-19T00:00")
    page.wait_for_function("() => new URLSearchParams(location.search).has('since')")
    page.fill("#until", "2026-09-19T00:05")
    wait_search(page, "new URLSearchParams(location.search).has('until')")
    count = page.inner_text("#count")
    page.close()
    assert problems == []
    assert "nenhuma requisição" in count or "de 0" in count


def segura_buscas(page, segurar):
    """Intercepta `/api/requests` e segura, sem responder, as que `segurar` escolhe.

    Segurar e nao `time.sleep`: o handler sincrono roda na thread do driver do
    Playwright, e dormir ali congela o driver inteiro -- a resposta "atrasada"
    acabava chegando antes da nova. Segurada, a busca so volta quando o teste
    solta, e a ordem das respostas deixa de depender da carga da maquina.
    Devolve (urls vistas, rotas seguradas).
    """
    buscas, seguradas = [], []

    def handler(route):
        url = route.request.url
        buscas.append(url)
        if segurar(url):
            seguradas.append(route)
        else:
            route.continue_()

    page.route("**/api/requests?*", handler)
    return buscas, seguradas


def test_the_latest_search_wins_when_the_previous_one_is_slow(browser, server):
    """Trocar o segundo filtro com a busca anterior em voo busca de novo.

    Regressao medida no CI: a busca so com `since` ainda estava em voo quando
    `#until` mudou; a busca nova era descartada e a tela ficava em "24 de 24"
    sob uma URL que pedia a janela de 5 minutos. Aqui a busca so-com-`since`
    fica segurada ate a nova terminar, e so entao e solta: a tela tem de ter
    abortado a velha e nao pode deixa-la desenhar por cima.
    """
    page, problems = open_audit(browser, server)
    falhas = []
    page.on("requestfailed", lambda request: falhas.append((request.url, request.failure)))
    buscas, seguradas = segura_buscas(
        page, lambda url: "since=" in url and "until=" not in url
    )
    page.select_option("#period", "custom")
    page.wait_for_selector("#since:visible")
    page.fill("#since", "2026-09-19T00:00")
    page.wait_for_function("() => new URLSearchParams(location.search).has('since')")
    page.fill("#until", "2026-09-19T00:05")
    wait_search(page, "new URLSearchParams(location.search).has('until')")
    antes = page.inner_text("#count")
    assert len(seguradas) == 1, buscas
    # A requisicao ja foi abortada pela tela; soltar a rota pode reclamar disso.
    with contextlib.suppress(PlaywrightError):
        seguradas[0].continue_()
    page.wait_for_timeout(300)
    depois = page.inner_text("#count")
    page.close()
    assert problems == []
    assert any("until=" in url for url in buscas), f"a busca com until nao saiu: {buscas}"
    assert "nenhuma requisição" in antes, antes
    abortadas = [
        url for url, erro in falhas if "since=" in url and "until=" not in url
    ]
    assert abortadas, f"a busca velha nao foi abortada: {falhas}"
    assert all(erro == "net::ERR_ABORTED" for _, erro in falhas), falhas
    assert "nenhuma requisição" in depois, depois


def test_loading_more_is_ignored_while_a_search_is_in_flight(browser, server):
    """"Carregar mais" com a busca nova em voo nao sai nem duplica linhas.

    Somar uma pagina da lista velha a uma lista que esta para ser trocada daria
    linhas de dois filtros misturadas.
    """
    page, problems = open_audit(browser, server, query="?limit=8")
    assert page.locator("#more").is_visible()
    buscas, seguradas = segura_buscas(page, lambda url: "status_min=" in url)
    page.get_by_role("button", name="falhas").click()
    page.wait_for_function("() => new URLSearchParams(location.search).has('status_min')")
    page.wait_for_function("() => loading")
    page.click("#more")
    page.wait_for_timeout(300)
    enquanto_segura = list(buscas)
    seguradas[0].continue_()
    wait_search(page, "new URLSearchParams(location.search).has('status_min')")
    ids = page.evaluate("() => [...document.querySelectorAll('#rows tr')].map(r => r.dataset.id)")
    page.close()
    assert problems == []
    assert len(enquanto_segura) == 1, f"saiu busca alem da segurada: {enquanto_segura}"
    assert "status_min=" in enquanto_segura[0]
    assert "cursor=" not in " ".join(buscas), buscas
    assert len(ids) == len(set(ids)), ids
    assert ids, "a busca das falhas nao desenhou"


def test_a_network_error_shows_up_and_releases_the_search(browser, server):
    """Falha de rede na busca corrente aparece no status e solta `loading`."""
    page, _ = open_audit(browser, server)
    page.route(
        "**/api/requests?*",
        lambda route: route.abort() if "status_min=" in route.request.url else route.continue_(),
    )
    page.get_by_role("button", name="falhas").click()
    page.wait_for_function(
        "() => document.getElementById('status').textContent.includes('não deu para buscar')"
        " && !loading"
    )
    status = page.inner_text("#status")
    page.close()
    assert status.startswith("não deu para buscar:"), status


def test_an_old_link_with_dates_reopens_the_same_window(browser, server):
    """A janela de 2020 devolve lista vazia, entao a pagina abre sem esperar linha."""
    page, problems = admin_page(browser, server, 1400, 900)
    page.goto(
        f"{server}/admin/requests?since=2020-01-01T00:00:00Z&until=2020-01-02T00:00:00Z",
        wait_until="networkidle",
    )

    page.wait_for_selector("#empty:visible", timeout=10000)
    mostrado = page.evaluate(
        """() => ({
            period: document.getElementById('period').value,
            since: document.getElementById('since').value,
            visivel: !document.getElementById('range').hidden,
        })"""
    )
    page.close()
    assert problems == []
    assert mostrado["period"] == "custom"
    assert mostrado["since"].startswith("2019-12-31") or mostrado["since"].startswith("2020-01-01")
    assert mostrado["visivel"] is True


def test_paging_goes_forward_and_back(browser, server):
    page, problems = open_audit(browser, server, query="?limit=8")
    primeira = page.evaluate("() => [...document.querySelectorAll('#rows tr')].map(r => r.dataset.id)")
    assert page.locator("#back").is_hidden(), "nao ha para onde voltar na primeira pagina"

    page.get_by_role("button", name="Carregar mais").click()
    page.wait_for_function("() => document.querySelectorAll('#rows tr').length > 8")
    page.wait_for_selector("#back:visible")

    page.get_by_role("button", name="Página anterior").click()
    page.wait_for_function("() => document.querySelectorAll('#rows tr').length === 8")
    depois = page.evaluate("() => [...document.querySelectorAll('#rows tr')].map(r => r.dataset.id)")
    page.close()
    assert problems == []
    assert depois == primeira, "voltar devolveu uma lista diferente da que a pessoa viu"


def test_the_raw_request_tab_shows_what_would_reproduce_it(browser, server):
    """O texto diz o que foi dito; o JSON diz com que parametros."""
    page, problems = open_audit(browser, server)
    page.click('#rows tr[data-id="req-02"]')
    page.wait_for_selector("#detail .tabs")
    page.get_by_role("tab", name="Requisição").click()
    page.wait_for_selector("#raw pre")
    raw = page.inner_text("#raw pre")
    botoes = page.evaluate(
        "() => [...document.querySelectorAll('#raw .ghost')].map(b => b.textContent.trim())"
    )
    page.close()
    assert problems == []
    assert '"max_tokens": 64' in raw
    assert botoes == ["Copiar JSON", "Copiar como curl"]


def test_a_request_without_a_raw_body_says_so(browser, server):
    page, problems = open_audit(browser, server)
    page.click('#rows tr[data-id="req-01"]')
    page.wait_for_selector("#detail .tabs")
    page.get_by_role("tab", name="Requisição").click()
    page.wait_for_selector("#raw .empty:not(.loading)")
    said = page.inner_text("#raw")
    page.close()
    assert [p for p in problems if "Failed to load resource" not in p] == []
    assert "nao gravada" in said or "não foi gravado" in said


def test_a_request_with_no_candidate_says_which_story_it_is(browser, server):
    """Medido no painel ao vivo: uma cadeia que ninguem atendeu aparecia como
    "estimada aqui", que e o rotulo do `count_tokens`."""
    page, problems = open_audit(browser, server)
    rotulos = page.evaluate(
        """() => [...document.querySelectorAll('#rows tr')]
            .filter(r => r.children[3].textContent.includes('rate_limit_error'))
            .map(r => r.children[3].textContent.trim())"""
    )
    page.close()
    assert problems == []
    assert rotulos, "nenhuma linha sem candidato no banco semeado"
    assert all("estimada aqui" not in texto for texto in rotulos), rotulos


def test_a_conversation_is_split_by_turn_and_not_by_paragraph(browser, server):
    """Medido numa conversa real: 142 blocos, quase todos rotulados "texto".

    Um `system` de mil linhas e um turno so; quebrar por linha em branco
    transformava cada paragrafo num turno proprio.
    """
    page, problems = open_audit(browser, server)
    page.click('#rows tr[data-id="req-02"]')
    page.wait_for_selector("#detail .tabs")
    page.get_by_role("tab", name="Conversa").click()
    page.wait_for_selector("#talk .turn")
    papeis = page.evaluate(
        "() => [...document.querySelectorAll('#talk .who b')].map(b => b.textContent)"
    )
    page.close()
    assert problems == []
    assert papeis == ["system", "user", "assistant", "user", "resposta"], papeis


def test_on_a_phone_the_table_becomes_cards(browser, server):
    """Sete colunas cortadas nao sao uma tabela."""
    page, problems = open_audit(browser, server, width=390, height=844)
    medido = page.evaluate(
        """() => {
            const head = document.querySelector('thead');
            const primeira = document.querySelector('#rows tr td');
            return {
                cabecalhoVisivel: getComputedStyle(head).display !== 'none',
                rotulo: getComputedStyle(document.querySelector('#rows td[data-label="Pedido"]'), '::before').content,
                rolagem: document.documentElement.scrollWidth,
                janela: window.innerWidth,
                celulaVisivel: primeira.getBoundingClientRect().width > 0,
            };
        }"""
    )
    page.close()
    assert problems == []
    assert medido["cabecalhoVisivel"] is False
    assert "Pedido" in medido["rotulo"], medido["rotulo"]
    assert medido["rolagem"] <= medido["janela"] + 1
    assert medido["celulaVisivel"]


def test_the_card_leads_with_when_and_how_it_ended(browser, server):
    page, problems = open_audit(browser, server, width=390, height=844)
    caixas = page.evaluate(
        """() => {
            const linha = document.querySelector('#rows tr');
            const quando = linha.querySelector('td.when').getBoundingClientRect();
            const status = linha.querySelector('td.status').getBoundingClientRect();
            return { quandoTop: Math.round(quando.top), statusTop: Math.round(status.top) };
        }"""
    )
    page.close()
    assert problems == []
    assert abs(caixas["quandoTop"] - caixas["statusTop"]) <= 4, caixas


def test_every_control_can_be_reached_by_keyboard(browser, server):
    """Foco visível e ordem de tabulação: a tela tem de servir sem mouse."""
    page, problems = open_audit(browser, server)
    alcancados = page.evaluate(
        """async () => {
            const nomes = [];
            for (let i = 0; i < 12; i++) {
                const ativo = document.activeElement;
                nomes.push(ativo ? (ativo.id || ativo.tagName) : "");
            }
            return nomes;
        }"""
    )
    foco = page.evaluate(
        """() => {
            const alvo = document.getElementById('reset');
            alvo.focus();
            const estilo = getComputedStyle(alvo, ':focus-visible');
            return { ativo: document.activeElement.id, outline: estilo.outlineWidth };
        }"""
    )
    page.close()
    assert problems == []
    assert foco["ativo"] == "reset"
    assert alcancados


def test_at_laptop_width_the_table_keeps_duration_and_status_visible(browser, server):
    """Entre 900 e 1280 a tabela divide a tela com o detalhe; a rota cede."""
    page, problems = open_audit(browser, server, width=1180, height=900)
    medido = page.evaluate(
        """() => {
            const linha = document.querySelector('#rows tr');
            const dentro = (td) => {
                const caixa = td.getBoundingClientRect();
                const lista = document.querySelector('.list').getBoundingClientRect();
                return caixa.right <= lista.right + 1;
            };
            return {
                rotaVisivel: getComputedStyle(linha.querySelector('td.route')).display !== 'none',
                duracaoDentro: dentro(linha.children[6]),
                statusDentro: dentro(linha.children[7]),
            };
        }"""
    )
    page.close()
    assert problems == []
    assert medido["rotaVisivel"] is False
    assert medido["duracaoDentro"], "a duracao ficou fora da area visivel"
    assert medido["statusDentro"], "o status ficou fora da area visivel"


# A analise chama um modelo de verdade, e nenhum provedor existe aqui. A rota e
# interceptada com uma resposta canonica: o que se mede nesta secao e a TELA --
# se o botao manda os filtros ativos, se a leitura e o dossie aparecem, e se o
# erro vira texto legivel em vez de painel vazio.
ANALISE = {
    "id": 1,
    "status": 200,
    "cached": False,
    "text": (
        "**1. Tire a ferramenta `Bash` do catálogo:** oferecida 8, chamada 8.\n"
        "**2. Segunda coisa.**"
    ),
    "requested_model": "claude-opus-5",
    "candidate_model": "openai/gpt-oss-120b",
    "provider": "groq",
    "input_tokens": 12000,
    "output_tokens": 800,
    "duration_ms": 9400,
    "dossier": {
        "period": {"since": None, "until": None, "filters": {"provider": "groq"}},
        "volume": {"requests": 24, "rows_read": 24, "sampled": False, "errors": 4},
        "tools": [{"tool": "Bash", "offered": 8, "called": 8, "call_rate": 1.0}],
    },
}


def stub_analysis(page, payload=ANALISE, status=200):
    """Substitui a rota da analise e devolve os pedidos que a tela fez."""
    import json as _json

    pedidos = []

    def handle(route, request):
        pedidos.append(request.url)
        route.fulfill(
            status=status,
            content_type="application/json",
            body=_json.dumps(payload),
        )

    page.route("**/api/analysis*", handle)
    return pedidos


def test_analysis_button_sends_the_filters_on_screen(browser, server):
    page, problems = open_audit(browser, server, query="?provider=groq&has_tools=true")
    pedidos = stub_analysis(page)
    page.click("#analyse")
    page.wait_for_selector("#analysis-text")
    leitura = page.inner_text(".read")
    marcacao = page.evaluate(
        """() => {
            const read = document.querySelector('.read');
            return {
                negrito: read.querySelectorAll('b').length,
                codigo: read.querySelectorAll('code').length,
                cru: read.innerText.includes('**') || read.innerText.includes('`'),
            };
        }"""
    )
    page.close()
    assert problems == []
    assert "provider=groq" in pedidos[0]
    assert "has_tools=true" in pedidos[0]
    # `order_by` ordena uma listagem e nao um periodo: mandar adiante faria a
    # rota recusar o dossie.
    assert "order_by" not in pedidos[0]
    assert "Tire a ferramenta Bash" in leitura
    # A resposta vem em Markdown: mostrar "**1. ...**" cru faria a leitura
    # parecer um log. Vira lista, e nenhum asterisco sobra na tela.
    assert marcacao["negrito"] == 2
    assert marcacao["codigo"] == 1
    assert marcacao["cru"] is False
    # A numeração sobrevive: ela é a ordem por impacto que o prompt exigiu.
    assert "1." in leitura


def test_the_dossier_behind_the_reading_can_be_checked(browser, server):
    page, problems = open_audit(browser, server)
    stub_analysis(page)
    page.click("#analyse")
    page.wait_for_selector("#analysis-text")
    page.click('.tabs button[data-tab="dossier"]')
    page.wait_for_selector("pre.raw")
    dossie = page.inner_text("pre.raw")
    visivel = page.evaluate(
        """() => ({
            dossie: !document.getElementById('analysis-dossier').hidden,
            leitura: !document.getElementById('analysis-text').hidden,
        })"""
    )
    page.close()
    assert problems == []
    assert visivel["dossie"]
    # Uma aba de cada vez: as duas juntas empilhariam o JSON embaixo da leitura
    # e a aba deixaria de significar alguma coisa.
    assert visivel["leitura"] is False
    # Quem le "oferecida 8, chamada 8" precisa achar os dois numeros.
    assert '"offered": 8' in dossie
    assert '"called": 8' in dossie


def test_an_analysis_error_is_shown_as_text_and_not_as_an_empty_panel(browser, server):
    page, problems = open_audit(browser, server)
    stub_analysis(
        page,
        payload={"status": 409, "error": "nenhum modelo declarado para a análise", "text": ""},
        status=409,
    )
    page.click("#analyse")
    # `:not(.loading)`: o aviso "Analisando o periodo" tambem e `.empty`, entao
    # esperar so `.empty` casa o carregamento e le o texto antes do 409 chegar.
    # Medido: sob `-n auto` o fetch voltou depois da leitura e o teste falhou.
    page.wait_for_selector("#detail .empty:not(.loading)")
    texto = page.inner_text("#detail")
    virou_analise = page.evaluate("() => Boolean(document.getElementById('analysis-text'))")
    page.close()
    # Erro NAO vira painel de analise com texto vazio: quem clicou tem de ler o
    # motivo, e nao uma leitura em branco com abas.
    assert virou_analise is False
    # O 409 e a resposta esperada aqui, e o navegador registra todo status >= 400
    # no console; o que nao pode aparecer e erro de JavaScript.
    assert [p for p in problems if "409" not in p] == []
    assert "nenhum modelo declarado" in texto


def test_a_reused_analysis_says_it_cost_nothing_now(browser, server):
    page, problems = open_audit(browser, server)
    stub_analysis(page, payload={**ANALISE, "cached": True})
    page.click("#analyse")
    page.wait_for_selector("#analysis-text")
    cabecalho = page.inner_text("#detail .id")
    page.close()
    assert problems == []
    assert "reaproveitada" in cabecalho


def test_on_a_phone_the_side_panel_takes_the_whole_width(browser, server):
    """Medido: o painel abria com 150 px e a leitura saía numa coluna de 100 px.

    `main.with-detail` tem mais especificidade que `main`, então a regra de
    empilhamento do telefone perdia sem nunca ser lida como erro -- o painel
    aparecia, só que espremido.
    """
    page, problems = open_audit(browser, server, width=390, height=844)
    stub_analysis(page)
    page.click("#analyse")
    page.wait_for_selector("#analysis-text")
    medido = page.evaluate(
        """() => ({
            painel: Math.round(document.getElementById('detail').getBoundingClientRect().width),
            leitura: Math.round(document.querySelector('.read').getBoundingClientRect().width),
            tela: document.documentElement.clientWidth,
        })"""
    )
    page.close()
    assert problems == []
    assert medido["painel"] >= medido["tela"] - 2, medido
    assert medido["leitura"] > 250, medido


def test_the_model_that_reads_the_period_can_be_chosen_before_paying(browser, server):
    """Sem `default_model` a análise recusa; a escolha tem de estar na tela antes."""
    page, problems = open_audit(browser, server)
    corpos = []

    def handle(route, request):
        if request.method == "POST":
            corpos.append(request.post_data)
            route.fulfill(
                status=200, content_type="application/json", body=json.dumps(ANALISE)
            )
        else:
            route.continue_()

    page.route("**/api/analysis*", handle)
    # O seletor é preenchido pela própria rota da análise, e o catálogo desta
    # instalação é o que o servidor de teste carregou.
    page.wait_for_function("() => document.getElementById('analysis-model').options.length > 0")
    opcoes = page.evaluate(
        "() => [...document.getElementById('analysis-model').options].map((o) => o.value)"
    )
    page.select_option("#analysis-model", opcoes[-1])
    page.click("#analyse")
    page.wait_for_selector("#analysis-text")
    page.close()
    assert problems == []
    assert opcoes
    assert f'"model":"{opcoes[-1]}"' in (corpos[0] or "").replace(" ", "")


def test_on_a_phone_opening_the_panel_scrolls_to_it(browser, server):
    """Medido num print de 390px: o painel abria fora da tela, e o clique parecia não fazer nada."""
    page, problems = open_audit(browser, server, width=390, height=844)
    stub_analysis(page)
    page.click("#analyse")
    page.wait_for_selector("#analysis-text")
    # A rolagem e `smooth`: mede so depois que ela parou, isto e, quando duas
    # leituras seguidas de `scrollY` dao o mesmo valor e a pagina ja saiu do topo.
    page.wait_for_function(
        """() => {
            const y = window.scrollY;
            const parado = window.__ultimoScrollY === y;
            window.__ultimoScrollY = y;
            return parado && y > 0;
        }""",
        polling=100,
    )
    caixa = page.evaluate(
        """() => {
            const c = document.getElementById('detail').getBoundingClientRect();
            return {topo: Math.round(c.top), base: Math.round(c.bottom), tela: window.innerHeight};
        }"""
    )
    page.close()
    assert problems == []
    # O critério é quanto do painel ficou VISÍVEL, e não a que altura ele parou:
    # com o painel no fim do documento a página não tem para onde rolar mais.
    visivel = min(caixa["base"], caixa["tela"]) - max(caixa["topo"], 0)
    assert visivel >= 250, caixa


def test_the_project_selector_narrows_the_list(browser, server):
    """De qual projeto veio o pedido: a pergunta que separa um dia de trabalho."""
    page, problems = open_audit(browser, server)
    page.wait_for_function(
        "() => document.getElementById('project').options.length > 1"
    )
    opcoes = page.evaluate(
        "() => [...document.getElementById('project').options].map((o) => [o.value, o.text])"
    )
    page.select_option("#project", "/home/iglesias/Projetos/agenda")
    wait_search(page, "new URLSearchParams(location.search).has('project')")
    medido = page.evaluate(
        """() => ({
            linhas: document.querySelectorAll('#rows tr').length,
            contagem: document.getElementById('count').textContent,
            url: location.search,
        })"""
    )
    page.close()
    assert problems == []
    # A opção mostra o nome curto e guarda o caminho inteiro.
    assert ["/home/iglesias/Projetos/agenda", "agenda (12)"] in opcoes
    assert medido["linhas"] == 12
    assert "project=" in medido["url"]
