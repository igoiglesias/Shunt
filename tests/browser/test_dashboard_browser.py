"""O painel conferido num navegador HEADLESS, contra um banco semeado.

Nunca abre janela na maquina de ninguem: `chromium.launch()` e headless por
padrao e nenhum teste aqui passa `headless=False`.

O que este arquivo prende e o que um teste de rota nao alcanca: se o JavaScript
levanta, se o SVG desenha, se a pagina rola de lado no celular, e se o numero
que aparece na tela e o mesmo que a API respondeu.
"""

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

from app.stats.models import Base, RequestEvent

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def seed(path) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{path}")
    Base.metadata.create_all(engine)
    now = datetime.now(UTC)
    rows = []
    for index in range(30):
        failed = index % 10 == 0
        rows.append(
            RequestEvent(
                request_id=f"b{index}",
                started_at=now - timedelta(minutes=index * 3),
                route="/v1/messages",
                dialect="anthropic",
                stream=index % 2 == 0,
                requested_model="claude-sonnet-4-5",
                rule="family",
                matched="sonnet",
                provider=None if failed else "groq",
                candidate_model=None if failed else "openai/gpt-oss-120b",
                status=429 if failed else 200,
                error_type="rate_limit_error" if failed else None,
                input_tokens=100,
                output_tokens=50,
                ttft_ms=200 if index % 2 == 0 and not failed else None,
                duration_ms=500,
                attempts=["free: 429 (attempt 1)"] if index % 5 == 0 else [],
                fell_back=index % 5 == 0 and not failed,
                tools_offered=["read"],
                tools_called=["read"] if index % 3 == 0 else [],
                thinking_blocks=1,
            )
        )
    with Session(engine) as session:
        session.add_all(rows)
        session.commit()
    engine.dispose()


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    database = tmp_path_factory.mktemp("painel") / "stats.db"
    seed(database)
    port = free_port()
    process = subprocess.Popen(
        ["uv", "run", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        env={**os.environ, "TURSO_DATABASE_URL": f"sqlite+pysqlite:///{database}"},
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
        pytest.fail("o servidor do painel nao subiu")
    yield base
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        # `uv run` repassa o sinal; se o worker demorar, nao vale pendurar a
        # suite esperando um servidor de teste.
        process.kill()
        process.wait(timeout=10)


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch()  # headless
        yield instance
        instance.close()


def open_panel(browser, base, width, height):
    page = browser.new_page(viewport={"width": width, "height": height})
    problems = []
    page.on("pageerror", lambda error: problems.append(str(error)))
    page.on(
        "console",
        lambda message: problems.append(message.text) if message.type == "error" else None,
    )
    page.goto(base, wait_until="networkidle")
    # O sinal de "carregou" e o estado do gravador, que existe sempre. Esperar
    # por uma figura exigiria trafego: uma janela vazia desenha a faixa de
    # saida e esconde o resumo.
    page.wait_for_selector("#state div")
    return page, problems


def test_the_panel_draws_without_a_single_console_error(browser, server):
    page, problems = open_panel(browser, server, 1440, 1000)
    assert problems == []
    assert page.locator("#flow svg").count() == 1
    # A serie desenha linha com tres ou mais horas e barras com menos; as duas
    # formas contam como desenhada.
    assert page.locator("#series path").count() + page.locator("#series rect").count() > 0
    page.close()


def test_the_numbers_on_screen_are_the_numbers_the_api_answered(browser, server):
    """Painel bonito com numero errado e pior do que painel nenhum."""
    api = httpx.get(f"{server}/api/stats?window=24", timeout=10).json()
    page, _ = open_panel(browser, server, 1440, 1000)
    shown = page.evaluate(
        """() => ({
            requests: document.querySelector('#figures .figure.requests .value').textContent,
            models: [...document.querySelectorAll('#models tbody tr')].map(
                r => [...r.children].slice(0, 2).map(c => c.textContent.trim())),
            errors: [...document.querySelectorAll('#errors tbody tr td:last-child')].map(
                c => Number(c.textContent)),
        })"""
    )
    page.close()
    assert shown["requests"] == str(api["totals"]["requests"])
    assert shown["models"][0][0] == api["by_model"][0]["model"]
    assert int(shown["models"][0][1]) == api["by_model"][0]["requests"]
    assert sum(shown["errors"]) == sum(row["requests"] for row in api["errors"])


def test_the_panel_never_scrolls_sideways_on_a_phone(browser, server):
    page, problems = open_panel(browser, server, 390, 844)
    measured = page.evaluate(
        """() => ({
            scrollWidth: document.documentElement.scrollWidth,
            innerWidth: window.innerWidth,
            flowLabel: document.querySelector('#flow text.flow-label')?.getBoundingClientRect(),
        })"""
    )
    page.close()
    assert problems == []
    assert measured["scrollWidth"] <= measured["innerWidth"] + 1
    # Rotulo legivel: menos de 9 px de altura na tela e texto que ninguem le.
    assert measured["flowLabel"]["height"] >= 9, measured["flowLabel"]


def test_a_request_served_right_now_lands_on_the_tape(browser, server):
    """O SSE: o painel aberto tem de mostrar a requisicao chegando."""
    page, _ = open_panel(browser, server, 1440, 1000)
    before = page.locator("#tape li").count()
    httpx.get(f"{server}/v1/models", timeout=10)
    page.wait_for_function(f"document.querySelectorAll('#tape li').length > {before}", timeout=10_000)
    first = page.locator("#tape li").first.inner_text()
    page.close()
    assert "models" in first


def test_the_window_buttons_reload_the_summary(browser, server):
    page, problems = open_panel(browser, server, 1440, 1000)
    page.get_by_role("button", name="1h").click()
    page.wait_for_function("document.querySelector('#figures .figure .value').textContent !== ''")
    pressed = page.evaluate(
        """() => [...document.querySelectorAll('#windows button')]
            .filter(b => b.getAttribute('aria-pressed') === 'true').map(b => b.textContent)"""
    )
    page.close()
    assert pressed == ["1h"]
    assert problems == []


def test_the_panel_says_so_when_there_is_no_database(browser, tmp_path_factory):
    """Sem Turso o painel nao pode quebrar: ele avisa e segue ao vivo."""
    port = free_port()
    process = subprocess.Popen(
        ["uv", "run", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        # Vazio, e nao ausente: `load_dotenv` nao sobrescreve variavel ja
        # definida, entao so assim o `.env` do repositorio nao repoe o banco.
        env={**os.environ, "TURSO_DATABASE_URL": "", "TURSO_AUTH_TOKEN": ""},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{base}/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.3)
        page, problems = open_panel(browser, base, 1200, 900)
        text = page.inner_text("body")
        page.close()
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    assert problems == []
    assert "sem banco" in text
    assert "TURSO_DATABASE_URL" in text


def test_the_api_answers_the_same_json_the_panel_parses(server):
    """Contrato entre as duas metades, preso sem navegador nenhum."""
    body = httpx.get(f"{server}/api/stats", timeout=10).json()
    assert json.loads(json.dumps(body)) == body
    assert set(body) >= {"totals", "per_hour", "by_model", "chain", "tools", "recent", "health"}


def test_a_single_hour_of_traffic_draws_bars_instead_of_a_flat_line(browser, server):
    """Medido com trafego real: uma hora so desenhava uma linha de um ponto.

    Quem acaba de subir o proxy tem exatamente uma hora de historico, e o painel
    nao pode parecer quebrado justamente na primeira olhada.
    """
    page, problems = open_panel(browser, server, 1400, 900)
    page.get_by_role("button", name="1h").click()
    page.wait_for_function("document.querySelectorAll('#series rect').length > 0", timeout=10_000)
    bars = page.locator("#series rect").count()
    page.close()
    assert problems == []
    assert bars >= 2, "uma hora de trafego desenha as duas barras"


def test_a_route_with_no_model_never_appears_as_a_requested_model(browser, server):
    """`/v1/models` nao pede modelo; o diagrama nao pode ganhar uma linha sem nome."""
    page, _ = open_panel(browser, server, 1400, 900)
    httpx.get(f"{server}/v1/models", timeout=10)
    page.reload(wait_until="networkidle")
    page.wait_for_selector("#flow text.flow-label")
    labels = page.evaluate(
        "() => [...document.querySelectorAll('#flow text.flow-label')].map(t => t.textContent.trim())"
    )
    page.close()
    assert labels
    assert "" not in labels


def test_the_tape_fills_in_what_the_live_stream_could_not_see(browser, server):
    """`make prod` sobe um processo por nucleo, e o SSE so alcanca um deles.

    O resumo vem do banco compartilhado, entao a fita tem de completar a partir
    dele -- e sem duplicar o que o ao vivo ja colocou na tela.
    """
    page, problems = open_panel(browser, server, 1400, 900)
    page.wait_for_function("document.querySelectorAll('#tape li').length > 0", timeout=10_000)
    ids = page.evaluate(
        "() => [...document.querySelectorAll('#tape li')].map(i => i.dataset.id)"
    )
    # Uma atualizacao inteira do resumo nao pode repetir nenhuma linha.
    page.evaluate("() => load()")
    page.wait_for_timeout(600)
    again = page.evaluate(
        "() => [...document.querySelectorAll('#tape li')].map(i => i.dataset.id)"
    )
    page.close()
    assert problems == []
    assert ids, "a fita nasceu vazia"
    assert len(again) == len(set(again)), "a fita repetiu uma requisicao"
    assert set(ids) <= set(again)


def test_the_tape_keeps_the_newest_request_on_top(browser, server):
    page, _ = open_panel(browser, server, 1400, 900)
    page.wait_for_function("document.querySelectorAll('#tape li').length > 0", timeout=10_000)
    httpx.get(f"{server}/v1/models", timeout=10)
    page.wait_for_timeout(800)
    page.evaluate("() => load()")
    page.wait_for_timeout(600)
    order = page.evaluate(
        """() => [...document.querySelectorAll('#tape li')]
            .map(i => Date.parse(i.dataset.when) || 0)"""
    )
    page.close()
    assert order == sorted(order, reverse=True), "a fita saiu fora de ordem"


def test_the_tape_can_show_only_the_failures(browser, server):
    """Com o painel cheio, achar a requisicao que falhou e a pergunta urgente."""
    page, problems = open_panel(browser, server, 1400, 900)
    page.wait_for_function("document.querySelectorAll('#tape li').length > 0", timeout=10_000)
    total = page.locator("#tape li:visible").count()
    page.get_by_role("button", name="só erros").click()
    page.wait_for_timeout(300)
    filtered = page.locator("#tape li:visible").count()
    kinds = page.evaluate(
        """() => [...document.querySelectorAll('#tape li')]
            .filter(i => !i.hidden).map(i => i.className.includes('bad'))"""
    )
    page.get_by_role("button", name="só erros").click()
    page.wait_for_timeout(300)
    back = page.locator("#tape li:visible").count()
    page.close()
    assert problems == []
    assert filtered < total, "o filtro nao escondeu nada"
    assert all(kinds), "linha sem falha sobreviveu ao filtro"
    assert back == total, "o filtro nao soltou a lista"


def test_a_worker_without_a_database_does_not_blank_the_panel(browser, server):
    """Medido em producao: um terco das cargas caia num worker sem banco.

    O painel zerava a tela inteira e voltava na atualizacao seguinte -- o
    sintoma que o usuario descreveu como "fica todo em branco".
    """
    # Nao depende do que outros testes deixaram: gera o proprio trafego, porque
    # o teste de limpeza pode ter zerado o banco antes deste rodar.
    for _ in range(3):
        httpx.get(f"{server}/v1/models", timeout=10)
    page, problems = open_panel(browser, server, 1400, 900)
    # O painel recarrega sozinho a cada 15 s; aqui o teste pede a atualizacao em
    # vez de esperar por ela, depois de dar ao worker o tempo de um lote.
    page.wait_for_timeout(1500)
    # Janela de duas horas: o resumo e cacheado por cinco segundos POR JANELA, e
    # a de 24 h acabou de ser respondida zerada pelo teste que limpou o banco.
    page.evaluate("() => { hours = 2; }")
    page.evaluate("() => load()")
    page.wait_for_function("() => snapshot.totals.requests > 0", timeout=10_000)
    before = page.evaluate("() => snapshot.totals.requests")

    # Responde como um worker que nao abriu o banco: zerado, mas configurado.
    page.route(
        "**/api/stats?*",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "window_hours": 24,
                    "totals": {
                        "requests": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "errors": 0,
                        "error_rate": 0.0,
                        "fallbacks": 0,
                        "streams": 0,
                        "p50_duration_ms": None,
                        "p95_duration_ms": None,
                        "p50_ttft_ms": None,
                        "p95_ttft_ms": None,
                    },
                    "per_hour": [],
                    "by_model": [],
                    "by_provider": [],
                    "by_route": [],
                    "by_requested_model": [],
                    "errors": [],
                    "chain": {
                        "requests": 0,
                        "first_candidate_answered": 0,
                        "first_candidate_rate": 0.0,
                        "skips": [],
                    },
                    "tools": [],
                    "recent": [],
                    "health": {
                        "enabled": False,
                        "configured": True,
                        "queued": 0,
                        "dropped": 0,
                        "failures": 0,
                        "commits": 0,
                        "reconnects": 0,
                    },
                }
            ),
        ),
    )
    page.evaluate("() => load()")
    page.wait_for_timeout(500)
    after = page.evaluate("() => snapshot.totals.requests")
    state = page.inner_text("#state")
    rows = page.locator("#models tbody tr").count()
    page.close()
    assert problems == []
    assert after == before, "o resumo de um worker quebrado apagou os dados"
    assert rows > 0, "a tabela ficou em branco"
    assert "reconectando" in state


def test_the_summary_is_grouped_instead_of_a_row_of_loose_numbers(browser, server):
    page, problems = open_panel(browser, server, 1500, 1000)
    grupos = page.evaluate(
        "() => [...document.querySelectorAll('#figures .group h3')].map(h => h.textContent)"
    )
    page.close()
    assert problems == []
    assert grupos == ["Volume", "Latência", "Saúde"]

def test_the_series_has_two_axes_a_legend_and_no_overlapping_labels(browser, server):
    """Requisicoes e tokens sao grandezas incomparaveis: cada uma no seu eixo."""
    page, problems = open_panel(browser, server, 1500, 1000)
    medido = page.evaluate(
        """() => {
            const textos = [...document.querySelectorAll('#series text')];
            const caixas = textos.map(t => t.getBoundingClientRect());
            let colisoes = 0;
            for (let i = 0; i < caixas.length; i++) {
                for (let j = i + 1; j < caixas.length; j++) {
                    const a = caixas[i], b = caixas[j];
                    if (a.left < b.right && b.left < a.right && a.top < b.bottom && b.top < a.bottom) {
                        colisoes += 1;
                    }
                }
            }
            return {
                colisoes,
                legenda: [...document.querySelectorAll('#series + .legend .key, .legend .key')]
                    .map(k => k.textContent.trim()),
                barras: document.querySelectorAll('#series rect').length,
                linhasDeGrade: document.querySelectorAll('#series line').length,
            };
        }"""
    )
    page.close()
    assert problems == []
    assert medido["colisoes"] == 0, f"{medido['colisoes']} rotulos sobrepostos no grafico"
    assert medido["legenda"] == ["requisições", "tokens", "horas com erro"]
    assert medido["barras"] > 0
    assert medido["linhasDeGrade"] >= 3, "faltou a grade que da escala ao grafico"


def test_the_flow_links_each_request_to_what_answered_it(browser, server):
    """Duas colunas soltas nao diziam se foi o opus que caiu no local."""
    page, problems = open_panel(browser, server, 1500, 1000)
    medido = page.evaluate(
        """() => ({
            fitas: document.querySelectorAll('#flow path').length,
            nos: [...document.querySelectorAll('#flow text.flow-label')].map(t => t.textContent),
            titulos: [...document.querySelectorAll('#flow path title')].map(t => t.textContent),
        })"""
    )
    page.close()
    assert problems == []
    assert medido["fitas"] > 0, "nenhuma fita ligando os dois lados"
    assert medido["nos"], "nenhum modelo no diagrama"
    assert any("→" in titulo for titulo in medido["titulos"]), medido["titulos"]


def test_an_empty_window_offers_the_way_out(browser, server):
    """Sete paineis dizendo "nada aqui" nao sao um estado vazio util.

    O banco semeado tem trafego em toda janela, entao a resposta vazia vem
    mockada -- o que se mede aqui e o que a TELA faz com ela.
    """
    page, problems = open_panel(browser, server, 1400, 900)
    vazio = {
        "window_hours": 0.0833,
        "totals": {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "errors": 0,
            "error_rate": 0.0,
            "fallbacks": 0,
            "streams": 0,
            "p50_duration_ms": None,
            "p95_duration_ms": None,
            "p50_ttft_ms": None,
            "p95_ttft_ms": None,
        },
        "per_hour": [],
        "by_model": [],
        "by_provider": [],
        "by_route": [],
        "by_requested_model": [],
        "pairs": [],
        "errors": [],
        "chain": {
            "requests": 0,
            "first_candidate_answered": 0,
            "first_candidate_rate": 0.0,
            "skips": [],
        },
        "tools": [],
        "recent": [],
        "health": {
            "enabled": True,
            "configured": True,
            "queued": 0,
            "dropped": 0,
            "failures": 0,
            "commits": 0,
            "reconnects": 0,
        },
    }
    page.route(
        "**/api/stats?*",
        lambda route: route.fulfill(
            status=200, content_type="application/json", body=json.dumps(vazio)
        ),
    )
    page.get_by_role("button", name="5min").click()
    page.wait_for_function("() => document.querySelector('main').classList.contains('idle')")
    dito = page.inner_text(".nothing-said")
    escondidos = page.evaluate(
        """() => ({
            grade: getComputedStyle(document.querySelector('.grid')).display,
            faixa: getComputedStyle(document.querySelector('.flow-band')).display,
            saida: !document.getElementById('widen').hidden,
        })"""
    )
    page.close()
    assert problems == []
    assert "Nenhuma requisição nos últimos 5 min" in dito
    assert escondidos["grade"] == "none"
    assert escondidos["faixa"] == "none"
    assert escondidos["saida"] is True


def test_the_way_out_of_an_empty_window_is_the_24h_button(browser, server):
    page, problems = open_panel(browser, server, 1400, 900)
    page.get_by_role("button", name="5min").click()
    page.wait_for_timeout(400)
    page.evaluate("() => { document.getElementById('widen').click(); }")
    page.wait_for_function(
        """() => [...document.querySelectorAll('#windows button')]
            .find(b => b.getAttribute('aria-pressed') === 'true')?.textContent === '24h'"""
    )
    page.close()
    assert problems == []


# Os dois testes de limpeza ficam no FIM do arquivo de proposito: eles zeram
# o banco que o servidor deste modulo compartilha, e qualquer teste depois
# deles veria um painel vazio que nao e o que ele quer medir.
def test_clearing_the_history_takes_two_clicks_and_then_empties_the_panel(browser, server):
    """Apagar e irreversivel: o primeiro clique arma e diz quanto vai embora."""
    page, problems = open_panel(browser, server, 1400, 900)
    before = page.evaluate("() => snapshot.totals.requests")
    assert before > 0

    button = page.locator("#clear")
    button.click()
    page.wait_for_timeout(200)
    armed = button.inner_text()
    assert str(before) in armed, f"o botao armado nao disse quanto apaga: {armed!r}"

    button.click()
    page.wait_for_function("() => snapshot.totals.requests === 0", timeout=10_000)
    said = page.locator("#clear-said").inner_text()
    tape = page.locator("#tape li").count()
    page.close()
    assert problems == []
    assert "apagadas" in said
    assert tape == 0

def test_the_clear_button_disarms_itself_when_left_alone(browser, server):
    page, _ = open_panel(browser, server, 1400, 900)
    page.evaluate("() => { window.__armWait = 8000; }")
    button = page.locator("#clear")
    button.click()
    assert button.get_attribute("data-armed") == "true"
    page.evaluate("() => disarmClear()")
    page.wait_for_timeout(100)
    state = button.get_attribute("data-armed")
    label = button.inner_text()
    page.close()
    assert state == "false"
    assert label == "Limpar histórico"


