"""A tela de auditoria num navegador HEADLESS, contra um banco semeado.

O que so aparece aqui: se a busca estreita de verdade, se o detalhe mostra a
cadeia da requisicao clicada, e se o estado da busca vive na URL -- que e o que
transforma uma investigacao num link.
"""

import os
import socket
import subprocess
import time
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.stats.models import Base, RequestBody, RequestEvent

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
            )
        )
    conversas = [
        RequestBody(
            request_id=f"req-{index:02d}",
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
        )
        for index in range(0, 24, 2)
    ]
    with Session(engine) as session:
        session.add_all(rows)
        session.add_all(conversas)
        session.commit()
    engine.dispose()


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    database = tmp_path_factory.mktemp("auditoria") / "stats.db"
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


def open_audit(browser, base, width=1500, height=1000, query=""):
    page = browser.new_page(viewport={"width": width, "height": height})
    problems = []
    page.on("pageerror", lambda error: problems.append(str(error)))
    page.on(
        "console",
        lambda message: problems.append(message.text) if message.type == "error" else None,
    )
    page.goto(f"{base}/requests{query}", wait_until="networkidle")
    page.wait_for_selector("#rows tr")
    return page, problems


def test_the_screen_lists_the_requests_newest_first(browser, server):
    page, problems = open_audit(browser, server)
    ids = page.evaluate("() => [...document.querySelectorAll('#rows tr')].map(r => r.dataset.id)")
    count = page.inner_text("#count")
    page.close()
    assert problems == []
    assert ids[:3] == ["req-00", "req-01", "req-02"]
    assert "de 24 requisições" in count


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
    page.wait_for_timeout(500)
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
            .find(r => r.textContent.includes('fallback'))?.dataset.id"""
    )
    assert target, "nenhuma linha com fallback para abrir"
    page.click(f'#rows tr[data-id="{target}"]')
    page.wait_for_selector("#detail .chain li")
    detail = page.inner_text("#detail")
    api = httpx.get(f"{server}/api/requests/{target}", timeout=10).json()
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
    page.wait_for_function("() => new URLSearchParams(location.search).get('order_by') === 'duration'")
    page.wait_for_timeout(400)
    durations = page.evaluate(
        """() => [...document.querySelectorAll('#rows tr')]
            .map(r => r.children[5].textContent.trim())"""
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


def test_the_export_link_carries_the_same_filters(browser, server):
    page, _ = open_audit(browser, server)
    page.get_by_role("button", name="falhas").click()
    page.wait_for_timeout(400)
    href = page.get_attribute("#export", "href")
    page.close()
    assert "status_min=400" in href
    csv_text = httpx.get(f"{server}{href}", timeout=10).text
    assert csv_text.startswith("started_at,request_id")
    assert csv_text.count("\n") > 1


def test_clearing_the_filters_brings_everything_back(browser, server):
    page, problems = open_audit(browser, server)
    total = page.inner_text("#count")
    page.get_by_role("button", name="falhas").click()
    page.wait_for_timeout(400)
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
    page.wait_for_function("() => new URLSearchParams(location.search).get('provider') === 'groq'")
    page.wait_for_timeout(400)
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
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    problems = []
    page.on("pageerror", lambda error: problems.append(str(error)))
    page.goto(f"{server}/requests?provider=um-provedor-que-sumiu", wait_until="networkidle")
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


def test_a_request_with_no_stored_conversation_says_how_to_turn_it_on(browser, server):
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
    assert "SHUNT_STORE_BODIES=1" in said


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
