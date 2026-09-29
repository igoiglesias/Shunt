// Executa o JavaScript da tela de auditoria sem navegador.

// A lista, o detalhe e a sincronizacao do filtro sao desenhados por funcoes que
// vivem num `<script>` do template (`rowOf`, `detailOf`, `SELECTS`). O
// navegador tem a sua bateria propria (`tests/browser/test_audit_browser.py`),
// mas o RED de uma task nao pode depender de subir um servidor + Chromium para
// ficar vermelho: aqui o `<script>` e extraido do template e rodado num
// contexto com o minimo de DOM que essas funcoes tocam.

// O shim e pequeno de proposito. So existe a superficie que o `<script>` reacha
// no carregamento e nos caminhos testados: getElementById com elementos
// pre-registrados para os `<select>` estaticos (options de verdade do HTML),
// addEventListener/setText/innerHTML/hidden/value, querySelectorAll dos chips e
// dos th ordenaveis, history, location e fetch. Nada de arvore de verdade --
// quem precisasse de layout estaria no teste de navegador.

// Uso: node tests/stats/audit_page.js <cenario> [<cenario>...]
// Cada cenario roda num contexto NOVO e a saida e um JSON por linha.

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const TEMPLATE = path.join(__dirname, "..", "..", "app", "templates", "audit.html");
const HTML = fs.readFileSync(TEMPLATE, "utf-8");
const SCRIPT = /<script>([\s\S]*?)<\/script>/.exec(HTML)[1];

// Chips e colunas ordenaveis sao lidos do proprio HTML: duplica-los aqui
// faria o teste discordar da tela no dia em que ela ganhar um chip novo.
const CHIPS = [...HTML.matchAll(/class="chip[^"]*" data-filter="([^"]*)" data-value="([^"]*)"/g)].map(
  (m) => ({ filter: m[1], value: m[2] })
);
const ORDERS = [...HTML.matchAll(/<th[^>]*data-order="([^"]+)"/g)].map((m) => m[1]);

// Um elemento lazioso: so existe o que a tela de fato usa.
function makeElement(tag = "div") {
  const listeners = {};
  return {
    tag,
    _html: "",
    _text: "",
    value: "",
    hidden: false,
    dataset: {},
    className: "",
    listeners,
    addEventListener(type, fn) {
      (listeners[type] ||= []).push(fn);
    },
    setAttribute(name, val) {
      this["_" + name] = val;
    },
    removeAttribute(name) {
      delete this["_" + name];
    },
    getAttribute(name) {
      return this["_" + name] ?? null;
    },
    get innerHTML() {
      return this._html;
    },
    set innerHTML(v) {
      this._html = String(v);
    },
    get textContent() {
      return this._text;
    },
    set textContent(v) {
      this._text = String(v);
    },
    insertAdjacentHTML(where, chunk) {
      this._html += String(chunk);
    },
    // `render` conta os filhos e seleciona a linha pelo id do dataset.
    get children() {
      return [...this._html.matchAll(/<tr\b[^>]*>/g)].map((m) => {
        const id = /data-id="([^"]*)"/.exec(m[0]);
        return makeTr(id ? id[1] : "");
      });
    },
    querySelector(sel) {
      const want = /data-id="([^"]*)"/.exec(sel);
      if (!want) return null;
      const re = new RegExp(`<tr\\b[^>]*data-id="${want[1]}"[\\s\\S]*?</tr>`);
      const found = re.exec(this._html);
      return found ? makeTr(want[1]) : null;
    },
    querySelectorAll() {
      return [];
    },
    classList: {
      _set: new Set(),
      add(c) {
        this._set.add(c);
      },
      remove(c) {
        this._set.delete(c);
      },
      contains(c) {
        return this._set.has(c);
      },
    },
    focus() {},
    scrollIntoView() {},
    // O `.closest` do script da tela sobe de um <button> ate o <tr> da linha;
    // aqui nao ha arvore, e o elemento e a propria resposta.
    closest() {
      return this;
    },
    click() {
      (listeners.click || []).forEach((fn) => fn({ target: this }));
    },
  };
}

function makeTr(id) {
  const tr = makeElement("tr");
  tr.dataset = { id };
  return tr;
}

// Os `<select>` estaticos (options fixas no HTML) sao pre-registrados: e o que
// prova que o `kind` nasce com as options certas e que nenhuma chamada as apaga.
function seedStaticSelects(registry) {
  for (const block of HTML.matchAll(/<select id="([^"]+)"[\s\S]*?<\/select>/g)) {
    const id = block[1];
    const elx = registry[id] || makeElement("select");
    elx._options = [...block[0].matchAll(/<option value="([^"]*)"[^>]*>([\s\S]*?)<\/option>/g)].map(
      (m) => ({ value: m[1], text: m[2].trim() })
    );
    registry[id] = elx;
  }
}

// Acha um elemento no HTML do detalhe por classe (`.empty`) ou por tag (`h2`).
// O `[\s\S]*?` nao respeita aninhamento, mas o detalhe so tem elementos rasos
// -- um `<p>` de explicacao e um `<h2>` de cabecalho -- e a casa fechada e a do
// MESMO nome de tag, por isso a abertura e o fechamento usam o nome do alvo.
function achaElemento(html, alvo) {
  const classm = /^\.([\w-]+)$/.exec(alvo);
  const re = classm
    ? new RegExp(
        `<([a-zA-Z][\\w-]*)[^>]*class="[^"]*\\b${classm[1]}\\b[^"]*"[\\s\\S]*?</([a-zA-Z][\\w-]*)>`
      )
    : new RegExp(`<${alvo}\\b[\\s\\S]*?</${alvo}>`);
  const found = re.exec(html);
  if (!found) return null;
  // Se a busca foi por classe, a tag certa e a capturada na abertura; o
  // fechamento de `[\s\S]*?` pode ser de uma tag interna.
  const tag = classm ? found[1] : alvo;
  const fecho = found[0].lastIndexOf(`</${tag}>`);
  const inner =
    fecho > -1
      ? found[0].slice(found[0].indexOf(">") + 1, fecho)
      : found[0].slice(found[0].indexOf(">") + 1, found[0].lastIndexOf("<"));
  return { tag, text: inner };
}

// Fabrica de pagina: contexto novo, script rodado de novo, estado zerado.
function buildPage(initialSearch = "") {
  const registry = {};
  seedStaticSelects(registry);
  const created = [];
  const documents = {
    chips: CHIPS.map((c) => {
      const chip = makeElement("button");
      chip.dataset = { filter: c.filter, value: c.value };
      chip.getAttribute = (name) =>
        name === "aria-pressed" ? chip._aria_pressed ?? "false" : chip["_" + name] ?? null;
      chip._aria_pressed = "false";
      chip.setAttribute = (name, val) => {
        if (name === "aria-pressed") chip._aria_pressed = val;
        else chip["_" + name] = val;
      };
      return chip;
    }),
    orders: ORDERS.map((o) => {
      const th = makeElement("th");
      th.dataset = { order: o };
      return th;
    }),
  };

  const location = {
    search: initialSearch,
    pathname: "/admin/requests",
    origin: "http://shunt.test",
  };
  const historyLog = [];
  const fetchLog = [];

  // `search()` na carga nao pode quebrar a pagina: devolve uma pagina vazia e
  // as facets vazias, que e o que um banco sem dados responde.
  let fetchRoute = null;
  async function fetch(url) {
    fetchLog.push(url);
    if (fetchRoute) return fetchRoute(url);
    if (String(url).includes("/api/requests/facets")) {
      return { ok: true, json: async () => ({}) };
    }
    return { ok: true, json: async () => ({ events: [], total: 0, next_cursor: null }) };
  }

  const sandbox = {
    console,
    URLSearchParams,
    Date,
    Number,
    String,
    Boolean,
    Array,
    Object,
    JSON,
    Math,
    RegExp,
    Error,
    Promise,
    setTimeout,
    clearTimeout,
    location,
    history: {
      pushState: (_s, _t, url) => historyLog.push(["push", url]),
      replaceState: (_s, _t, url) => historyLog.push(["replace", url]),
    },
    window: {
      addEventListener() {},
      innerWidth: 1366,
    },
    document: {
      addEventListener() {},
      getElementById(id) {
        return (registry[id] ||= makeElement("div"));
      },
      // Sem arvore de verdade: os seletores que a tela usa fora do carregamento
      // procuram dentro do detalhe (`#detail h2`, `#detail .empty`), e o HTML
      // do detalhe ja esta no registro, entao a busca e no texto dele.
      querySelector(sel) {
        if (sel === "main") {
          return (registry.__main ||= makeElement("main"));
        }
        const idm = /^#([\w-]+)$/.exec(sel);
        if (idm) return registry[idm[1]] || null;
        // Tira o escopo (`#detail ...`) e fica so o alvo: classe ou tag.
        const alvo = sel.replace(/^#[\w-]+ /, "");
        const detail = registry.detail?._html ?? "";
        if (!detail) return registry[sel] || null;
        const achado = achaElemento(detail, alvo);
        if (!achado) return registry[sel] || null;
        const elx = makeElement(achado.tag);
        elx._text = achado.text;
        return elx;
      },
      querySelectorAll(sel) {
        if (sel === ".chip") return documents.chips;
        if (sel === "th[data-order]") return documents.orders;
        return [];
      },
      createElement: () => makeElement("a"),
    },
    navigator: { clipboard: { writeText: async () => {} } },
    // `AbortController` nao existe dentro de um contexto `vm` do Node; a tela
    // so o usa para abortar buscas sobrepostas, e aqui nenhuma rede existe.
    AbortController: class {
      constructor() {
        this.signal = { aborted: false, addEventListener() {} };
      }
      abort() {
        this.signal.aborted = true;
      }
    },
    CSS: { escape: (s) => String(s).replace(/[!"#$%&'()*+,./:;<=>?@[\\\]^`{|}~]/g, "\\$&") },
    fetch,
  };
  const ctx = vm.createContext(sandbox);
  vm.runInContext(SCRIPT, ctx, { filename: "audit.html/script" });

  return {
    sandbox,
    registry,
    historyLog,
    fetchLog,
    setFetchRoute(fn) {
      fetchRoute = fn;
    },
    // Roda UM trecho dentro do sandbox `vm` do proprio template da tela.
    //
    // Nao e o `eval` do JS no sentido perigoso: a entrada e sempre codigo
    // escrito AQUI neste arquivo (chamadas a `rowOf`/`detailOf`/`render` e
    // leitura de `state`/`SELECTS`), nunca dados de fora -- o que vem da API
    // entra por JSON.stringify, e o sandbox isolado e o que mantem a tela e o
    // teste no mesmo contexto sem navegador.
    probe(expr) {
      return vm.runInContext(expr, ctx, { filename: "probe" });
    },
    // Dispara o listener de change registrado pela tela, como o navegador faria.
    change(id, value) {
      const elx = sandbox.document.getElementById(id);
      elx.value = value;
      for (const fn of elx.listeners.change || []) fn({ target: elx });
    },
    // Renderiza pela funcao REAL da tela, no caminho REAL do `render`.
    render(events, total = events.length) {
      this.probe(`render({ events: ${JSON.stringify(events)}, total: ${total}, next_cursor: null }, false)`);
    },
    // Caminho real do detalhe: `openDetail` e async, entao o aguardo e aqui.
    async openDetail(event) {
      this.setFetchRoute((url) => ({
        ok: true,
        json: async () => event,
      }));
      this.probe(`openDetail(${JSON.stringify(event.request_id)})`);
      for (let i = 0; i < 20; i++) {
        await new Promise((r) => setTimeout(r, 0));
      }
    },
    // `loadFacets` e async: sem aguardar, a afirmacao sobre as options do `kind`
    // poderia correr antes da chamada as facets sequer terminar.
    async settle() {
      for (let i = 0; i < 20; i++) {
        await new Promise((r) => setTimeout(r, 0));
      }
    },
  };
}

// Evento de relay no formato que `_as_event` entrega: medido em None, e nao em
// zero (`app/stats/queries.py`). `null` aqui e o None do Python.
const RELAY_EVENT = {
  id: 7,
  request_id: "repasse",
  kind: "relay",
  started_at: "2026-09-27T10:00:00Z",
  route: "/api/oauth/usage",
  dialect: "anthropic",
  stream: false,
  requested_model: "",
  rule: "none",
  matched: null,
  provider: null,
  candidate_model: null,
  status: 502,
  error_type: "upstream_unreachable",
  input_tokens: null,
  output_tokens: null,
  ttft_ms: null,
  duration_ms: 99999,
  attempts: ["x: 502 (attempt 1)"],
  fell_back: false,
  tools_offered: ["read"],
  tools_called: ["read"],
  thinking_blocks: 0,
  project: null,
  session_id: null,
  cached_input_tokens: null,
  cache_write_tokens: null,
};

const MODEL_EVENT = {
  ...RELAY_EVENT,
  request_id: "ok",
  kind: "model",
  route: "/v1/messages",
  requested_model: "claude-haiku-4-5",
  rule: "family",
  matched: "haiku",
  provider: "groq",
  candidate_model: "openai/gpt-oss-120b",
  status: 200,
  error_type: null,
  input_tokens: 10,
  output_tokens: 5,
  ttft_ms: 120,
  duration_ms: 100,
  attempts: [],
  cached_input_tokens: 8,
  cache_write_tokens: 2,
};

const SCENARIOS = {
  // RED 1: o select existe com tres options e, depois de `loadFacets`, elas
  // continuam la -- `fill` so e chamado para as 4 facets reais.
  async static_kind_select() {
    const page = buildPage();
    await page.settle();
    // `registry.kind` so existe quando o `<select id="kind">` esta no HTML.
    // Sem ele, `undefined` e a propria prova do RED 1.
    const select = page.registry.kind;
    return {
      selects_has_kind: page.probe("'kind' in SELECTS"),
      selects_keys: page.probe("Object.keys(SELECTS)"),
      label: page.probe("SELECTS.kind"),
      options: select ? select._options : null,
      kind_option_values: select ? select._options.map((o) => o.value) : null,
      kind_option_texts: select ? select._options.map((o) => o.text) : null,
      // Depois de `loadFacets` as options continuam la (o risco apontado pelo
      // brief: a iteracao de facets alcancar `kind` e apagar as 3 options).
      options_after_loadfacets: select
        ? select._options.map((o) => o.value)
        : null,
    };
  },

  // RED 2: a linha de relay mostra "repasse" e "—", nunca "0".
  async relay_row() {
    const page = buildPage();
    page.render([RELAY_EVENT], 1);
    return {
      rows_html: page.registry.rows.innerHTML,
      served_label: page.probe(`servedLabel(${JSON.stringify(RELAY_EVENT)})`),
      served_label_model: page.probe(`servedLabel(${JSON.stringify(MODEL_EVENT)})`),
    };
  },

  // RED 3: escolher o filtro na tela poe `kind` na URL. Roda DEPOIS do RED 1
  // (o select so existe com a mudanca), mas o cenario e independente: ele
  // dispara o listener que o proprio `SELECTS` registra.
  async filter_url() {
    const page = buildPage();
    page.change("kind", "relay");
    return {
      history: page.historyLog,
      state: page.probe("JSON.stringify(state)"),
      params_sent: page.fetchLog,
      // O pedido a `/api/requests` e o que prova que o filtro chegou a busca,
      // e nao so a URL.
      request_query: page.fetchLog.filter((u) => u.includes("/api/requests?")),
    };
  },

  // RED 3b: o caminho inverso -- um link `?kind=relay` preenche o select.
  async filter_from_url() {
    const page = buildPage("?kind=relay");
    return {
      kind_value: page.probe("document.getElementById('kind').value"),
      state: page.probe("JSON.stringify(state)"),
    };
  },

  // RED 4: o detalhe de uma linha de relay omite as secoes de modelo.
  async relay_detail() {
    const page = buildPage();
    await page.openDetail(RELAY_EVENT);
    return {
      detail_html: page.registry.detail.innerHTML,
      heading: page.probe("document.querySelector('#detail h2')?.textContent"),
      // A frase e o que explica por que aquela linha esta na auditoria sem
      // ter modelo nem tokens: sem ela o detalhe parece uma tela quebrada.
      explica: page.probe(
        "document.querySelector('#detail .empty')?.textContent?.trim() ?? null"
      ),
    };
  },

  // GREEN de regressao: a linha de modelo continua como antes.
  async model_row_and_detail() {
    const page = buildPage();
    // O detalhe e desenhado DEPOIS da lista: `openDetail` esvazia o mesmo
    // registro que `render` usou, e ler os dois juntos so faz sentido nesta
    // ordem.
    await page.openDetail(MODEL_EVENT);
    return {
      detail_html: page.registry.detail.innerHTML,
      rows_html: page.registry.rows.innerHTML,
    };
  },
};

async function main() {
  const names = process.argv.slice(2);
  const out = [];
  for (const name of names) {
    const scenario = SCENARIOS[name];
    if (!scenario) throw new Error(`cenario desconhecido: ${name}`);
    // Cenario novo = contexto novo: nenhum estado vaza entre eles.
    out.push([name, await scenario()]);
  }
  process.stdout.write(JSON.stringify(out, null, 2) + "\n");
}

main().catch((err) => {
  process.stderr.write(`erro: ${err && err.stack ? err.stack : err}\n`);
  process.exit(1);
});
