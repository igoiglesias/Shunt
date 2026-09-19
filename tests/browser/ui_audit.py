"""Varredura de UI: acha o que o olho deixa passar.

Escrita depois de uma rodada de design em que eu olhei os prints e disse que
estava bom, e o usuario apontou tres defeitos na mesma tela. Medir e o que
impede isso: a varredura reprova texto cortado, transbordo, sobreposicao,
contraste abaixo do minimo e alvo de clique pequeno -- 105 achados na primeira
execucao.

Mede, em cada largura: texto truncado, transbordo horizontal, sobreposicao de
texto, contraste abaixo do minimo, alvo de clique pequeno demais e elemento
cortado pelo pai.
"""

AUDITORIA = r"""() => {
  const problemas = [];
  const visivel = (el) => {
    const e = getComputedStyle(el);
    if (e.display === 'none' || e.visibility === 'hidden' || Number(e.opacity) === 0) return false;
    const c = el.getBoundingClientRect();
    return c.width > 0 && c.height > 0;
  };
  const nome = (el) => {
    const id = el.id ? `#${el.id}` : '';
    const cls = (el.className && typeof el.className === 'string')
      ? '.' + el.className.trim().split(/\\s+/).slice(0, 2).join('.') : '';
    return `${el.tagName.toLowerCase()}${id}${cls}`;
  };
  const texto = (el) => (el.textContent || '').trim().slice(0, 40);

  // 1. Texto cortado dentro do proprio elemento.
  for (const el of document.querySelectorAll('td, th, .label, .value, .context, h2, h3, option, button, a, .mark, .pill, .tag')) {
    if (!visivel(el)) continue;
    if (el.scrollWidth > el.clientWidth + 1 && getComputedStyle(el).overflow !== 'auto') {
      problemas.push({tipo: 'texto cortado', onde: nome(el), texto: texto(el),
                      medida: `${el.scrollWidth} > ${el.clientWidth}`});
    }
  }

  // 2. Transbordo horizontal da pagina.
  if (document.documentElement.scrollWidth > window.innerWidth + 1) {
    problemas.push({tipo: 'pagina rola de lado', onde: 'html',
                    medida: `${document.documentElement.scrollWidth} > ${window.innerWidth}`});
  }

  // 3. Elemento que vaza do pai com overflow visivel.
  for (const el of document.querySelectorAll('section, aside, .group, .figure, .row')) {
    if (!visivel(el)) continue;
    const pai = el.getBoundingClientRect();
    for (const filho of el.children) {
      if (!visivel(filho)) continue;
      const c = filho.getBoundingClientRect();
      if (c.right > pai.right + 2 || c.left < pai.left - 2) {
        problemas.push({tipo: 'filho vaza do pai', onde: `${nome(el)} > ${nome(filho)}`,
                        medida: `${Math.round(c.right)} vs ${Math.round(pai.right)}`});
      }
    }
  }

  // 4. Sobreposicao entre textos irmaos.
  const textos = [...document.querySelectorAll('text, .label, .value, .context, td, .mark')]
    .filter(visivel).slice(0, 400);
  for (let i = 0; i < textos.length; i++) {
    for (let j = i + 1; j < textos.length; j++) {
      const a = textos[i].getBoundingClientRect(), b = textos[j].getBoundingClientRect();
      if (textos[i].contains(textos[j]) || textos[j].contains(textos[i])) continue;
      const sobrepoe = a.left < b.right - 2 && b.left < a.right - 2 && a.top < b.bottom - 2 && b.top < a.bottom - 2;
      if (sobrepoe) {
        problemas.push({tipo: 'texto sobreposto', onde: `${nome(textos[i])} x ${nome(textos[j])}`,
                        texto: `${texto(textos[i])} / ${texto(textos[j])}`});
      }
    }
  }

  // 5. Contraste de texto.
  const lum = (cor) => {
    const [r, g, b] = cor.match(/\\d+(\\.\\d+)?/g).slice(0, 3).map(Number).map(v => {
      const s = v / 255; return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  };
  // Fundo com alfa precisa ser COMPOSTO sobre o que está atrás dele. Ler os
  // três primeiros números de um `rgba(228, 87, 76, 0.08)` como se fossem
  // opacos dava 1.25:1 numa linha que, na tela, tem contraste de sobra.
  const canal = (cor) => (cor.match(/[\d.]+/g) || []).map(Number);
  const sobre = (frente, fundo) => {
    const a = frente.length > 3 ? frente[3] : 1;
    return [0, 1, 2].map((i) => frente[i] * a + fundo[i] * (1 - a));
  };
  const fundoDe = (el) => {
    const camadas = [];
    let no = el;
    while (no && no !== document.documentElement) {
      const bg = canal(getComputedStyle(no).backgroundColor);
      const a = bg.length > 3 ? bg[3] : 1;
      if (a > 0) camadas.push(bg);
      if (a === 1) break;
      no = no.parentElement;
    }
    let resultado = [14, 22, 32];
    for (const camada of camadas.reverse()) resultado = sobre(camada, resultado);
    return `rgb(${resultado.map((v) => Math.round(v)).join(', ')})`;
  };
  for (const el of document.querySelectorAll('.context, .label, .axis, .tag, .empty, small, .mark, th, footer, .chip')) {
    if (!visivel(el) || !texto(el)) continue;
    const e = getComputedStyle(el);
    const fg = lum(e.color), bg = lum(fundoDe(el));
    const razao = (Math.max(fg, bg) + 0.05) / (Math.min(fg, bg) + 0.05);
    const tamanho = parseFloat(e.fontSize);
    const minimo = tamanho >= 18.66 || (tamanho >= 14 && Number(e.fontWeight) >= 700) ? 3 : 4.5;
    if (razao < minimo) {
      problemas.push({tipo: 'contraste baixo', onde: nome(el), texto: texto(el),
                      medida: `${razao.toFixed(2)}:1 (min ${minimo})`});
    }
  }

  // 6. Alvo de clique pequeno.
  for (const el of document.querySelectorAll('button, a, select, input, [role="tab"]')) {
    if (!visivel(el)) continue;
    const c = el.getBoundingClientRect();
    if (c.height < 24 || c.width < 24) {
      problemas.push({tipo: 'alvo pequeno', onde: nome(el), texto: texto(el),
                      medida: `${Math.round(c.width)}x${Math.round(c.height)}`});
    }
  }
  return problemas;
}"""

def auditar(page) -> list[dict]:
    """Os problemas que a pagina tem agora, sem duplicatas."""
    achados = page.evaluate(AUDITORIA)
    vistos = set()
    unicos = []
    for a in achados:
        chave = (a["tipo"], a.get("onde"), a.get("texto", ""))
        if chave in vistos:
            continue
        vistos.add(chave)
        unicos.append(a)
    return unicos


def descrever(achados: list[dict]) -> str:
    return "\n".join(
        f"  - {a['tipo']} | {a.get('onde')} | {a.get('texto', '')} | {a.get('medida', '')}"
        for a in achados
    )
