"""A analise do periodo: o Shunt manda o proprio historico para um modelo.

Tres decisoes que valem mais que o codigo:

- **A chamada sai pelo proprio proxy.** O pedido vira um `ShuntRequest` e vai
  para o mesmo `dispatch` que atende o harness, com a mesma resolucao, a mesma
  cadeia e as mesmas credenciais. Um cliente HTTP separado aqui teria a propria
  configuracao, e a analise passaria a funcionar (ou quebrar) por motivos que
  nao valem para o resto do proxy. O efeito colateral e desejado: a analise
  aparece no painel como qualquer outra requisicao.
- **O dossie e guardado com o texto.** Quem le "tire a ferramenta X, oferecida
  900 vezes e chamada 3" precisa poder conferir os dois numeros. O periodo ja
  passou, e a mesma consulta amanha devolve outro valor -- entao o numero tem de
  ser guardado, nao recalculado.
- **Analise que falhou nao vira cache.** A janela e a chave, e um 502 gravado
  sob ela devolveria o mesmo erro para sempre sem nunca tentar de novo.
"""

import asyncio
import hashlib
import json
import time
from datetime import UTC, datetime

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

# Teto da resposta. Uma analise util cabe em algumas paginas; pedir mais so
# aumenta a conta e o tempo de espera de quem clicou no botao.
from app.config.config import ANALYSIS_MAX_OUTPUT_TOKENS as MAX_OUTPUT_TOKENS
from app.config.settings import Settings
from app.core.dispatcher import ShuntRequest, dispatch
from app.stats import dossier as dossie_mod
from app.stats.models import Analysis

SYSTEM = """Você é um especialista em fluxo de desenvolvimento assistido por modelos.
Recebe o dossiê de um período de uso de um proxy de LLM (o Shunt, que traduz
entre o protocolo Anthropic e provedores compatíveis com a OpenAI) e responde
como melhorar esse fluxo: skills, plugins, ferramentas, prompts, escolha de
modelo e configuração do harness.

Regras da resposta:

1. Recomendações acionáveis, ordenadas por impacto — a primeira é a que muda
   mais, e diga por quê.
2. Toda recomendação cita o número do dossiê que a sustenta. Uma recomendação
   sem número é conselho genérico, e não serve.
3. Quando o dossiê não sustentar uma conclusão, diga que não sustenta em vez de
   preencher a lacuna. `sampled: true` significa que os agregados descrevem uma
   amostra do período, e não o período inteiro.
4. Em `volume.cache`, `silent_requests` são requisições cujo provedor não
   informa cache — ausência de dado, e não ausência de cache. `hit_rate` é
   calculado só sobre `input_tokens_of_reported`. Não recomende ligar cache sem
   olhar esses números: um provedor pode já estar cacheando sozinho.
5. Português do Brasil, sem preâmbulo e sem elogio ao dossiê.
"""

INSTRUCTION = (
    "Dossiê do período (JSON). Analise e responda segundo as regras.\n\n"
)


def _text_of(body: dict) -> str:
    """O texto da resposta Anthropic, que e o unico formato que sai daqui.

    `dispatch` ja traduziu de volta para o dialeto do pedido, e o pedido e
    sempre Anthropic -- entao ler `content` basta, e ler `choices` aqui seria
    codigo para um caminho que nao existe.
    """
    blocks = body.get("content")
    if not isinstance(blocks, list):
        return ""
    return "\n".join(
        str(block.get("text", ""))
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


def _error_text(body: dict) -> str:
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("type") or "")
    return str(error or "")


def fingerprint(filters: dict, model: str) -> str:
    """A identidade da janela: mesmo periodo, mesmos filtros, mesmo modelo.

    O modelo entra porque a mesma janela lida por outro modelo e outra analise
    -- reaproveitar ali entregaria o texto de um modelo dizendo ser de outro.
    """
    declared = {
        key: (value.isoformat() if isinstance(value, datetime) else value)
        for key, value in sorted(filters.items())
        if value not in (None, "")
    }
    payload = json.dumps({"filters": declared, "model": model}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _as_row(row: Analysis, *, with_dossier: bool) -> dict:
    item = {
        "id": row.id,
        "created_at": row.created_at.replace(tzinfo=row.created_at.tzinfo or UTC).isoformat(),
        "since": row.since.isoformat() if row.since else None,
        "until": row.until.isoformat() if row.until else None,
        "filters": json.loads(row.filters_json or "{}"),
        "requested_model": row.requested_model,
        "candidate_model": row.candidate_model,
        "provider": row.provider,
        "status": row.status,
        "text": row.text,
        "input_tokens": row.input_tokens,
        "output_tokens": row.output_tokens,
        "duration_ms": row.duration_ms,
    }
    if with_dossier:
        item["dossier"] = json.loads(row.dossier_json or "{}")
    return item


def recent(engine: Engine, limit: int = 20) -> list[dict]:
    """As analises ja pagas, da mais recente para a mais antiga, sem o dossie."""
    with Session(engine) as session:
        rows = list(
            session.scalars(select(Analysis).order_by(Analysis.id.desc()).limit(limit))
        )
    return [_as_row(row, with_dossier=False) for row in rows]


def one(engine: Engine, analysis_id: int) -> dict | None:
    """Uma analise inteira, com o dossie que a gerou, ou None."""
    with Session(engine) as session:
        row = session.get(Analysis, analysis_id)
        return _as_row(row, with_dossier=True) if row is not None else None


def _cached(engine: Engine, mark: str) -> dict | None:
    with Session(engine) as session:
        row = session.scalars(
            select(Analysis)
            .where(Analysis.fingerprint == mark)
            .order_by(Analysis.id.desc())
            .limit(1)
        ).first()
        return _as_row(row, with_dossier=True) if row is not None else None


def _store(engine: Engine, **fields) -> int:
    with Session(engine) as session:
        row = Analysis(**fields)
        session.add(row)
        session.commit()
        return row.id


async def analyse(
    engine: Engine,
    settings: Settings,
    pool,
    *,
    filters: dict | None = None,
    model: str | None = None,
    refresh: bool = False,
    **dossier_options,
) -> dict:
    """Le o periodo, manda para o modelo pelo proprio proxy, e guarda."""
    filters = dict(filters or {})
    requested = model or settings.default_model
    if not requested:
        return {
            "status": 409,
            "error": (
                "nenhum modelo declarado para a análise: configure `default_model` "
                "ou escolha o modelo na tela"
            ),
            "text": "",
            "cached": False,
        }

    mark = fingerprint(filters, requested)
    if not refresh:
        already = _cached(engine, mark)
        if already is not None:
            return {**already, "cached": True}

    dossie = await asyncio.to_thread(dossie_mod.build, engine, filters=filters, **dossier_options)
    body = {
        "model": requested,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": INSTRUCTION + json.dumps(dossie, ensure_ascii=False, indent=2),
            }
        ],
    }
    started = time.monotonic()
    result = await dispatch(
        ShuntRequest(protocol="anthropic", body=body, headers={}, endpoint="messages"),
        settings,
        pool,
    )
    duration_ms = int((time.monotonic() - started) * 1000)
    text = _text_of(result.body)

    if result.status >= 400 or not text:
        # Sem texto num 200 a chamada tambem falhou, so que em silencio: um
        # 200 com `content` vazio viraria uma analise em branco guardada como
        # se fosse boa.
        message = _error_text(result.body) or "o modelo respondeu sem texto"
        return {
            "status": result.status if result.status >= 400 else 502,
            "error": message,
            "text": "",
            "cached": False,
            "dossier": dossie,
            "trace": list(result.trace),
        }

    usage = result.body.get("usage") or {}
    analysis_id = await asyncio.to_thread(
        _store,
        engine,
        created_at=datetime.now(UTC),
        fingerprint=mark,
        since=filters.get("since"),
        until=filters.get("until"),
        filters_json=json.dumps(dossie["period"]["filters"], ensure_ascii=False),
        dossier_json=json.dumps(dossie, ensure_ascii=False),
        requested_model=requested,
        candidate_model=result.real_model,
        provider=result.real_provider,
        status=result.status,
        text=text,
        input_tokens=usage.get("input_tokens") if usage.get("input_tokens") is not None else usage.get("prompt_tokens"),
        output_tokens=usage.get("output_tokens") if usage.get("output_tokens") is not None else usage.get("completion_tokens"),
        duration_ms=duration_ms,
    )
    return {
        "id": analysis_id,
        "status": result.status,
        "text": text,
        "cached": False,
        "dossier": dossie,
        "requested_model": requested,
        "candidate_model": result.real_model,
        "provider": result.real_provider,
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "duration_ms": duration_ms,
    }
