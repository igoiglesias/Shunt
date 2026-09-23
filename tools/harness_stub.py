"""Stub que registra o que o harness chama e responde o minimo valido.

So para o passo 0: nenhuma linha daqui roda em producao.
"""
import argparse, json, re, time, uuid
from collections.abc import Mapping
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

SECRET = {"authorization", "x-api-key", "x-shunt-token", "cookie", "chatgpt-account-id"}

def mask_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {k.lower(): ("***" if k.lower() in SECRET else v) for k, v in headers.items()}

def mask_path(path: str) -> str:
    return re.sub(r"^/t/[^/]+/", "/t/***/", path)

def mask_query(query: str) -> str:
    return re.sub(r"(^|&)token=[^&]*", r"\1token=***", query)

def _sse(name: str, data: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()

def responses_sse(model: str) -> list[bytes]:
    rid = f"resp_{uuid.uuid4().hex}"
    item = {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}]}
    resp = {"id": rid, "object": "response", "model": model, "status": "in_progress", "output": []}
    return [
        _sse("response.created", {"type": "response.created", "sequence_number": 0, "response": resp}),
        _sse("response.output_item.added", {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": {**item, "status": "in_progress", "content": []}}),
        _sse("response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": 2, "output_index": 0, "content_index": 0, "item_id": "msg_1", "delta": "ok"}),
        _sse("response.output_item.done", {"type": "response.output_item.done", "sequence_number": 3, "output_index": 0, "item": item}),
        _sse("response.completed", {"type": "response.completed", "sequence_number": 4, "response": {**resp, "status": "completed", "output": [item], "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}}),
    ]

def build_stub(log_path: Path) -> FastAPI:
    app = FastAPI()

    async def log(request: Request, body: bytes) -> None:
        line = {"at": time.time(), "method": request.method, "path": mask_path(request.url.path),
                "query": mask_query(request.url.query), "headers": mask_headers(request.headers),
                "body": body[:4000].decode("utf-8", "replace")}
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"])
    async def catch(path: str, request: Request):
        raw = await request.body()
        await log(request, raw)
        body = json.loads(raw) if raw[:1] == b"{" else {}
        model = body.get("model", "stub")
        if path.endswith("v1/responses") and body.get("stream"):
            return StreamingResponse(iter(responses_sse(model)), media_type="text/event-stream")
        if path.endswith("v1/messages"):
            if body.get("stream"):
                events = [("message_start", {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": [], "stop_reason": None, "usage": {"input_tokens": 1, "output_tokens": 0}}}),
                          ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                          ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
                          ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                          ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
                          ("message_stop", {"type": "message_stop"})]
                return StreamingResponse(iter([_sse(n, d) for n, d in events]), media_type="text/event-stream")
            return {"id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}
        if path.endswith("chat/completions"):
            return {"id": "chatcmpl-1", "object": "chat.completion", "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
        if path.endswith("models"):
            return {"object": "list", "data": [{"id": "stub", "object": "model", "created": 0, "owned_by": "stub"}], "models": []}
        return JSONResponse({})
    return app

if __name__ == "__main__":
    import uvicorn
    p = argparse.ArgumentParser(); p.add_argument("--port", type=int, default=8099); p.add_argument("--log", default="docs/superpowers/measurements/2026-09-23-harness-stub.jsonl")
    a = p.parse_args(); Path(a.log).parent.mkdir(parents=True, exist_ok=True)
    uvicorn.run(build_stub(Path(a.log)), host="127.0.0.1", port=a.port, log_level="warning")
