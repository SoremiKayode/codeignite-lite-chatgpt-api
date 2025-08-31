# main.py
import os
import json
import asyncio
from typing import List, Literal, AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse, JSONResponse

load_dotenv()

# ---- Config ----
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:3b-instruct-q4_K_M")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful AI assistant. Use Markdown in responses. Be concise and include code blocks when needed."
)

# Visible typing controls (optional)
STREAM_THROTTLE_MS = int(os.getenv("STREAM_THROTTLE_MS", "0"))   # e.g., 60..120
STREAM_SPLIT_CHARS = int(os.getenv("STREAM_SPLIT_CHARS", "0"))   # e.g., 12

# Timeouts
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)

# CORS (lock down in prod)
app = FastAPI(title="Chat with Qwen2.5 3B (Ollama)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Role = Literal["system", "user", "assistant"]


class Turn(BaseModel):
    role: Role
    content: str = Field(..., min_length=1)


class ChatBody(BaseModel):
    messages: List[Turn]


class ChatResponse(BaseModel):
    content: str


def _to_ollama_chat(messages: List[Turn]) -> List[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _build_payload(messages: List[Turn], stream: bool) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": _to_ollama_chat(messages),
        "stream": stream,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_ctx": 4096},
    }
    if not any(m.role == "system" for m in messages):
        payload["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}] + payload["messages"]
    return payload


async def _post_ollama(client: httpx.AsyncClient, payload: dict) -> httpx.Response:
    return await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)


def _iter_slices(s: str, n: int):
    if n <= 0 or n >= len(s):
        yield s
        return
    for i in range(0, len(s), n):
        yield s[i:i + n]


# ----------------- Non-streaming (for fallbacks) -----------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat(body: ChatBody):
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        try:
            resp = await _post_ollama(client, _build_payload(body.messages, stream=False))
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Ollama connection error: {e}") from e

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    content = (data.get("message") or {}).get("content", "")
    return ChatResponse(content=content)


# ----------------- SSE streaming -----------------
@app.post("/api/chat/stream")
async def chat_stream(body: ChatBody):
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    async def event_gen() -> AsyncGenerator[bytes, None]:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            try:
                resp = await _post_ollama(client, _build_payload(body.messages, stream=True))
            except httpx.RequestError as e:
                yield f"event: error\ndata: {str(e)}\n\n".encode("utf-8")
                return

            if resp.status_code != 200:
                yield f"event: error\ndata: {resp.text}\n\n".encode("utf-8")
                return

            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                if obj.get("done"):
                    yield b"data: [DONE]\n\n"
                    break

                msg = (obj.get("message") or {}).get("content", "")
                if not msg:
                    continue

                for piece in _iter_slices(msg, STREAM_SPLIT_CHARS):
                    if STREAM_THROTTLE_MS > 0:
                        await asyncio.sleep(STREAM_THROTTLE_MS / 1000.0)
                    yield f"data: {json.dumps({'content': piece})}\n\n".encode("utf-8")

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ----------------- NDJSON streaming (diagnostic) -----------------
@app.post("/api/chat/ndjson")
async def chat_ndjson(body: ChatBody):
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    async def gen() -> AsyncGenerator[bytes, None]:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            try:
                resp = await _post_ollama(client, _build_payload(body.messages, stream=True))
            except httpx.RequestError as e:
                yield json.dumps({"error": str(e)}).encode() + b"\n"
                return

            if resp.status_code != 200:
                yield json.dumps({"error": resp.text}).encode() + b"\n"
                return

            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                if obj.get("done"):
                    break

                msg = (obj.get("message") or {}).get("content", "")
                if not msg:
                    continue

                for piece in _iter_slices(msg, STREAM_SPLIT_CHARS):
                    if STREAM_THROTTLE_MS > 0:
                        await asyncio.sleep(STREAM_THROTTLE_MS / 1000.0)
                    yield json.dumps({"content": piece}).encode() + b"\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson; charset=utf-8",
                             headers={"Cache-Control": "no-cache, no-transform"})


# ----------------- WebSocket streaming (robust on Windows) -----------------
@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    try:
        # Expect a JSON message: {"messages": [...]} same shape as ChatBody
        first = await ws.receive_text()
        data = json.loads(first)
        messages = [Turn(**m) for m in data.get("messages", [])]
        if not messages:
            await ws.send_text(json.dumps({"error": "messages must not be empty"}))
            await ws.close()
            return

        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            resp = await _post_ollama(client, _build_payload(messages, stream=True))
            if resp.status_code != 200:
                await ws.send_text(json.dumps({"error": resp.text}))
                await ws.close()
                return

            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                if obj.get("done"):
                    await ws.send_text(json.dumps({"done": True}))
                    break

                msg = (obj.get("message") or {}).get("content", "")
                if not msg:
                    continue

                for piece in _iter_slices(msg, STREAM_SPLIT_CHARS):
                    if STREAM_THROTTLE_MS > 0:
                        await asyncio.sleep(STREAM_THROTTLE_MS / 1000.0)
                    await ws.send_text(json.dumps({"content": piece}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
    finally:
        await ws.close()


# ----------------- Health -----------------
@app.get("/api/health")
async def health():
    return JSONResponse({"ok": True, "model": MODEL_NAME, "ollama": OLLAMA_HOST})
