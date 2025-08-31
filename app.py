import os
from typing import List, Literal, Optional, AsyncGenerator

import httpx
from fastapi import FastAPI, HTTPException
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

# ---- FastAPI App ----
app = FastAPI(title="Chat with Qwen2.5 3B (Ollama)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set your frontend origin if you want to lock it down
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Schemas ----

Role = Literal["system", "user", "assistant"]

class Turn(BaseModel):
    role: Role
    content: str = Field(..., min_length=1)

class ChatBody(BaseModel):
    messages: List[Turn]

class ChatResponse(BaseModel):
    content: str


# ---- Helpers ----

def to_ollama_chat(messages: List[Turn]) -> List[dict]:
    """
    Convert our message schema to Ollama's expected format.
    Ollama chat expects: [{"role": "user"|"assistant"|"system", "content": "..."}]
    """
    return [{"role": m.role, "content": m.content} for m in messages]

async def call_ollama_chat(
    client: httpx.AsyncClient,
    messages: List[Turn],
    stream: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_ctx: int = 4096,
) -> httpx.Response:
    """
    Call Ollama's /api/chat endpoint.
    Docs: https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
    """
    payload = {
        "model": MODEL_NAME,
        "messages": to_ollama_chat(messages),
        "stream": stream,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": num_ctx,
        },
        # You can also set a system prompt here. Some model builds honor a system message
        # in the messages list instead. We'll prepend one below if not present.
    }

    # If user didn't include a system message, prepend our default
    if not any(m.role == "system" for m in messages):
        payload["messages"] = [{"role": "system", "content": SYSTEM_PROMPT}] + payload["messages"]

    url = f"{OLLAMA_HOST}/api/chat"
    return await client.post(url, json=payload, timeout=None)


# ---- Endpoints ----

@app.post("/api/chat", response_model=ChatResponse)
async def chat(body: ChatBody):
    """
    Non-streaming chat endpoint.
    Accepts: { "messages": [{role, content}, ...] }
    Returns: { "content": "assistant text" }
    """
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    async with httpx.AsyncClient() as client:
        try:
            resp = await call_ollama_chat(client, body.messages, stream=False)
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Ollama connection error: {e}") from e

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    data = resp.json()
    # Ollama non-streaming response shape:
    # {
    #   "model": "...",
    #   "created_at": "...",
    #   "message": {"role": "assistant", "content": "..."},
    #   "done": true,
    #   ...
    # }
    content = (data.get("message") or {}).get("content", "")
    return ChatResponse(content=content)


@app.post("/api/chat/stream")
async def chat_stream(body: ChatBody):
    """
    Streaming version using Server-Sent Events.
    Emits lines like: `data: {"content": "partial chunk"}` and ends with `data: [DONE]`
    """
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    async def event_gen() -> AsyncGenerator[bytes, None]:
        async with httpx.AsyncClient() as client:
            try:
                resp = await call_ollama_chat(client, body.messages, stream=True)
            except httpx.RequestError as e:
                yield f"event: error\ndata: {str(e)}\n\n".encode("utf-8")
                return

            if resp.status_code != 200:
                yield f"event: error\ndata: {resp.text}\n\n".encode("utf-8")
                return

            # Ollama streaming returns newline-separated JSON objects until done
            async for line in resp.aiter_lines():
                if not line:
                    continue
                try:
                    obj = httpx.Response(200, content=line).json()
                except Exception:
                    # sometimes lines may be split; just skip invalid fragments
                    continue

                # Typical chunk format:
                # {"model":"...","created_at":"...","message":{"role":"assistant","content":"..."},"done":false,...}
                done = obj.get("done", False)
                if done:
                    yield b"data: [DONE]\n\n"
                    break

                msg = (obj.get("message") or {}).get("content", "")
                if msg:
                    payload = {"content": msg}
                    yield f"data: {payload}\n\n".encode("utf-8")

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/api/health")
async def health():
    return JSONResponse({"ok": True, "model": MODEL_NAME, "ollama": OLLAMA_HOST})
