# server.py
# requirements: fastapi uvicorn[standard] httpx
import os
from typing import List, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── 내부 Ollama 주소 (게이트웨이 ↔ Ollama 로컬 통신)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ── 데이터 모델 (단축 엔드포인트 /chat에서 사용)
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    model: str = "llama3.2"  # 기본값. 가벼운 tinyllama 권장 가능
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    reply: str
    model: str

# ── FastAPI 앱
app = FastAPI(
    title="Raspberry Pi LLM Gateway",
    description="FastAPI(8080) ↔ Ollama(11434) Proxy Gateway",
    version="1.0.0",
)

# (옵션) 브라우저 직접 호출 필요하면 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ── Health Check
@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
        if r.status_code == 200:
            return {"status": "ok", "ollama": "connected"}
        return {"status": "ok", "ollama": f"unreachable ({r.status_code})"}
    except Exception as e:
        return {"status": "ok", "ollama": f"error: {e}"}

# ── 슬라이드 예시 그대로 쓰기 위한 전체 프록시 (/api/*)
@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy_all(path: str, request: Request):
    """
    예: /api/generate, /api/chat, /api/tags, /api/embeddings ...
    클라이언트 → (8080 Gateway) → (11434 Ollama)
    """
    url = f"{OLLAMA_HOST}/api/{path}"
    method = request.method
    body = await request.body()
    headers = {"Content-Type": request.headers.get("content-type", "application/json")}
    timeout = httpx.Timeout(600.0)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.request(method, url, content=body, headers=headers)
        # Ollama의 상태코드/본문을 최대한 그대로 전달
        try:
            content = resp.json()
        except Exception:
            content = {"raw": resp.text}
        return JSONResponse(status_code=resp.status_code, content=content)
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": f"proxy_failed: {e}"})

# ── 편의용: /chat (단일 JSON 응답으로 reply만 추출)
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    payload = {
        "model": req.model,
        "messages": [m.dict() for m in req.messages],
        "stream": False  # 단일 JSON 응답 강제 (파싱/에러 단순화)
    }
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ollama_connect_failed: {e}")

    if r.status_code >= 400:
        # Ollama가 에러를 반환하면 원문 보여주기
        raise HTTPException(status_code=502, detail=f"ollama_error {r.status_code}: {r.text}")

    # 단일 JSON으로 들어온 응답 파싱
    try:
        data = r.json()
        reply = data.get("message", {}).get("content", "")
    except Exception:
        reply = r.text or ""

    return ChatResponse(reply=reply.strip(), model=req.model)

# ── 편의용: /load (미리 모델 올려두기)
@app.post("/load")
async def load_model(req: ChatRequest):
    payload = {"model": req.model, "prompt": ""}
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            r = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
        return JSONResponse(status_code=r.status_code, content=r.json())
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": f"load_failed: {e}"})

# ── 편의용: /unload (메모리 해제)
@app.post("/unload")
async def unload_model(req: ChatRequest):
    payload = {"model": req.model, "keep_alive": 0}
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
        return JSONResponse(status_code=r.status_code, content=r.json())
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": f"unload_failed: {e}"})