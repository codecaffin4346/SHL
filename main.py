"""
FastAPI Service — SHL Assessment Recommender
GET  /health  → {"status": "ok"}
POST /chat    → Stateless conversational agent
"""
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from typing import Optional

from agent import chat, _get_vector_store, _load_full_catalog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ready
    logger.info("Startup: loading full catalog index...")
    _load_full_catalog()
    logger.info("Startup: pre-loading FAISS vector store...")
    start = time.time()
    _get_vector_store()
    logger.info(f"Vector store loaded in {time.time() - start:.1f}s. Service ready.")
    _ready = True
    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL assessments",
    version="1.0.0",
    lifespan=lifespan,
)


# ──────────────────────────────────────────────────────────────
# Schemas (non-negotiable per spec)
# ──────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v):
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class ChatRequest(BaseModel):
    messages: list[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Readiness probe. Returns 200 when the service is warm."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    if len(request.messages) > 20:
        raise HTTPException(status_code=400, detail="Too many messages (max 20)")

    messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        result = chat(messages_dicts)
        return ChatResponse(
            reply=result["reply"],
            recommendations=[
                Recommendation(**r) for r in result.get("recommendations", [])
            ],
            end_of_conversation=result.get("end_of_conversation", False),
        )
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        # Return a graceful degradation response instead of 500
        return ChatResponse(
            reply="I encountered a technical issue. Please rephrase your request and try again.",
            recommendations=[],
            end_of_conversation=False,
        )


# ──────────────────────────────────────────────────────────────
# Exception handler for validation errors
# ──────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
