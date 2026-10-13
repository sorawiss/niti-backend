import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.api.rag import router as rag_router
from app.chat.service import ChatService
from app.llm.factory import create_llm_gateway


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm_gateway = create_llm_gateway()
    app.state.chat_service = ChatService(llm_gateway)
    try:
        yield
    finally:
        await llm_gateway.close()


app = FastAPI(
    title="Thai Labor Law Contract Review API",
    version="0.2.0",
    lifespan=lifespan,
)

allow_origins_str = os.getenv(
    "ALLOW_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)
allow_origins = [
    origin.strip()
    for origin in allow_origins_str.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "thai-labor-law-contract-review",
    }


app.include_router(chat_router)
app.include_router(rag_router)
