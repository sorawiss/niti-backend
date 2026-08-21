from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.rag import router as rag_router


app = FastAPI(
    title="Thai Labor Law Contract Review API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
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


app.include_router(rag_router)
