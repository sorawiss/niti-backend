from fastapi import FastAPI

from app.api.rag import router as rag_router


app = FastAPI(
    title="Thai Labor Law Contract Review API",
    version="0.1.0",
)


@app.get("/")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "thai-labor-law-contract-review",
    }


app.include_router(rag_router)
