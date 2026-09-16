"""FastAPI service for the two-stage ranker's recommendations.

Serves what ``api/precompute.py`` cached: the model's top 12 per customer, what
that customer actually bought in the held-out week, and the product photos.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.store import RecommendationStore

app = FastAPI(title="Aurora", description="H&M fashion recommendations from a two-stage retrieval and ranking model")
app.add_middleware(
    CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["GET"], allow_headers=["*"]
)
store = RecommendationStore()


@app.get("/api/metrics")
def metrics() -> dict:
    return store.metrics


@app.get("/api/customers")
def customers(limit: int = Query(40, ge=1, le=200), q: str | None = None) -> list[dict]:
    return store.summaries(limit, q)


@app.get("/api/customers/{customer_id}")
def customer(customer_id: str) -> dict:
    detail = store.detail(customer_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Unknown customer")
    return detail


@app.get("/api/images/{article_id}")
def image(article_id: int) -> FileResponse:
    path = store.image_path(article_id)
    if path is None:
        raise HTTPException(status_code=404, detail="No photo for this article")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})
