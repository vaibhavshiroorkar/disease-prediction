"""FastAPI application.

    uvicorn app.main:app --reload

The model artifact is loaded during startup, not on first request, so a cold
process fails its health check rather than serving a slow first prediction.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import db
from app.config import (
    ALLOWED_ORIGINS,
    API_TITLE,
    API_VERSION,
    MAX_REQUEST_BYTES,
    PRELOAD_MODELS,
    RATE_LIMIT_PER_MINUTE,
    STATIC_DIR,
)
from app.registry import ModelNotAvailable, registry
from app.routers import meta, predict, review, risk
from app.schemas import HealthResponse
from app.security import RequestGuard, SlidingWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if PRELOAD_MODELS:
        started = time.perf_counter()
        registry.preload()
        log.info("model preload finished in %.2fs", time.perf_counter() - started)
    else:
        log.info("PRELOAD_MODELS disabled, the artifact loads on first use")

    # Opens the pool and applies the schema. Logs and continues if Postgres is
    # unreachable, because logging predictions is not part of serving them.
    db.init_pool()
    yield
    db.close_pool()


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "Ranks likely conditions from reported symptoms. Estimates the risk of heart "
        "disease, diabetes and liver disease from clinical measurements, and the risk that a "
        "confirmed dengue case turns serious. Reads single blood cell images for malaria. "
        "Optional PostgreSQL logging stores predictions for later review. "
        "History and review routes require an administrator API key."
    ),
    lifespan=lifespan,
)

app.state.prediction_limiter = SlidingWindow(limit=RATE_LIMIT_PER_MINUTE)
app.add_middleware(RequestGuard, state=app.state, max_body=MAX_REQUEST_BYTES)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Record how long a request took, for spotting cold-start stalls in logs."""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
    return response


@app.exception_handler(ModelNotAvailable)
async def model_unavailable_handler(_: Request, exc: ModelNotAvailable) -> JSONResponse:
    """Any artifact problem that escapes a route becomes a 503, never a 500."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


app.include_router(meta.router)
app.include_router(predict.router)
app.include_router(review.router)
app.include_router(risk.router)
app.include_router(risk.review_router)


@app.get("/api", tags=["meta"], summary="Service description")
def root() -> dict[str, object]:
    return {
        "service": "disease-prediction-api",
        "version": API_VERSION,
        "docs": "/docs",
        "endpoints": [
            "GET  /health",
            "GET  /ready",
            "GET  /symptoms",
            "GET  /diseases",
            "GET  /diseases/{name}/symptoms",
            "GET  /models",
            "GET  /models/{name}",
            "POST /predict/symptoms",
            "POST /predict/heart",
            "POST /predict/diabetes",
            "POST /predict/liver",
            "POST /predict/dengue",
            "POST /predict/malaria",
            "GET  /predictions",
            "GET  /predictions/wrong",
            "GET  /predictions/stats",
            "POST /predictions/{id}/review",
            "GET  /risk-predictions",
            "GET  /risk-predictions/stats",
            "POST /risk-predictions/{id}/review",
        ],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"], summary="Liveness and model status")
def health() -> HealthResponse:
    """Reports model load state and whether prediction logging is live.

    Returns 200 when the database is down: the service still predicts, it just
    stops recording. `status` is `degraded` only when the model is unavailable.
    """
    status = registry.status()
    healthy = all(entry["loaded"] for entry in status.values())
    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=API_VERSION,
        models=status,
        database={"logging_enabled": db.enabled(), "configured": db.configured()},
    )


@app.get("/ready", tags=["meta"], summary="Deployment readiness")
def ready() -> JSONResponse:
    if PRELOAD_MODELS:
        available = all(entry["loaded"] for entry in registry.status().values())
    else:
        # Lazy loading reads each artifact on first use, so the service is ready
        # once every artifact is on disk and none has failed to load.
        available = registry.loadable()
    return JSONResponse({"ready": available}, status_code=200 if available else 503)


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR, check_dir=False), name="static")
