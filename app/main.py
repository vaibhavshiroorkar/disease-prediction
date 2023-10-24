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
from fastapi.responses import JSONResponse

from app.config import ALLOWED_ORIGINS, API_TITLE, API_VERSION, PRELOAD_MODELS
from app.registry import ModelNotAvailable, registry
from app.routers import meta, predict
from app.schemas import HealthResponse

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

    yield


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "Ranks likely conditions from reported symptoms using a Random Forest "
        "trained on a public symptom-disease dataset."
    ),
    lifespan=lifespan,
)

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


@app.get("/", tags=["meta"], summary="Service description")
def root() -> dict[str, object]:
    return {
        "service": "disease-prediction-api",
        "version": API_VERSION,
        "docs": "/docs",
        "endpoints": [
            "GET  /health",
            "GET  /symptoms",
            "GET  /diseases",
            "GET  /diseases/{name}/symptoms",
            "POST /predict/symptoms",
        ],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"], summary="Liveness and model status")
def health() -> HealthResponse:
    """Reports whether the model artifact loaded.

    Returns 200 with `degraded` rather than failing, so the cause is visible in
    the payload instead of an opaque connection error.
    """
    status = registry.status()
    healthy = all(entry["loaded"] for entry in status.values())
    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=API_VERSION,
        models=status,
    )
