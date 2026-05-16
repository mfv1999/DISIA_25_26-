from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

import time
from prometheus_client import make_asgi_app
from app.monitoring.metrics import REQUEST_COUNT, REQUEST_IN_PROGRESS, REQUEST_LATENCY, collect_system_metrics
from app.api.routes.analysis import router as analysis_router
from app.config import settings
from app.preprocessing.feature_extraction import load_lexicons


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.lexicons = load_lexicons(settings.LEXICON_DIR)
    app.state.negation_window = settings.NEGATION_WINDOW
    collect_system_metrics()
    yield


app = FastAPI(
    title    = settings.APP_NAME,
    version  = settings.APP_VERSION,
    lifespan = lifespan,
)

app.mount("/metrics", make_asgi_app())

@app.middleware("http")
async def track_requests(request: Request, call_next):
    if request.url.path == "/metrics":
        return await call_next(request)
    REQUEST_IN_PROGRESS.inc()
    start    = time.perf_counter()
    response = await call_next(request)
    latency  = time.perf_counter() - start
    REQUEST_IN_PROGRESS.dec()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(latency)
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status_code=response.status_code).inc()
    collect_system_metrics()
    return response
    
app.include_router(analysis_router)