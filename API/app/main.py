from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.analysis import router as analysis_router
from app.config import settings
from app.preprocessing.feature_extraction import load_lexicons


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.lexicons        = load_lexicons(settings.LEXICON_DIR)
    app.state.negation_window = settings.NEGATION_WINDOW
    yield


app = FastAPI(
    title    = settings.APP_NAME,
    version  = settings.APP_VERSION,
    lifespan = lifespan,
)

app.include_router(analysis_router)
