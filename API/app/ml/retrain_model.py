from pathlib import Path
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = BASE_DIR / "metrics"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

RETRAINING_DATASET = METRICS_DIR / "feedback_retraining_dataset.csv"
RETRAINED_MODEL_PATH = ARTIFACTS_DIR / "emotion_model_retrained.joblib"


def retrain_model_from_feedback() -> dict:
    if not RETRAINING_DATASET.exists():
        return {
            "trained": False,
            "rows": 0,
            "model_path": str(RETRAINED_MODEL_PATH),
            "message": "No existe feedback_retraining_dataset.csv",
        }

    df = pd.read_csv(RETRAINING_DATASET)

    if df.empty:
        return {
            "trained": False,
            "rows": 0,
            "model_path": str(RETRAINED_MODEL_PATH),
            "message": "El dataset de reentrenamiento está vacío",
        }

    required_columns = {"Text", "Emotion"}
    if not required_columns.issubset(df.columns):
        return {
            "trained": False,
            "rows": 0,
            "model_path": str(RETRAINED_MODEL_PATH),
            "message": "Faltan columnas necesarias en el dataset de reentrenamiento",
        }

    X = df["Text"].astype(str)
    y = df["Emotion"].astype(str)

    if len(df) < 2:
        return {
            "trained": False,
            "rows": len(df),
            "model_path": str(RETRAINED_MODEL_PATH),
            "message": "Se necesitan al menos 2 filas para un reentrenamiento simple",
        }

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LinearSVC()),
    ])

    pipeline.fit(X, y)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, RETRAINED_MODEL_PATH)

    return {
        "trained": True,
        "rows": len(df),
        "model_path": str(RETRAINED_MODEL_PATH),
        "message": "Modelo reentrenado y guardado correctamente",
    }