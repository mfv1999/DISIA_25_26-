from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = BASE_DIR / "metrics"

FEEDBACK_LOG = METRICS_DIR / "feedback_log.csv"
RETRAINING_DATASET = METRICS_DIR / "feedback_retraining_dataset.csv"


def build_feedback_retraining_dataset() -> dict:
    if not FEEDBACK_LOG.exists():
        return {
            "created": False,
            "rows": 0,
            "output_path": str(RETRAINING_DATASET),
            "message": "No existe feedback_log.csv",
        }

    df = pd.read_csv(FEEDBACK_LOG)

    if df.empty:
        return {
            "created": False,
            "rows": 0,
            "output_path": str(RETRAINING_DATASET),
            "message": "El feedback_log.csv está vacío",
        }

    required_columns = {"original_text", "corrected_emotion"}
    if not required_columns.issubset(df.columns):
        return {
            "created": False,
            "rows": 0,
            "output_path": str(RETRAINING_DATASET),
            "message": "Faltan columnas necesarias en feedback_log.csv",
        }

    retraining_df = df[["original_text", "corrected_emotion"]].copy()

    retraining_df = retraining_df.rename(
        columns={
            "original_text": "Text",
            "corrected_emotion": "Emotion",
        }
    )

    retraining_df = retraining_df.dropna(subset=["Text", "Emotion"])
    retraining_df["Text"] = retraining_df["Text"].astype(str).str.strip()
    retraining_df["Emotion"] = retraining_df["Emotion"].astype(str).str.strip()

    retraining_df = retraining_df[
        (retraining_df["Text"] != "") & (retraining_df["Emotion"] != "")
    ]

    retraining_df = retraining_df.drop_duplicates()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    retraining_df.to_csv(RETRAINING_DATASET, index=False, encoding="utf-8")

    return {
        "created": True,
        "rows": len(retraining_df),
        "output_path": str(RETRAINING_DATASET),
        "message": "Dataset de reentrenamiento generado correctamente",
    }