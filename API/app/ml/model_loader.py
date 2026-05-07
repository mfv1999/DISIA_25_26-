from pathlib import Path
import joblib

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "emotion_model.joblib"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.joblib"

_model = None
_label_encoder = None


def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def load_label_encoder():
    global _label_encoder
    if _label_encoder is None:
        _label_encoder = joblib.load(LABEL_ENCODER_PATH)
    return _label_encoder