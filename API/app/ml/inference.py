import pandas as pd

from app.ml.model_loader import load_label_encoder, load_model
from app.preprocessing.feature_extraction import extract


def predict_emotion(text: str, lexicons, negation_window: int = 3) -> dict:
    model = load_model()
    label_encoder = load_label_encoder()

    features = extract(
        text=text,
        lexicons=lexicons,
        negation_window=negation_window,
    )

    clean_features = {}
    for key, value in features.items():
        if value is None or pd.isna(value):
            clean_features[key] = 0
        else:
            clean_features[key] = value

    row = {
        "Text": text,
        **clean_features,
    }

    X = pd.DataFrame([row]).fillna(0)

    pred = model.predict(X)[0]
    prediction = label_encoder.inverse_transform([pred])[0]

    return {
        "original_text": text,
        "features": clean_features,
        "prediction": str(prediction),
    }