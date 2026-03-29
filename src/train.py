import os
import sys
import json
import joblib
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data
from src.preprocessing import preprocess
from src.model import get_model


EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Housing-Price-Prediction")


def _log_model_params(model, used_keys=None):
    if used_keys is None:
        used_keys = set()
    if hasattr(model, "get_params"):
        for key, value in model.get_params().items():
            if key in used_keys:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                mlflow.log_param(key, value)


X, y = load_data("dataset/housing.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_test = preprocess(X_train, X_test, scale=True)

os.makedirs("model", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run():
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("split_random_state", 42)
    mlflow.log_param("scale_features", True)
    mlflow.log_param("train_samples", int(len(X_train)))
    mlflow.log_param("test_samples", int(len(X_test)))

    model = get_model()
    _log_model_params(
        model,
        used_keys={"test_size", "split_random_state", "scale_features", "train_samples", "test_samples"},
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    joblib.dump(model, "model/model.joblib")

    metrics = {
        "dataset_size": int(len(X_train)),
        "rmse": float(rmse),
        "r2": float(r2)
    }

    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    mlflow.log_metric("rmse", float(rmse))
    mlflow.log_metric("r2", float(r2))
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("model/model.joblib")
    mlflow.log_artifact("artifacts/metrics.json")

print(f"Training samples: {len(X_train)}")
print(f"RMSE={rmse}")
print(f"R2={r2}")

