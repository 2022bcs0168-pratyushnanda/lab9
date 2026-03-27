import os
import sys
import json
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_data
from src.preprocessing import preprocess
from src.model import get_model


X, y = load_data("dataset/housing.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_test = preprocess(X_train, X_test, scale=True)

model = get_model()
model.fit(X_train, y_train)

preds = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

os.makedirs("model", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

joblib.dump(model, "model/model.joblib")

metrics = {
    "dataset_size": int(len(X_train)),
    "rmse": float(rmse),
    "r2": float(r2)
}

with open("artifacts/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Training samples: {len(X_train)}")
print(f"RMSE={rmse}")
print(f"R2={r2}")

