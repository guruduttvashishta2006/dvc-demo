import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

TEST_PATH = Path("data/test.csv")
MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("metrics/metrics.json")
PREDICTIONS_PATH = Path("metrics/predictions.csv")
TARGET_COL = "SalePrice"

def main():
    bundle = joblib.load(MODEL_PATH)
    model, columns = bundle["model"], bundle["columns"]

    df = pd.read_csv(TEST_PATH)
    X = df.drop(columns=[TARGET_COL])
    X = pd.get_dummies(X, drop_first=True).reindex(columns=columns, fill_value=0)
    y = df[TARGET_COL]

    preds = model.predict(X)
    metrics = {
        "mae": mean_absolute_error(y, preds),
        "rmse": root_mean_squared_error(y, preds),
        "r2": r2_score(y, preds),
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"actual": y, "predicted": preds}).to_csv(PREDICTIONS_PATH, index=False)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
