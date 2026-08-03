import yaml
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

params = yaml.safe_load(open("params.yaml"))["train"]

TRAIN_PATH = Path("data/train.csv")
MODEL_PATH = Path("models/model.pkl")
TARGET_COL = "SalePrice"

def main():
    df = pd.read_csv(TRAIN_PATH)
    X = df.drop(columns=[TARGET_COL])
    X = pd.get_dummies(X, drop_first=True)
    y = df[TARGET_COL]

    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"],
    )
    model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": list(X.columns)}, MODEL_PATH)
    print(f"Model trained on {len(X)} rows -> {MODEL_PATH}")

if __name__ == "__main__":
    main()
