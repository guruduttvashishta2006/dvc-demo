import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["preprocess"]

RAW_PATH = Path("data/house_prices.csv")
TRAIN_PATH = Path("data/train.csv")
TEST_PATH = Path("data/test.csv")

def main():
    df = pd.read_csv(RAW_PATH)
    df = df.dropna(how="all").drop_duplicates()

    train_df, test_df = train_test_split(
        df, test_size=params["test_size"], random_state=params["random_state"]
    )

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"train: {len(train_df)} rows -> {TRAIN_PATH}")
    print(f"test:  {len(test_df)} rows -> {TEST_PATH}")

if __name__ == "__main__":
    main()
