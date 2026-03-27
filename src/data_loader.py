import pandas as pd

def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]
    return X, y

