import numpy as np
import pandas as pd
from typing import BinaryIO
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


categorical_impute_options = ["Most Frequent", "Custom Value"]
numeric_impute_options = ["Drop", "Mean", "Median", "Mode", "Custom Value"]

def read_file(file: BinaryIO):
    df = None

    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)

        if file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
    except Exception as e:
        print(f"Exception occurred while trying to read the file, {e}")
    finally:
        return df

def impute(df: pd.DataFrame, strategy: str, dtype: str, value: float = 0.0):
    columns = []

    for col in df.columns:
        cmp = np.issubdtype(df[col].dtype, np.number)

        if dtype == "Numeric" and cmp:
            columns.append(col)

        if dtype == "Categorical" and not cmp:
            columns.append(col)

    if strategy == "Drop":
        return df.dropna(subset=columns)

    if strategy ==  "Mode" or strategy == "Most Frequent":
        strategy = "most_frequent"

    if strategy == "Custom Value":
        strategy = "constant"

    imputer = SimpleImputer(strategy=strategy.lower(), fill_value=value)

    for col in columns:
        col_dt = df[col].dtype

        if dtype == "Numeric":
            df[col] = df[col].astype(float)

        df[col] = imputer.fit_transform(df[[col]]).ravel()
        df[col] = df[col].astype(col_dt)

    return df
