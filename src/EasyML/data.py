import numpy as np
import pandas as pd
from typing import BinaryIO
from sklearn.impute import SimpleImputer


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

def get_summary_statistics(df: pd.DataFrame) -> dict:
    """
    Function to compute summary statistcs for the dataframe based off of the type of column being 'numerical' or 'categorical'
    """
    # Numeric Statistics
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        numeric_summary = numeric_df.describe().T
        numeric_summary['Median'] = numeric_df.median()
        numeric_summary['Mode'] = numeric_df.mode().iloc[0] if not numeric_df.mode().empty else None # Select First mode if multiple exist
        numeric_summary['Skew'] = numeric_df.skew()
        numeric_summary['Kurtosis'] = numeric_df.kurtosis()
    else:
        numeric_summary = pd.DataFrame

    # Categorical Statistics
    categorical_df = df.select_dtypes(exclude=['number'])
    if not categorical_df.empty:
        categorcial_summary = pd.DataFrame({
            'Count': categorical_df.count(),
            'Unique Values': categorical_df.nunique(),
            'Top': categorical_df.mode().iloc[0] if not categorical_df.mode().empty else None,
            'Top Freq': categorical_df.apply(lambda x: x.value_counts().iloc[0] if not x.value_counts().empty else None)
        })
    else:
        categorcial_summary = pd.DataFrame

    return {'numeric': numeric_summary, 'categorical': categorcial_summary}

def recommend_imputation(df: pd.DataFrame, target_column: str = None) -> dict:
    """
    Function to analyze datafrane and recommends an imputation strategy to the user for features.
    
    For Numeric Features:
    - If more than 50% values are missing, recommend 'Drop'.
    - If column is highly skewed i.e. skew>1, recommend 'Median'.
    - Otherwise recommend 'Mean'.

    For Caregorical Features:
    - If the column is the target ccolumn, skip recommendation.
    - If >50% missing, recommend 'Drop'.
    - Otherwise, recommend 'Most Frequent'.

    Returns:
        A dictionary ith keys, 'Numeric','Categorical' mapped to column names to recommend strategies.
    """

    recommendations = {'Numeric': {}, 'Categorical': {}}

    # Numeric Columns
    numeric_df = df.select_dtypes(include = ['number'])
    for col in numeric_df.columns:
        if target_column and col == target_column:
            continue
        missing_pct = numeric_df[col].isna().mean()
        skew_val = numeric_df[col].skew()
        if missing_pct > 0.5:
            rec = 'Drop'
        else:
            rec = 'Median' if abs(skew_val) > 1 else "Mean"
        recommendations['Numeric'][col] = rec
    
    # Categorical Columns
    categorical_df = df.select_dtypes(exclude = ['number'])
    for col in categorical_df.columns:
        if target_column and col == target_column:
            continue

        missing_pct = categorical_df[col].isna().mean()
        rec = 'Drop' if missing_pct > 0.5 else 'Most Frequent'
        recommendations['Categorical'][col] = rec

    return recommendations