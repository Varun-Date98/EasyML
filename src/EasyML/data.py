import numpy as np
import pandas as pd
from collections import Counter
from typing import BinaryIO
from sklearn.impute import SimpleImputer


categorical_impute_options = ["Drop", "Most Frequent", "Custom Value"]
numeric_impute_options = ["Drop", "Mean", "Median", "Mode", "Custom Value"]

def read_file(file: BinaryIO):
    """
    Read and return the uploaded file as a pandas DataFrame
    Args:
        file: uploaded file

    Returns:
        DataFrame: File read as a DataFrame
    """
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

def impute(df: pd.DataFrame, strategy: str, dtype: str, value: float = 0.0, category: str = ""):
    """
    Imputes missing values in the data according to the selected strategy

    Args:
        df (DataFrame): input DataFrame
        strategy (str): One of [Drop, Most Frequent, Custom Value] for categorical columns and
                        one of [Drop, Mean, Median, Mode, Custom Value] for numeric columns
        dtype (str): Either Categorical or Numeric depending on which type of column is being worked on
        value (float): User given value for numeric column imputation
        category (str): User given value for categorical column imputation

    Returns:
        DataFrame: A copy of the input DataFrame after handling missing values
    """
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

    if dtype == "Categorical":
        custom_value = category
    else:
        custom_value = value

    imputer = SimpleImputer(strategy=strategy.lower(), fill_value=custom_value)

    for col in columns:
        col_dt = df[col].dtype

        if dtype == "Numeric":
            df[col] = df[col].astype(float)

        df[col] = imputer.fit_transform(df[[col]]).ravel()
        df[col] = df[col].astype(col_dt)

    return df

def get_summary_statistics(df: pd.DataFrame) -> dict:
    """
    Function to compute summary statistics for the dataframe based off of the type of column being 'numerical' or 'categorical'
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
        categorical_summary = pd.DataFrame({
            'Count': categorical_df.count(),
            'Unique Values': categorical_df.nunique(),
            'Top': categorical_df.mode().iloc[0] if not categorical_df.mode().empty else None,
            'Top Freq': categorical_df.apply(lambda x: x.value_counts().iloc[0] if not x.value_counts().empty else None)
        })
    else:
        categorical_summary = pd.DataFrame

    return {'numeric': numeric_summary, 'categorical': categorical_summary}

def recommend_imputation(df: pd.DataFrame, target_column: str = None) -> dict:
    """
    Function to analyze dataframe and recommends an imputation strategy to the user for features.
    
    For Numeric Features:
    - If more than 50% values are missing, recommend 'Drop'.
    - If column is highly skewed i.e. skew>1, recommend 'Median'.
    - Otherwise recommend 'Mean'.

    For Categorical Features:
    - If the column is the target column, skip recommendation.
    - If >50% missing, recommend 'Drop'.
    - Otherwise, recommend 'Most Frequent'.

    Majority Vote:
    - For Numeric columns, gather each column's recommendation, then pick the overall majority.
    - If there's a tie, suggest both.
    - For categorical columns, do the same.


    Returns:
        A dictionary with two keys, "Numeric" and "Categorical"
        Each is itself a dictionary with:
        {
            "columns": { col_name: recommendation},
            "majority_recommendation": <single recommendation or tie string>,
            "implications": <string explaining the recommendation or tie>
        }
    """

    # Calculate per column recommendations
    numeric_recs = {}
    categorical_recs = {}

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
        
        numeric_recs[col] = rec
    
    # Categorical Columns
    categorical_df = df.select_dtypes(exclude = ['number'])
    for col in categorical_df.columns:
        if target_column and col == target_column:
            continue

        missing_pct = categorical_df[col].isna().mean()
        if missing_pct > 0.5:
            rec = 'Drop'
        else:
            rec = 'Most Frequent'
        
        categorical_recs[col] = rec

    # Compute majority
    numeric_majority, numeric_implications = get_majority_vote(numeric_recs.values(), feature_type='numeric')
    categorical_majority, categorical_implications = get_majority_vote(categorical_recs.values(), feature_type='categorical')

    # Final output dictionary
    recommendations = {
        "Numeric": {
            "columns": numeric_recs,
            "majority_recommendation": numeric_majority,
            "implications": numeric_implications
        },
        "Categorical": {
            "columns": categorical_recs,
            "majority_recommendation": categorical_majority,
            "implications": categorical_implications
        }
    }

    return recommendations

def get_majority_vote(recommendations, feature_type="numeric"):
    """
    Helper function to find the majority recommendation. 
    If there is a tie, returns a combined recommendation and a note on implications.
    """

    # Edge case: if no recommendations exist, return empty
    if not recommendations:
        return "", ""

    # Count how many times each recommendation occurs
    counter = Counter(recommendations)

    # Find the highest count
    max_count = max(counter.values())
    # Collect all recs with that count
    winners = [rec for rec, cnt in counter.items() if cnt == max_count]

    # If we have a single winner
    if len(winners) == 1:
        majority_recommendation = winners[0]
        implications = get_implications(majority_recommendation, feature_type)
    else:
        # Tie -> join them with an "or"
        majority_recommendation = " or ".join(winners)
        implications = (
            "There is a tie between the recommended strategies. "
            "Choose based on your data characteristics:\n\n"
            + "\n\n".join([get_implications(w, feature_type) for w in winners])
        )

    return majority_recommendation, implications


def get_implications(recommendation, feature_type="numeric"):
    """
    Function to return a short explanation or 'implications' for each recommendation.
    """
    # Some example text
    if feature_type == "numeric":
        if recommendation == "Mean":
            return (
                "Mean is usually suitable for data that is not heavily skewed. "
                "However, it can be sensitive to outliers."
            )
        elif recommendation == "Median":
            return (
                "Median is robust against outliers and is often preferred for skewed data."
            )
        elif recommendation == "Drop":
            return (
                "Dropping rows with null values can lead to loss of data. "
                "Use only if a large fraction of data is missing."
            )
        else:
            return ""
    else:
        # categorical
        if recommendation == "Most Frequent":
            return (
                "Replacing missing values with the most frequent category can work well, "
                "but may bias the distribution toward that category."
            )
        elif recommendation == "Drop":
            return (
                "Dropping rows with missing values may cause data loss. "
                "Use only if a large fraction of data is missing."
            )
        else:
            return ""

def check_target_column_null_values(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise KeyError(f"{target} not found in dataframe columns")

    nulls = df[target].isna().any()

    if nulls:
        raise ValueError(f"Null values present in target ({target}) column. Please make sure there are "
                         f"no null values in the target column.")
