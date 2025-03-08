import numpy as np
import pandas as pd
from typing import BinaryIO


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
