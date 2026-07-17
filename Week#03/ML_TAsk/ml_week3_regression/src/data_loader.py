
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
  
    df = pd.read_csv(path)
    if df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])
    return df


def check_data_quality(df: pd.DataFrame) -> None:
   
    print("Shape:", df.shape)
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nSummary statistics:\n", df.describe())


def split_features_target(df: pd.DataFrame, target_col: str = "sales"):
  
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def get_train_test_split(X, y, test_size: float = 0.2, random_state: int = 42):

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
