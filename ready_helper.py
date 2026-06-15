from typing import List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from pydantic import BaseModel, Field

# Config layer — defines what preprocessing steps to apply
class ScalingConfig(BaseModel):
    method: str = Field(..., description="standard, minmax, or robust")
    test_size: float = 0.2
    random_state: int = 42

# Column detection — utility
class ColumnAnalyzer:
    @staticmethod
    def get_numeric_cols(df: pd.DataFrame) -> List[str]:
        return df.select_dtypes(include=['number']).columns.tolist()
    
    @staticmethod
    def get_categorical_cols(df: pd.DataFrame) -> List[str]:
        return df.select_dtypes(include=['object', 'category']).columns.tolist()

# Data handling — operates on DataFrames directly
class DataPipeline:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.scaler = None  # Fitted scaler stored per instance
    
    def train_test_split(
        self, 
        target_col: str, 
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        target_col: name of target column
        feature_cols: list of feature column names (if None, uses all except target)
        """
        if feature_cols is None:
            feature_cols = [c for c in self.df.columns if c != target_col]
        
        X = self.df[feature_cols]
        y = self.df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, y_train, X_test, y_test

# Scaling techniques — grouped by type
class Scalers:
    @staticmethod
    def standard(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit on train, apply to both"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return pd.DataFrame(X_train_scaled, columns=X_train.columns), \
            pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    @staticmethod
    def minmax(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return pd.DataFrame(X_train_scaled, columns=X_train.columns), \
            pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    @staticmethod
    def robust(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return pd.DataFrame(X_train_scaled, columns=X_train.columns), \
pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Usage pattern for Streamlit
if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    
    # Detect columns
    numeric = ColumnAnalyzer.get_numeric_cols(df)
    categorical = ColumnAnalyzer.get_categorical_cols(df)
    
    # Setup pipeline
    pipeline = DataPipeline(df)
    X_train, y_train, X_test, y_test = pipeline.train_test_split(
        target_col="target", 
        feature_cols=numeric
    )
    
    # Scale
    X_train_scaled, X_test_scaled = Scalers.standard(X_train, X_test)