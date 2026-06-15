from typing import List, Tuple, Optional, Union
import io
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

from pydantic import BaseModel, Field
from typing import Annotated


# ─────────────────────────────────────────────
# INTERNAL PYDANTIC MODELS
# (not meant to be used directly by the user)
# These carry validated, parsed config after
# the tuple args are unpacked
# ─────────────────────────────────────────────

class _ScalerConfig(BaseModel):
    type: str
    quantile_range: Tuple[float, float] = (25.0, 75.0)  # only for robust

class _EncoderConfig(BaseModel):
    type: str
    categories: Optional[List] = None     # for ordinal
    drop: Optional[str] = None            # for onehot: 'first' or 'if_binary'
    handle_unknown: str = 'ignore'        # for onehot

class _ImputerConfig(BaseModel):
    type: str
    fill_value: Optional[Union[int, float, str]] = None   # for constant
    n_neighbors: int = 5                                   # for knn
    max_iter: int = 10                                     # for iterative

class _OutlierConfig(BaseModel):
    type: str
    treatment: str = 'remove'
    threshold: float = 1.5               # 1.5 for iqr, 3.0 for zscore

class _PipeConfig(BaseModel):
    test_size: Annotated[float, Field(gt=0, lt=1)] = 0.2
    random_state: int = 42


# ─────────────────────────────────────────────
# INTERNAL PARSERS
# Convert user-facing tuple args into configs
# ─────────────────────────────────────────────

def _parse_scaling(args) -> _ScalerConfig:
    if args is None:
        return None
    if isinstance(args, str):
        args = (args,)

    technique = args[0]
    kwargs = args[1] if len(args) > 1 and isinstance(args[1], dict) else {}

    valid = ('standard', 'minmax', 'robust')
    if technique not in valid:
        raise ValueError(f"scaling must be one of {valid}, got '{technique}'")

    return _ScalerConfig(type=technique, **kwargs)


def _parse_encoding(args) -> _EncoderConfig:
    if args is None:
        return None
    if isinstance(args, str):
        args = (args,)

    technique = args[0]
    rest = args[1:]

    valid = ('onehot', 'ordinal', 'label')
    if technique not in valid:
        raise ValueError(f"encoding must be one of {valid}, got '{technique}'")

    config_kwargs = {'type': technique}

    if technique == 'ordinal':
        # ('ordinal', ['low', 'mid', 'high'])
        if rest and isinstance(rest[0], list):
            config_kwargs['categories'] = rest[0]

    elif technique == 'onehot':
        # ('onehot', drop='first')
        if rest and isinstance(rest[0], dict):
            config_kwargs.update(rest[0])

    return _EncoderConfig(**config_kwargs)


def _parse_imputation(args) -> _ImputerConfig:
    if args is None:
        return None
    if isinstance(args, str):
        args = (args,)

    technique = args[0]
    rest = args[1:]

    valid = ('mean', 'median', 'most_frequent', 'constant', 'knn', 'iterative', 'drop_rows', 'drop_cols')
    if technique not in valid:
        raise ValueError(f"imputation must be one of {valid}, got '{technique}'")

    config_kwargs = {'type': technique}

    # pick up optional keyword dict if passed as second element
    if rest and isinstance(rest[0], dict):
        config_kwargs.update(rest[0])

    return _ImputerConfig(**config_kwargs)


def _parse_outliers(args) -> _OutlierConfig:
    if args is None:
        return None
    if isinstance(args, str):
        args = (args,)

    technique = args[0]
    rest = args[1:]

    valid = ('iqr', 'zscore')
    if technique not in valid:
        raise ValueError(f"outlier detection must be one of {valid}, got '{technique}'")

    config_kwargs = {'type': technique}
    if rest and isinstance(rest[0], dict):
        config_kwargs.update(rest[0])

    return _OutlierConfig(**config_kwargs)


# ─────────────────────────────────────────────
# TRANSFORMERS
# Stateless classes, each owns one concern
# ─────────────────────────────────────────────

class _Scaler:
    """Fits on train, transforms both train and test. Numerical cols only."""

    def __init__(self, config: _ScalerConfig):
        self.config = config
        self._fitted = None

    def _get_sklearn_scaler(self):
        if self.config.type == 'standard':
            return StandardScaler()
        elif self.config.type == 'minmax':
            return MinMaxScaler()
        elif self.config.type == 'robust':
            return RobustScaler(quantile_range=self.config.quantile_range)

    def fit_transform(self, train: pd.DataFrame, test: pd.DataFrame, num_cols: List[str]):
        if not num_cols:
            return train, test

        scaler = self._get_sklearn_scaler()
        train = train.copy()
        test = test.copy()

        train[num_cols] = scaler.fit_transform(train[num_cols])
        test[num_cols] = scaler.transform(test[num_cols])

        self._fitted = scaler
        return train, test


class _Encoder:
    """Fits on train, transforms both train and test. Categorical cols only."""

    def __init__(self, config: _EncoderConfig):
        self.config = config
        self._fitted = {}

    def fit_transform(self, train: pd.DataFrame, test: pd.DataFrame, cat_cols: List[str]):
        if not cat_cols:
            return train, test

        train = train.copy()
        test = test.copy()

        if self.config.type == 'label':
            for col in cat_cols:
                enc = LabelEncoder()
                train[col] = enc.fit_transform(train[col].astype(str))
                # LabelEncoder doesn't handle unseen labels — we map unknowns to -1
                test[col] = test[col].map(
                    lambda x: enc.transform([x])[0] if x in enc.classes_ else -1
                )
                self._fitted[col] = enc

        elif self.config.type == 'ordinal':
            categories = self.config.categories
            enc = OrdinalEncoder(
                categories=[categories] * len(cat_cols) if categories else 'auto',
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            train[cat_cols] = enc.fit_transform(train[cat_cols])
            test[cat_cols] = enc.transform(test[cat_cols])
            self._fitted['encoder'] = enc

        elif self.config.type == 'onehot':
            enc = OneHotEncoder(
                sparse_output=False,
                drop=self.config.drop,
                handle_unknown=self.config.handle_unknown
            )
            train_encoded = enc.fit_transform(train[cat_cols])
            test_encoded = enc.transform(test[cat_cols])

            new_cols = enc.get_feature_names_out(cat_cols)
            train = train.drop(columns=cat_cols)
            test = test.drop(columns=cat_cols)
            train[new_cols] = train_encoded
            test[new_cols] = test_encoded
            self._fitted['encoder'] = enc

        return train, test


class _Imputer:
    """Handles missing values. Fits on train only."""

    def __init__(self, config: _ImputerConfig):
        self.config = config
        self._fitted = {}

    def fit_transform(self, train: pd.DataFrame, test: pd.DataFrame):
        t = self.config.type
        train = train.copy()
        test = test.copy()

        if t == 'drop_rows':
            train = train.dropna()
            test = test.dropna()
            return train, test

        if t == 'drop_cols':
            cols_with_nulls = train.columns[train.isnull().any()].tolist()
            train = train.drop(columns=cols_with_nulls)
            test = test.drop(columns=cols_with_nulls)
            return train, test

        if t in ('mean', 'median', 'most_frequent'):
            imputer = SimpleImputer(strategy=t)

        elif t == 'constant':
            imputer = SimpleImputer(strategy='constant', fill_value=self.config.fill_value)

        elif t == 'knn':
            imputer = KNNImputer(n_neighbors=self.config.n_neighbors)

        elif t == 'iterative':
            imputer = IterativeImputer(max_iter=self.config.max_iter, random_state=0)

        cols = train.columns.tolist()
        train[cols] = imputer.fit_transform(train)
        test[cols] = imputer.transform(test)
        self._fitted['imputer'] = imputer

        return train, test


class _OutlierHandler:
    """Detects and treats outliers on numerical cols. Fit on train only."""

    def __init__(self, config: _OutlierConfig):
        self.config = config
        self._bounds = {}  # stores (lower, upper) per col, fitted on train

    def fit_transform(self, train: pd.DataFrame, test: pd.DataFrame, num_cols: List[str]):
        if not num_cols:
            return train, test

        train = train.copy()
        test = test.copy()

        # compute bounds from train only
        for col in num_cols:
            if self.config.type == 'iqr':
                Q1 = train[col].quantile(0.25)
                Q3 = train[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - self.config.threshold * IQR
                upper = Q3 + self.config.threshold * IQR

            elif self.config.type == 'zscore':
                mean = train[col].mean()
                std = train[col].std()
                lower = mean - self.config.threshold * std
                upper = mean + self.config.threshold * std

            self._bounds[col] = (lower, upper)

        # apply treatment
        for col in num_cols:
            lower, upper = self._bounds[col]

            if self.config.treatment == 'remove':
                mask_train = (train[col] >= lower) & (train[col] <= upper)
                train = train[mask_train]
                # for test: same mask logic
                mask_test = (test[col] >= lower) & (test[col] <= upper)
                test = test[mask_test]

            elif self.config.treatment == 'cap':
                train[col] = train[col].clip(lower=lower, upper=upper)
                test[col] = test[col].clip(lower=lower, upper=upper)

        return train, test


# ─────────────────────────────────────────────
# DATA CLASS
# The main user-facing entry point
# ─────────────────────────────────────────────

class data:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def num_cols(self) -> List[str]:
        return self.df.select_dtypes(include=['number']).columns.tolist()

    def cat_cols(self) -> List[str]:
        return self.df.select_dtypes(include=['object', 'category']).columns.tolist()

    def overview(self) -> Tuple[pd.DataFrame, str, pd.DataFrame]:
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        return (
            self.df.head(),
            buffer.getvalue(),
            self.df.describe()
        )

    def feature_transformation(
        self,
        target_col: str,
        scaling: Optional[tuple] = ('standard',),
        encoding: Optional[tuple] = ('ordinal',),
        imputation: Optional[tuple] = ('mean',),
        outliers: Optional[tuple] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> '_TransformedData':
        """
        Main transformation entry point.

        Examples
        --------
        data.feature_transformation(
            target_col  = 'price',
            scaling     = ('robust', {'quantile_range': (10.0, 90.0)}),
            encoding    = ('onehot', {'drop': 'first'}),
            imputation  = ('knn', {'n_neighbors': 3}),
            outliers    = ('iqr', {'treatment': 'cap', 'threshold': 1.5}),
        )
        """
        # ── parse all configs ──────────────────
        scaler_cfg  = _parse_scaling(scaling)
        encoder_cfg = _parse_encoding(encoding)
        imputer_cfg = _parse_imputation(imputation)
        outlier_cfg = _parse_outliers(outliers)

        # ── split X / y ────────────────────────
        df = self.df.copy()
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        num_cols = data(X_train).num_cols()
        cat_cols = data(X_train).cat_cols()

        # ── imputation (before outliers / scaling) ──
        if imputer_cfg:
            X_train, X_test = _Imputer(imputer_cfg).fit_transform(X_train, X_test)
            # cols may have changed if drop_cols was used
            num_cols = data(X_train).num_cols()
            cat_cols = data(X_train).cat_cols()

        # ── outlier handling ───────────────────
        if outlier_cfg:
            X_train, X_test = _OutlierHandler(outlier_cfg).fit_transform(X_train, X_test, num_cols)
            y_train = y_train.loc[X_train.index]
            y_test  = y_test.loc[X_test.index]

        # ── scaling ────────────────────────────
        if scaler_cfg:
            X_train, X_test = _Scaler(scaler_cfg).fit_transform(X_train, X_test, num_cols)

        # ── encoding ───────────────────────────
        if encoder_cfg:
            X_train, X_test = _Encoder(encoder_cfg).fit_transform(X_train, X_test, cat_cols)

        return _TransformedData(X_train, X_test, y_train, y_test)


# ─────────────────────────────────────────────
# TRANSFORMED DATA
# Returned by feature_transformation
# Holds the split, transformed data + fit() method
# ─────────────────────────────────────────────

class _TransformedData:

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

    def fit(self, model):
        """Train a sklearn model and return a _FittedModel."""
        model.fit(self.X_train, self.y_train)
        return _FittedModel(model, self.X_test, self.y_test)

    def shapes(self):
        return {
            'X_train': self.X_train.shape,
            'X_test':  self.X_test.shape,
            'y_train': self.y_train.shape,
            'y_test':  self.y_test.shape,
        }


# ─────────────────────────────────────────────
# FITTED MODEL
# Returned by _TransformedData.fit()
# Holds model + evaluation
# ─────────────────────────────────────────────

class _FittedModel:

    def __init__(self, model, X_test, y_test):
        self.model  = model
        self.X_test = X_test
        self.y_test = y_test
        self.predictions = model.predict(X_test)

    def evaluate(self):
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score,
            accuracy_score, classification_report
        )
        try:
            # try regression metrics first
            mse  = mean_squared_error(self.y_test, self.predictions)
            return {
                'rmse': float(np.sqrt(mse)),
                'mae':  float(mean_absolute_error(self.y_test, self.predictions)),
                'r2':   float(r2_score(self.y_test, self.predictions)),
            }
        except Exception:
            # fall back to classification
            return {
                'accuracy': float(accuracy_score(self.y_test, self.predictions)),
                'report':   classification_report(self.y_test, self.predictions),
            }