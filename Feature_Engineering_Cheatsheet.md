# Feature Engineering Cheatsheet

A one stop, practical reference based on the notebooks in the Feature engineering folder. Each section explains when and why to use a technique, then shows a tiny, fast to read code example.

## Quick principles

- Fit transforms on training data only to avoid leakage.
- Use pipelines to keep preprocessing and modeling consistent.
- Prefer robust methods (median, RobustScaler) when outliers exist.
- Keep the feature meaning: transform only if it improves signal or model fit.

---

## Missing values

### 1) Simple imputation (mean, median, most_frequent)
**Where/why:** Use for numeric or categorical columns when missingness is low or missing at random. Median is more robust to outliers.

```python
from sklearn.impute import SimpleImputer

num_imp = SimpleImputer(strategy="median")
X_num = num_imp.fit_transform(X_num)
```

### 2) Add missing indicator
**Where/why:** Use when missingness itself is informative. Often paired with SimpleImputer.

```python
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy="median", add_indicator=True)
X_num = imp.fit_transform(X_num)
```

### 3) Multivariate imputation (IterativeImputer)
**Where/why:** Use when features are correlated and missingness is higher; models each feature as a function of others.

```python
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

imp = IterativeImputer(random_state=42)
X_num = imp.fit_transform(X_num)
```

### 4) Grid search imputer params
**Where/why:** Use to compare strategies (mean vs median) or models for IterativeImputer in a pipeline.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

pipe = Pipeline([("imp", SimpleImputer()), ("model", Ridge())])
param_grid = {"imp__strategy": ["mean", "median"]}
GridSearchCV(pipe, param_grid, cv=5).fit(X_train, y_train)
```

---

## Scaling and normalization

### 1) Standardization (z-score)
**Where/why:** Use for models sensitive to feature scale (linear, SVM, kNN). Centers and scales to unit variance.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)
```

### 2) MinMax scaling
**Where/why:** Use when features must be in a fixed range, or for distance based models with bounded inputs.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_num)
```

### 3) Robust scaling
**Where/why:** Use when outliers exist; scales using median and IQR.

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_num)
```

### 4) Normalization (unit norm)
**Where/why:** Use for text or sparse data, or when only direction matters (cosine similarity).

```python
from sklearn.preprocessing import Normalizer

norm = Normalizer()
X_norm = norm.fit_transform(X_num)
```

---

## Encoding categorical variables

### 1) One-hot encoding
**Where/why:** Use for nominal categories with no intrinsic order. Avoid for very high cardinality.

```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown="ignore")
X_cat = enc.fit_transform(X_cat)
```

### 2) Ordinal encoding
**Where/why:** Use when categories have a natural order (low < medium < high).

```python
from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()
X_cat = enc.fit_transform(X_cat)
```

### 3) Frequency encoding (manual)
**Where/why:** Use for high cardinality categories; encodes by category frequency.

```python
freq = X["city"].value_counts(normalize=True)
X["city_freq"] = X["city"].map(freq)
```

---

## ColumnTransformer (mixed types)

**Where/why:** Use to apply different preprocessing to numeric and categorical columns in a single, leak safe pipeline.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

num_pipe = [
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
]
cat_pipe = [
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh", OneHotEncoder(handle_unknown="ignore")),
]

pre = ColumnTransformer([
    ("num", Pipeline(num_pipe), num_cols),
    ("cat", Pipeline(cat_pipe), cat_cols),
])
```

---

## Pipelines

### 1) Manual pipeline (step by step)
**Where/why:** Use when learning or debugging, or when you need custom branching logic.

```python
X_num = SimpleImputer(strategy="median").fit_transform(X_num)
X_num = StandardScaler().fit_transform(X_num)
model.fit(X_num, y)
```

### 2) scikit-learn Pipeline
**Where/why:** Use to prevent leakage and to make tuning and deployment easy.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

pipe = Pipeline([
    ("pre", pre),
    ("model", Ridge()),
])
pipe.fit(X_train, y_train)
```

---

## FunctionTransformer (custom logic)

**Where/why:** Use to wrap small custom transforms so they fit into pipelines.

```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer

log_tf = FunctionTransformer(np.log1p, validate=False)
X_log = log_tf.fit_transform(X_num)
```

---

## Power transforms and distribution fixes

### 1) Log transform
**Where/why:** Use for right skewed positive features to reduce skew and stabilize variance.

```python
import numpy as np

X["income_log"] = np.log1p(X["income"])
```

### 2) Yeo-Johnson / Box-Cox
**Where/why:** Use for skewed features. Box-Cox requires strictly positive data; Yeo-Johnson works with zero or negative.

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method="yeo-johnson")
X_t = pt.fit_transform(X_num)
```

### 3) Quantile transform
**Where/why:** Use when you want a Gaussian-like distribution, often for tree + linear combos.

```python
from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution="normal", random_state=42)
X_t = qt.fit_transform(X_num)
```

---

## Binning

### 1) Equal-width / equal-frequency bins
**Where/why:** Use to capture non linear effects or to reduce noise in continuous features.

```python
import pandas as pd

X["age_bin"] = pd.cut(X["age"], bins=5)
X["age_qbin"] = pd.qcut(X["age"], q=5, duplicates="drop")
```

### 2) KBinsDiscretizer
**Where/why:** Use in a pipeline and to choose strategy: uniform, quantile, or kmeans.

```python
from sklearn.preprocessing import KBinsDiscretizer

kb = KBinsDiscretizer(n_bins=5, encode="onehot", strategy="quantile")
X_binned = kb.fit_transform(X_num)
```

---

## Datetime features

**Where/why:** Use when raw timestamps hide seasonal or cyclic patterns.

```python
dt = pd.to_datetime(X["date"])
X["year"] = dt.dt.year
X["month"] = dt.dt.month
X["dayofweek"] = dt.dt.dayofweek
```

### Cyclical encoding
**Where/why:** Use for periodic features like month or hour to keep proximity (Dec close to Jan).

```python
import numpy as np

X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)
```

---

## Outlier handling

### 1) IQR rule (detect + clip)
**Where/why:** Use for univariate outliers; quick and interpretable.

```python
q1, q3 = X["price"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
X["price"] = X["price"].clip(low, high)
```

### 2) Z-score (detect)
**Where/why:** Use when data is roughly normal; easy to explain.

```python
import numpy as np

z = (X["price"] - X["price"].mean()) / X["price"].std()
outliers = X[np.abs(z) > 3]
```

### 3) Robust models or transforms
**Where/why:** Use when you prefer to keep all data but reduce impact of extreme values.

```python
from sklearn.preprocessing import RobustScaler

X_scaled = RobustScaler().fit_transform(X_num)
```

---

## Univariate analysis (before engineering)

**Where/why:** Use to understand skew, outliers, and missingness before selecting transformations.

```python
X["price"].describe()
X["price"].hist(bins=30)
```

---

## End to end example (mixed types + model)

**Where/why:** Use as a template for real projects and competitions.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge

num_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("oh", OneHotEncoder(handle_unknown="ignore")),
])

pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

model = Pipeline([
    ("pre", pre),
    ("ridge", Ridge()),
])

model.fit(X_train, y_train)
```

---

## Quick checklist

- Choose imputation strategy based on missingness pattern.
- Scale numeric features for distance or regularized models.
- Encode categories based on order and cardinality.
- Transform skewed features if it helps the model.
- Use pipelines and ColumnTransformer to avoid leakage.
