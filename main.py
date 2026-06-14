import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

def evaluate_model(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "model": name,
        "R2": r2_score(y_test, pred),
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred))
    }

results = []

baseline = Pipeline([
    ("lr", LinearRegression())
])
results.append(evaluate_model("Baseline LR", baseline))

selectk = Pipeline([
    ("select", SelectKBest(score_func=f_regression, k=5)),
    ("lr", LinearRegression())
])
results.append(evaluate_model("LR + SelectKBest(k=5)", selectk))

rfe_model = Pipeline([
    ("rfe", RFE(estimator=LinearRegression(), n_features_to_select=5)),
    ("lr", LinearRegression())
])
results.append(evaluate_model("LR + RFE(5)", rfe_model))

pca_model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=5)),
    ("lr", LinearRegression())
])
results.append(evaluate_model("LR + PCA(5)", pca_model))

results_df = pd.DataFrame(results)
print(results_df)