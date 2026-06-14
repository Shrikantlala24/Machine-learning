import io
import warnings
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression, f_regression, RFE
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RANDOM_STATE = 42
st.set_page_config(page_title="Interactive House Price Prediction Explorer", layout="wide")


def init_state():
    defaults = {
        "raw_df": None,
        "current_df": None,
        "pipeline_steps": [],
        "train_results": None,
        "processed_ready": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def add_step(step: str):
    if step not in st.session_state.pipeline_steps:
        st.session_state.pipeline_steps.append(step)


def reset_with_df(df: pd.DataFrame):
    st.session_state.raw_df = df.copy()
    st.session_state.current_df = df.copy()
    st.session_state.pipeline_steps = ["Dataset uploaded"]
    st.session_state.train_results = None
    st.session_state.processed_ready = None


def num_cols(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def cat_cols(df):
    return df.select_dtypes(exclude=np.number).columns.tolist()


def descriptive_stats(df: pd.DataFrame):
    numeric = df.select_dtypes(include=np.number)
    if numeric.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "mean": numeric.mean(),
        "median": numeric.median(),
        "std": numeric.std(),
        "min": numeric.min(),
        "max": numeric.max(),
    })
    return out.round(4)


def adjusted_r2(r2, n, p):
    if n <= p + 1:
        return np.nan
    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))


def fig_to_streamlit(fig):
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def transform_series(s: pd.Series, method: str):
    clean = s.copy()
    if method == "Log Transformation":
        if (clean <= 0).any():
            raise ValueError("Log transformation requires strictly positive values.")
        return np.log(clean)
    if method == "Square Root Transformation":
        if (clean < 0).any():
            raise ValueError("Square root transformation requires non-negative values.")
        return np.sqrt(clean)
    if method == "Box-Cox Transformation":
        if (clean <= 0).any():
            raise ValueError("Box-Cox transformation requires strictly positive values.")
        arr, _ = stats.boxcox(clean)
        return pd.Series(arr, index=s.index)
    if method == "Yeo-Johnson Transformation":
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        arr = pt.fit_transform(clean.to_frame())[:, 0]
        return pd.Series(arr, index=s.index)
    return clean


def encode_target_mean(df: pd.DataFrame, col: str, target: str):
    means = df.groupby(col)[target].mean()
    encoded = df[col].map(means)
    mapping = means.reset_index().rename(columns={target: "target_mean"})
    return encoded, mapping


def detect_outliers(series: pd.Series, method: str, z_thr=3.0, p_low=5, p_high=95):
    s = series.dropna()
    idx = pd.Index([])
    if s.empty:
        return idx, None
    if method == "IQR Method":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        idx = s[(s < low) | (s > high)].index
        return idx, (low, high)
    if method == "Z-Score Method":
        z = np.abs(stats.zscore(s, nan_policy="omit"))
        idx = s[z > z_thr].index
        return idx, None
    if method == "Modified Z-Score":
        med = np.median(s)
        mad = np.median(np.abs(s - med))
        if mad == 0:
            return pd.Index([]), None
        mz = 0.6745 * (s - med) / mad
        idx = s[np.abs(mz) > z_thr].index
        return idx, None
    if method == "Isolation Forest":
        iso = IsolationForest(contamination="auto", random_state=RANDOM_STATE)
        pred = iso.fit_predict(s.to_frame())
        idx = s[pred == -1].index
        return idx, None
    if method == "Capping/Clipping":
        low, high = np.percentile(s, [p_low, p_high])
        idx = s[(s < low) | (s > high)].index
        return idx, (low, high)
    return idx, None


def make_csv_bytes(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")


init_state()
st.title("Interactive House Price Prediction with Linear Regression")
st.caption("Upload a CSV, preprocess features step-by-step, train a linear regression model, and inspect metrics and assumptions.")

with st.sidebar:
    st.header("Navigation")
    step = st.radio(
        "Choose a step",
        [
            "Step 1: Upload Dataset",
            "Step 2: Feature Scaling",
            "Step 3: Encoding",
            "Step 4: Feature Transformation",
            "Step 5: Missing Value Handling",
            "Step 6: Outlier Handling",
            "Step 7: Feature Selection",
            "Step 8: Train & Evaluate",
        ],
    )
    if st.session_state.current_df is not None:
        st.markdown("---")
        st.write("**Applied steps**")
        for i, s in enumerate(st.session_state.pipeline_steps, start=1):
            st.write(f"{i}. {s}")
        if st.button("Reset to uploaded dataset"):
            reset_with_df(st.session_state.raw_df)
            st.rerun()

if step == "Step 1: Upload Dataset":
    uploaded = st.file_uploader("Upload house price CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            reset_with_df(df)
            st.success("Dataset uploaded successfully.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    if st.session_state.current_df is not None:
        df = st.session_state.current_df
        with st.expander("Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total records", df.shape[0])
        c2.metric("Total features", df.shape[1] - 1 if df.shape[1] > 0 else 0)
        target_guess = "price" if "price" in [c.lower() for c in df.columns] else "Not selected yet"
        c3.metric("Target variable", target_guess)
        if df.shape[0] < 10:
            st.warning("Dataset has fewer than 10 rows; evaluation may be unstable.")
        if df.shape[0] > 0 and df.shape[1] - 1 > 0 and (df.shape[1] - 1) >= df.shape[0]:
            st.warning("Features are not much smaller than samples; risk of overfitting or sparse learning.")

        tabs = st.tabs(["Shape & Types", "Missing Values", "Descriptive Stats"])
        with tabs[0]:
            st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]), use_container_width=True)
        with tabs[1]:
            st.dataframe(df.isna().sum().rename("missing_count").to_frame(), use_container_width=True)
        with tabs[2]:
            st.dataframe(descriptive_stats(df), use_container_width=True)

elif st.session_state.current_df is None:
    st.info("Start with Step 1 and upload a CSV file.")

elif step == "Step 2: Feature Scaling":
    df = st.session_state.current_df.copy()
    st.subheader("Feature Scaling")
    st.write("Scaling puts numeric features on a comparable scale so large-magnitude variables do not dominate the model.")
    numeric_cols = [c for c in num_cols(df)]
    target = st.selectbox("Target variable", numeric_cols, key="scale_target")
    feature_cols = [c for c in numeric_cols if c != target]
    selected = st.multiselect("Numeric columns to scale", feature_cols, default=feature_cols)
    scaler_name = st.selectbox("Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler"])

    if st.button("Apply scaling"):
        if not selected:
            st.warning("Select at least one numeric feature.")
        else:
            scaler = {
                "StandardScaler": StandardScaler(),
                "MinMaxScaler": MinMaxScaler(),
                "RobustScaler": RobustScaler(),
            }[scaler_name]
            before = df[selected].agg(["mean", "min", "max"]).T.round(4)
            valid = df[selected].dropna()
            if valid.empty:
                st.warning("Selected columns do not have enough non-missing data to scale.")
            else:
                scaled = scaler.fit_transform(df[selected])
                df[selected] = scaled
                st.session_state.current_df = df
                add_step(f"Feature scaling: {scaler_name} on {', '.join(selected)}")
                after = df[selected].agg(["mean", "min", "max"]).T.round(4)
                st.write("Before scaling")
                st.dataframe(before, use_container_width=True)
                st.write("After scaling")
                st.dataframe(after, use_container_width=True)
                st.success("Scaling applied.")

    st.dataframe(st.session_state.current_df.head(10), use_container_width=True)

elif step == "Step 3: Encoding":
    df = st.session_state.current_df.copy()
    st.subheader("Encoding")
    st.write("Encoding converts categorical values into numeric form so linear regression can use them.")
    target_options = num_cols(df)
    if not target_options:
        st.error("No numeric target candidate found. Ensure the price column is numeric.")
    else:
        target = st.selectbox("Target variable", target_options, key="enc_target")
        categorical = cat_cols(df)
        cols = st.multiselect("Categorical columns to encode", categorical)
        method = st.radio("Encoding method", ["Label Encoding", "One-Hot Encoding", "Target Encoding"])
        if st.button("Apply encoding"):
            if not cols:
                st.warning("Choose at least one categorical column.")
            else:
                mapping_views = []
                if method == "Label Encoding":
                    for col in cols:
                        categories = {v: i for i, v in enumerate(df[col].astype(str).fillna("Missing").unique())}
                        df[col] = df[col].astype(str).fillna("Missing").map(categories)
                        mapping_views.append(pd.DataFrame({col: list(categories.keys()), "encoded": list(categories.values())}))
                elif method == "One-Hot Encoding":
                    uniq_before = df[cols].nunique(dropna=False).rename("unique_values")
                    df = pd.get_dummies(df, columns=cols, dummy_na=True, drop_first=False)
                    mapping_views.append(uniq_before.to_frame())
                else:
                    for col in cols:
                        enc, mapping = encode_target_mean(df, col, target)
                        df[col] = enc
                        mapping_views.append(mapping.rename(columns={col: "category"}))
                st.session_state.current_df = df
                add_step(f"Encoding: {method} on {', '.join(cols)}")
                st.success("Encoding applied.")
                for i, view in enumerate(mapping_views, start=1):
                    st.dataframe(view, use_container_width=True)
                enc_cols = [c for c in df.columns if any(base in c for base in cols)]
                st.write("Encoded columns overview")
                st.dataframe(df[enc_cols].head(10), use_container_width=True)

elif step == "Step 4: Feature Transformation":
    df = st.session_state.current_df.copy()
    st.subheader("Feature Transformation")
    st.write("Transformations can reduce skewness and better align data with linear regression assumptions like homoscedasticity.")
    numeric = num_cols(df)
    target = st.selectbox("Target variable", numeric, key="tr_target")
    candidates = [c for c in numeric if c != target]
    cols = st.multiselect("Numeric columns to transform", candidates)
    method = st.selectbox("Transformation", ["Log Transformation", "Square Root Transformation", "Box-Cox Transformation", "Yeo-Johnson Transformation"])
    if cols:
        selected_plot = st.selectbox("Inspect a column", cols)
        non_null = df[selected_plot].dropna()
        if not non_null.empty:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            sns.histplot(non_null, kde=True, ax=axes[0], color="#4C78A8")
            axes[0].set_title(f"Before: skew={non_null.skew():.3f}")
            try:
                transformed = transform_series(non_null, method)
                sns.histplot(transformed, kde=True, ax=axes[1], color="#F58518")
                axes[1].set_title(f"After: skew={pd.Series(transformed).skew():.3f}")
                fig_to_streamlit(fig)
            except Exception as e:
                axes[1].text(0.05, 0.5, str(e), wrap=True)
                axes[1].set_axis_off()
                fig_to_streamlit(fig)
    if st.button("Apply transformation"):
        reports = []
        for col in cols:
            non_null = df[col].dropna()
            if non_null.empty:
                continue
            try:
                before_skew = non_null.skew()
                transformed = transform_series(non_null, method)
                df.loc[non_null.index, col] = transformed.values
                after_skew = pd.Series(transformed).skew()
                reports.append({"column": col, "before_skew": before_skew, "after_skew": after_skew})
            except Exception as e:
                st.warning(f"{col}: {e}")
        st.session_state.current_df = df
        if reports:
            add_step(f"Transformation: {method} on {', '.join([r['column'] for r in reports])}")
            st.dataframe(pd.DataFrame(reports).round(4), use_container_width=True)
            st.success("Transformations applied where valid.")
        else:
            st.info("No transformation was applied.")

elif step == "Step 5: Missing Value Handling":
    df = st.session_state.current_df.copy()
    st.subheader("Missing Value Handling")
    st.write("Choose how to fill or remove missing values. In production, fitting imputation before the train-test split can cause leakage, so this app is for learning and exploration.")
    missing_cols = [c for c in df.columns if df[c].isna().sum() > 0]
    if not missing_cols:
        st.success("No missing values found in the current dataset.")
    else:
        st.dataframe(df[missing_cols].isna().sum().rename("missing_count").to_frame(), use_container_width=True)
        strategies: Dict[str, str] = {}
        for col in missing_cols:
            options = ["Deletion", "Mean Imputation", "Median Imputation", "Forward Fill (FFill)", "Backward Fill (BFill)", "KNN Imputation", "MICE"]
            if df[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                options = ["Deletion", "Forward Fill (FFill)", "Backward Fill (BFill)"]
            strategies[col] = st.selectbox(f"{col} strategy", options, key=f"miss_{col}")
        knn_k = st.slider("KNN k-value", 1, 15, 5)
        mice_iter = st.slider("MICE iterations", 1, 20, 10)
        if st.button("Apply missing value handling"):
            report_rows = []
            to_drop_rows = set()
            to_drop_cols = []
            for col, strat in strategies.items():
                before_idx = df[df[col].isna()].index.tolist()
                if strat == "Deletion":
                    to_drop_rows.update(before_idx)
                elif strat == "Mean Imputation":
                    fill = df[col].mean()
                    df[col] = df[col].fillna(fill)
                    report_rows.extend([{"column": col, "row_index": idx, "imputed_value": fill} for idx in before_idx])
                elif strat == "Median Imputation":
                    fill = df[col].median()
                    df[col] = df[col].fillna(fill)
                    report_rows.extend([{"column": col, "row_index": idx, "imputed_value": fill} for idx in before_idx])
                elif strat == "Forward Fill (FFill)":
                    df[col] = df[col].ffill()
                elif strat == "Backward Fill (BFill)":
                    df[col] = df[col].bfill()
            numeric_missing = [c for c in df.columns if df[c].isna().sum() > 0 and pd.api.types.is_numeric_dtype(df[c])]
            if any(v == "KNN Imputation" for v in strategies.values()) and numeric_missing:
                knn_cols = [c for c, v in strategies.items() if v == "KNN Imputation" and pd.api.types.is_numeric_dtype(df[c])]
                imputer = KNNImputer(n_neighbors=knn_k)
                df[knn_cols] = imputer.fit_transform(df[knn_cols])
            if any(v == "MICE" for v in strategies.values()) and numeric_missing:
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer
                mice_cols = [c for c, v in strategies.items() if v == "MICE" and pd.api.types.is_numeric_dtype(df[c])]
                imputer = IterativeImputer(random_state=RANDOM_STATE, max_iter=mice_iter)
                df[mice_cols] = imputer.fit_transform(df[mice_cols])
            if to_drop_rows:
                df = df.drop(index=list(to_drop_rows)).reset_index(drop=True)
            if to_drop_cols:
                df = df.drop(columns=to_drop_cols)
            st.session_state.current_df = df
            add_step("Missing value handling applied")
            if report_rows:
                st.dataframe(pd.DataFrame(report_rows).head(100), use_container_width=True)
            st.write("Remaining missing counts")
            st.dataframe(df.isna().sum().rename("missing_count").to_frame(), use_container_width=True)
            st.success("Missing value handling applied.")

elif step == "Step 6: Outlier Handling":
    df = st.session_state.current_df.copy()
    st.subheader("Outlier Handling")
    numeric = num_cols(df)
    if not numeric:
        st.error("No numeric columns available.")
    else:
        cols = st.multiselect("Numeric columns to inspect", numeric, default=numeric[: min(3, len(numeric))])
        method = st.radio("Detection method", ["IQR Method", "Z-Score Method", "Modified Z-Score", "Isolation Forest", "Capping/Clipping"])
        z_thr = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.1)
        p_low, p_high = st.slider("Percentile cap range", 0, 100, (5, 95))
        action = st.radio("Action", ["remove", "cap"])
        preview_col = st.selectbox("Preview column", cols if cols else numeric)
        out_idx, bounds = detect_outliers(df[preview_col], method, z_thr, p_low, p_high)
        st.write(f"Detected outliers in {preview_col}: {len(out_idx)}")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.boxplot(y=df[preview_col], ax=axes[0], color="#72B7B2")
        axes[0].set_title("Before")
        temp = df[preview_col].copy()
        if action == "cap" and len(out_idx) > 0:
            if bounds is None:
                low, high = np.percentile(df[preview_col].dropna(), [p_low, p_high])
            else:
                low, high = bounds
            temp.loc[out_idx] = temp.loc[out_idx].clip(low, high)
        elif action == "remove":
            temp = temp.drop(index=out_idx)
        sns.boxplot(y=temp, ax=axes[1], color="#E45756")
        axes[1].set_title("After")
        fig_to_streamlit(fig)
        if st.button("Apply outlier handling"):
            removed_all = set()
            count_rows = []
            for col in cols:
                idx, b = detect_outliers(df[col], method, z_thr, p_low, p_high)
                count_rows.append({"column": col, "detected_outliers": len(idx)})
                if action == "remove":
                    removed_all.update(idx.tolist())
                else:
                    if len(idx) > 0:
                        if b is None:
                            low, high = np.percentile(df[col].dropna(), [p_low, p_high])
                        else:
                            low, high = b
                        df.loc[idx, col] = df.loc[idx, col].clip(low, high)
            if action == "remove" and removed_all:
                df = df.drop(index=list(removed_all)).reset_index(drop=True)
            st.session_state.current_df = df
            add_step(f"Outlier handling: {method} with action={action}")
            st.dataframe(pd.DataFrame(count_rows), use_container_width=True)
            st.success("Outlier handling applied.")

elif step == "Step 7: Feature Selection":
    df = st.session_state.current_df.copy()
    st.subheader("Feature Selection")
    numeric = num_cols(df)
    if len(numeric) < 2:
        st.error("Need at least one numeric target and one numeric feature.")
    else:
        target = st.selectbox("Target variable", numeric, key="fs_target")
        X = df.drop(columns=[target])
        y = df[target]
        if X.isna().any().any() or y.isna().any():
            st.warning("Resolve missing values before feature selection.")
        else:
            methods = st.multiselect(
                "Methods",
                ["Correlation-based", "Variance Threshold", "Mutual Information", "SelectKBest", "RFE", "Permutation Importance"],
            )
            k = st.slider("Number of features to keep (K)", 1, max(1, X.shape[1]), min(5, max(1, X.shape[1])))
            corr_thr = st.slider("Correlation threshold", 0.5, 0.99, 0.9, 0.01)
            var_thr = st.slider("Variance threshold", 0.0, 1.0, 0.0, 0.01)
            if st.button("Apply feature selection"):
                Xw = pd.get_dummies(X, drop_first=False)
                reports = []
                selected_cols = list(Xw.columns)
                if "Correlation-based" in methods:
                    corr = Xw.corr(numeric_only=True).abs()
                    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                    drop_high = [column for column in upper.columns if any(upper[column] > corr_thr)]
                    target_corr = Xw.apply(lambda s: s.corr(y)).abs().fillna(0)
                    low_target = target_corr[target_corr < 0.01].index.tolist()
                    selected_cols = [c for c in selected_cols if c not in set(drop_high + low_target)]
                    reports.append(pd.DataFrame({"feature": Xw.columns, "corr_with_target": target_corr.reindex(Xw.columns).values}).sort_values("corr_with_target", ascending=False))
                    if (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)) > 0.95).sum().sum() > 0:
                        st.warning("Strong multicollinearity detected: some feature correlations exceed 0.95.")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.heatmap(corr.iloc[: min(20, len(corr)), : min(20, len(corr))], cmap="coolwarm", ax=ax)
                    ax.set_title("Correlation heatmap (first 20 features)")
                    fig_to_streamlit(fig)
                if "Variance Threshold" in methods and selected_cols:
                    vt = VarianceThreshold(threshold=var_thr)
                    vt.fit(Xw[selected_cols])
                    selected_cols = [c for c, keep in zip(selected_cols, vt.get_support()) if keep]
                    reports.append(pd.DataFrame({"feature": Xw.columns, "variance": Xw.var().values}).sort_values("variance", ascending=False))
                if "Mutual Information" in methods and selected_cols:
                    mi = mutual_info_regression(Xw[selected_cols], y, random_state=RANDOM_STATE)
                    mi_df = pd.DataFrame({"feature": selected_cols, "importance": mi}).sort_values("importance", ascending=False)
                    selected_cols = mi_df.head(min(k, len(mi_df)))["feature"].tolist()
                    reports.append(mi_df)
                if "SelectKBest" in methods and selected_cols:
                    skb = SelectKBest(score_func=f_regression, k=min(k, len(selected_cols)))
                    skb.fit(Xw[selected_cols], y)
                    scores = pd.DataFrame({"feature": selected_cols, "score": skb.scores_}).sort_values("score", ascending=False)
                    selected_cols = scores.head(min(k, len(scores)))["feature"].tolist()
                    reports.append(scores)
                if "RFE" in methods and selected_cols:
                    est = LinearRegression()
                    rfe = RFE(estimator=est, n_features_to_select=min(k, len(selected_cols)))
                    rfe.fit(Xw[selected_cols], y)
                    rank_df = pd.DataFrame({"feature": selected_cols, "rank": rfe.ranking_}).sort_values("rank")
                    selected_cols = rank_df.head(min(k, len(rank_df)))["feature"].tolist()
                    reports.append(rank_df)
                if "Permutation Importance" in methods and selected_cols:
                    est = LinearRegression().fit(Xw[selected_cols], y)
                    perm = permutation_importance(est, Xw[selected_cols], y, random_state=RANDOM_STATE)
                    perm_df = pd.DataFrame({"feature": selected_cols, "importance": perm.importances_mean}).sort_values("importance", ascending=False)
                    selected_cols = perm_df.head(min(k, len(perm_df)))["feature"].tolist()
                    reports.append(perm_df)
                final_df = pd.concat([Xw[selected_cols], y], axis=1)
                st.session_state.current_df = final_df
                add_step(f"Feature selection applied; kept {len(selected_cols)} features")
                st.write("Final selected features")
                st.write(selected_cols)
                for rep in reports:
                    st.dataframe(rep.head(20), use_container_width=True)
                st.success("Feature selection applied.")

elif step == "Step 8: Train & Evaluate":
    df = st.session_state.current_df.copy()
    st.subheader("Train & Evaluate")
    numeric = num_cols(df)
    if not numeric:
        st.error("No numeric columns available for training.")
    else:
        target = st.selectbox("Target variable", numeric, key="train_target")
        if not pd.api.types.is_numeric_dtype(df[target]):
            st.error("Target must be numeric.")
        else:
            X = df.drop(columns=[target])
            y = df[target]
            X = pd.get_dummies(X, drop_first=False)
            if X.empty:
                st.error("No features available after preprocessing.")
            elif X.isna().any().any() or y.isna().any():
                st.error("Resolve missing values before training.")
            else:
                test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
                rs = st.number_input("Random state", value=RANDOM_STATE, step=1)
                if X.shape[1] >= X.shape[0]:
                    st.warning("Number of features is close to or exceeds number of samples; model may overfit.")
                corr = X.corr().abs()
                if (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)) > 0.95).sum().sum() > 0:
                    st.warning("Strong multicollinearity detected among input features.")
                if st.button("Train linear regression"):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(rs))
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    pred_train = model.predict(X_train)
                    pred_test = model.predict(X_test)
                    train_r2 = r2_score(y_train, pred_train)
                    test_r2 = r2_score(y_test, pred_test)
                    metrics = pd.DataFrame([
                        {
                            "split": "Train",
                            "R2": train_r2,
                            "Adjusted R2": adjusted_r2(train_r2, len(y_train), X_train.shape[1]),
                            "MAE": mean_absolute_error(y_train, pred_train),
                            "MSE": mean_squared_error(y_train, pred_train),
                            "RMSE": np.sqrt(mean_squared_error(y_train, pred_train)),
                        },
                        {
                            "split": "Test",
                            "R2": test_r2,
                            "Adjusted R2": adjusted_r2(test_r2, len(y_test), X_test.shape[1]),
                            "MAE": mean_absolute_error(y_test, pred_test),
                            "MSE": mean_squared_error(y_test, pred_test),
                            "RMSE": np.sqrt(mean_squared_error(y_test, pred_test)),
                        },
                    ]).round(4)
                    coeffs = pd.DataFrame({"feature": X.columns, "coefficient": model.coef_}).sort_values("coefficient", key=np.abs, ascending=False)
                    st.session_state.train_results = {
                        "metrics": metrics,
                        "coeffs": coeffs,
                        "processed_df": pd.concat([X, y], axis=1),
                    }
                    add_step("Linear regression training and evaluation")
                    c1, c2 = st.columns(2)
                    c1.metric("Train size", len(X_train))
                    c2.metric("Test size", len(X_test))
                    st.dataframe(metrics, use_container_width=True)
                    st.write(f"Intercept: {model.intercept_:.4f}")
                    st.dataframe(coeffs, use_container_width=True)

                    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
                    axes[0].scatter(y_test, pred_test, alpha=0.7)
                    axes[0].set_xlabel("Actual")
                    axes[0].set_ylabel("Predicted")
                    axes[0].set_title("Actual vs Predicted")
                    residuals = y_test - pred_test
                    axes[1].scatter(pred_test, residuals, alpha=0.7)
                    axes[1].axhline(0, color="red", linestyle="--")
                    axes[1].set_xlabel("Predicted")
                    axes[1].set_ylabel("Residual")
                    axes[1].set_title("Residual Plot")
                    sns.histplot(residuals, kde=True, ax=axes[2], color="#54A24B")
                    axes[2].set_title("Residual Distribution")
                    fig.tight_layout()
                    fig_to_streamlit(fig)

                    st.markdown("### Final Summary")
                    st.write("Pipeline summary")
                    for i, s in enumerate(st.session_state.pipeline_steps, start=1):
                        st.write(f"{i}. {s}")
                    st.write("Final metrics")
                    st.dataframe(metrics, use_container_width=True)

                    processed_bytes = make_csv_bytes(pd.concat([X, y], axis=1))
                    coeff_bytes = coeffs.to_csv(index=False).encode("utf-8")
                    st.download_button("Download processed dataset", processed_bytes, file_name="processed_dataset.csv", mime="text/csv")
                    st.download_button("Download model coefficients", coeff_bytes, file_name="model_coefficients.csv", mime="text/csv")
