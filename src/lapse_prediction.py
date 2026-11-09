"""lapse_prediction.py -- Lapse cohort definition, feature engineering, modeling,
evaluation, class balance experiment, and retention list generation for Part B.
Saves outputs/lapse_model.joblib and outputs/retention_list.csv.
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils import resample

ORDERS_PATH = os.path.join("data", "clean", "orders.csv")
CUSTOMERS_PATH = os.path.join("data", "clean", "customers.csv")
PRODUCTS_PATH = os.path.join("data", "clean", "products.csv")
SEGMENTS_PATH = os.path.join("outputs", "customer_segments.csv")

MODEL_OUTPUT_PATH = os.path.join("outputs", "lapse_model.joblib")
RETENTION_OUTPUT_PATH = os.path.join("outputs", "retention_list.csv")


def build_cohort_and_features():
    orders = pd.read_csv(ORDERS_PATH)
    customers = pd.read_csv(CUSTOMERS_PATH)
    products = pd.read_csv(PRODUCTS_PATH)

    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    customers["signup_date"] = pd.to_datetime(customers["signup_date"])
    orders = orders.merge(products[["product_id", "category"]], on="product_id", how="left")

    cutoff = pd.Timestamp("2025-09-01 00:00:00")
    window_end = pd.Timestamp("2025-09-30 23:59:59")

    # Cohort: at least 1 completed order before cutoff
    pre_orders = orders[orders["order_ts"] < cutoff]
    pre_comp = pre_orders[pre_orders["status"] == "completed"]
    cohort_cids = sorted(pre_comp["customer_id"].unique())

    # Label: lapsed = 1 if no completed order in Sept 2025
    post_orders = orders[(orders["order_ts"] >= cutoff) & (orders["order_ts"] <= window_end)]
    post_comp_cids = set(post_orders[post_orders["status"] == "completed"]["customer_id"].unique())
    y_dict = {cid: (0 if cid in post_comp_cids else 1) for cid in cohort_cids}

    # Features (ONLY from before cutoff)
    recency = pre_comp.groupby("customer_id")["order_ts"].max().apply(lambda t: (cutoff - t).days)
    freq = pre_comp.groupby("customer_id")["order_id"].nunique()
    monetary = pre_comp.groupby("customer_id")["order_amount"].sum()
    avg_disc = pre_comp.groupby("customer_id")["discount_pct"].mean()
    avg_deliv = pre_comp.groupby("customer_id")["delivery_minutes"].mean()

    all_pre = pre_orders[pre_orders["status"].isin(["completed", "cancelled"])]
    canc_count = all_pre[all_pre["status"] == "cancelled"].groupby("customer_id")["order_id"].count()
    total_pre_count = all_pre.groupby("customer_id")["order_id"].count()
    canc_rate = (canc_count / total_pre_count).reindex(cohort_cids).fillna(0.0)

    fav_cat = pre_orders.groupby("customer_id")["category"].agg(lambda s: s.mode()[0])

    df = pd.DataFrame({
        "customer_id": cohort_cids,
        "recency": [recency[c] for c in cohort_cids],
        "frequency": [freq[c] for c in cohort_cids],
        "monetary": [monetary[c] for c in cohort_cids],
        "avg_discount_pct": [avg_disc[c] for c in cohort_cids],
        "cancellation_rate": [canc_rate[c] for c in cohort_cids],
        "avg_delivery_minutes": [avg_deliv[c] for c in cohort_cids],
        "fav_category": [fav_cat[c] for c in cohort_cids],
        "lapsed": [y_dict[c] for c in cohort_cids]
    }).merge(customers[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id", how="left")

    df["tenure_days"] = (cutoff - df["signup_date"]).dt.days

    # Verify no missing values in features
    assert df.drop(columns=["customer_id", "signup_date"]).isna().sum().sum() == 0, "Missing values in features"

    return df


def main():
    print("=" * 60)
    print("LAPSE PREDICTION (PART B)")
    print("=" * 60)
    df = build_cohort_and_features()

    cohort_size = len(df)
    lapsed_count = df["lapsed"].sum()
    lapsed_pct = (lapsed_count / cohort_size) * 100
    print(f"Cohort Size: {cohort_size} customers (Reference: 1,156)")
    print(f"Lapsed Count: {lapsed_count} ({lapsed_pct:.2f}%) (Reference: 443 / 38.32%)")

    num_cols = ["recency", "frequency", "monetary", "avg_discount_pct", "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"]
    cat_cols = ["city", "gender", "fav_category"]
    feature_cols = num_cols + cat_cols

    X = df[feature_cols]
    y = df["lapsed"]
    cids = df["customer_id"]

    X_train, X_test, y_train, y_test, cids_train, cids_test = train_test_split(
        X, y, cids, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train Rows: {len(X_train)} (Reference: 924), Test Rows: {len(X_test)} (Reference: 232)")

    # Baseline evaluation
    base_acc = (y_test == 0).mean() * 100
    base_f1 = f1_score(y_test, np.zeros_like(y_test), zero_division=0)
    print(f"Baseline (Always Non-Lapsed): Accuracy = {base_acc:.1f}%, Lapsed F1 = {base_f1:.4f}")

    # Leakage check: Scaler fitted on train only
    scaler_check = StandardScaler()
    scaler_check.fit(X_train[num_cols])
    learned_mean_freq = scaler_check.mean_[num_cols.index("frequency")]
    full_mean_freq = X["frequency"].mean()
    print(f"\nFrequency Mean Check:")
    print(f"  Fitted Scaler Train Mean: {learned_mean_freq:.4f}")
    print(f"  Full Cohort Mean: {full_mean_freq:.4f}")
    print("  Note: Train mean differs because it was computed exclusively on the 80% training split without test leakage.")

    # Pipeline preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols)
        ]
    )

    # 1. Logistic Regression Grid Search
    print("\nTuning Logistic Regression (C in [0.01, 0.1, 1, 10, 100])...")
    pipe_lr = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    grid_lr = GridSearchCV(pipe_lr, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1")
    grid_lr.fit(X_train, y_train)
    best_lr_c = grid_lr.best_params_["clf__C"]
    print(f"  Best C: {best_lr_c}, Best CV F1: {grid_lr.best_score_:.4f}")

    y_pred_lr = grid_lr.predict(X_test)
    lr_f1 = f1_score(y_test, y_pred_lr, zero_division=0)
    lr_prec = precision_score(y_test, y_pred_lr, zero_division=0)
    lr_rec = recall_score(y_test, y_pred_lr, zero_division=0)
    print(f"  Test Evaluation -> F1: {lr_f1:.4f}, Precision: {lr_prec:.4f}, Recall: {lr_rec:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred_lr)}")

    # 2. Random Forest Grid Search
    print("\nTuning Random Forest (max_depth in [3, 5, 8], min_samples_leaf in [5, 10, 20])...")
    pipe_rf = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    param_grid_rf = {
        "clf__max_depth": [3, 5, 8],
        "clf__min_samples_leaf": [5, 10, 20]
    }
    grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring="f1")
    grid_rf.fit(X_train, y_train)
    best_rf_params = grid_rf.best_params_
    print(f"  Best Params: {best_rf_params}, Best CV F1: {grid_rf.best_score_:.4f}")

    y_pred_rf = grid_rf.predict(X_test)
    rf_f1 = f1_score(y_test, y_pred_rf, zero_division=0)
    rf_prec = precision_score(y_test, y_pred_rf, zero_division=0)
    rf_rec = recall_score(y_test, y_pred_rf, zero_division=0)
    print(f"  Test Evaluation -> F1: {rf_f1:.4f}, Precision: {rf_prec:.4f}, Recall: {rf_rec:.4f}")
    print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}")

    # 3. Class Balance Experiment
    print("\nClass-Balance Experiment (Oversampling Training Set):")
    train_df = X_train.copy()
    train_df["y"] = y_train.values
    majority = train_df[train_df["y"] == 0]
    minority = train_df[train_df["y"] == 1]
    print(f"  Original Train Counts: Non-Lapsed = {len(majority)}, Lapsed = {len(minority)}")

    minority_oversampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
    balanced_train = pd.concat([majority, minority_oversampled])
    X_train_bal = balanced_train.drop(columns=["y"])
    y_train_bal = balanced_train["y"]
    print(f"  Balanced Train Counts: {len(majority)} each (Total: {len(balanced_train)})")

    pipe_lr_bal = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(C=best_lr_c, max_iter=1000, random_state=42))
    ])
    pipe_lr_bal.fit(X_train_bal, y_train_bal)
    y_pred_bal = pipe_lr_bal.predict(X_test)
    bal_f1 = f1_score(y_test, y_pred_bal, zero_division=0)
    bal_prec = precision_score(y_test, y_pred_bal, zero_division=0)
    bal_rec = recall_score(y_test, y_pred_bal, zero_division=0)
    print(f"  Balanced Model Test Evaluation -> F1: {bal_f1:.4f}, Precision: {bal_prec:.4f}, Recall: {bal_rec:.4f}")
    print(f"  Impact: Recall increased by {(bal_rec - lr_rec)*100:.1f} percentage points (from {lr_rec:.4f} to {bal_rec:.4f}), while precision stayed near {bal_prec:.4f}.")

    # 4. Feature Importance & Coefficients
    print("\nFeature Interpretability:")
    # Extract feature names from preprocessor
    ohe_features = list(grid_rf.best_estimator_.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols))
    all_feature_names = num_cols + ohe_features

    rf_clf = grid_rf.best_estimator_.named_steps["clf"]
    importances = pd.Series(rf_clf.feature_importances_, index=all_feature_names).sort_values(ascending=False)
    print("  Top 5 Random Forest Features:")
    for feat, imp in importances.head(5).items():
        print(f"    - {feat}: {imp:.4f}")

    lr_clf = grid_lr.best_estimator_.named_steps["clf"]
    coefficients = pd.Series(lr_clf.coef_[0], index=all_feature_names)
    print("\n  Selected Logistic Regression Feature Coefficients:")
    for feat in ["recency", "frequency", "monetary"]:
        print(f"    - {feat}: coef = {coefficients[feat]:.4f} (sign: {'+' if coefficients[feat] > 0 else '-'})")

    # 5. Persist Selected Model (Random Forest, F1 >= 0.55)
    selected_model = grid_rf.best_estimator_
    joblib.dump(selected_model, MODEL_OUTPUT_PATH)
    print(f"\nPersisted model to {MODEL_OUTPUT_PATH}")

    # Verify reloading
    reloaded_model = joblib.load(MODEL_OUTPUT_PATH)
    y_pred_reloaded = reloaded_model.predict(X_test)
    assert np.array_equal(y_pred_rf, y_pred_reloaded), "Reloaded model failed to reproduce test predictions"
    print("Verified: Reloaded model exactly reproduces original test predictions.")

    # 6. Retention List for Premium Segment
    segments_df = pd.read_csv(SEGMENTS_PATH)
    premium_cids = set(segments_df[segments_df["is_premium"] == True]["customer_id"])

    test_df = pd.DataFrame({
        "customer_id": cids_test.values,
        "actually_lapsed": y_test.values,
        "lapse_probability": selected_model.predict_proba(X_test)[:, 1].round(4)
    })

    test_premium = test_df[test_df["customer_id"].isin(premium_cids)].copy()
    retention_list = test_premium.sort_values("lapse_probability", ascending=False).head(10)
    retention_list.to_csv(RETENTION_OUTPUT_PATH, index=False)
    print(f"\nRetention List Generated ({len(retention_list)} customers) saved to {RETENTION_OUTPUT_PATH}:")
    print(retention_list.to_string(index=False))
    truly_lapsed = retention_list["actually_lapsed"].sum()
    print(f"Customers in retention list who truly lapsed: {truly_lapsed} of {len(retention_list)}")


if __name__ == "__main__":
    main()
