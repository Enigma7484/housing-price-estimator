from clean_data import load_and_clean
from train_pipeline import build_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import numpy as np

if __name__ == "__main__":
    df = load_and_clean()
    X = df.drop("price", axis=1)
    y = df["price"]

    # 1) Log-transform the target
    y_log = np.log1p(y)

    # 2) Split
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    # 3) Build & train
    numeric_feats = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_feats = ["zipcode","view","condition","waterfront",
                         "grade","sale_year","sale_month"]
    pipe = build_pipeline(numeric_feats, categorical_feats)
    pipe.fit(X_train, y_train_log)

    # 4) Predict & invert transform
    y_pred_log = pipe.predict(X_test)
    y_pred = np.expm1(y_pred_log)    # back to price-space
    y_true = np.expm1(y_test_log)

    # 5) Metrics on original scale
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print(f"Log-target MAE : {mae:,.2f}")
    print(f"Log-target RMSE: {rmse:,.2f}")
    print(f"Log-target R²  : {r2:.4f}")