from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from clean_data import load_and_clean
from train_pipeline import build_pipeline

if __name__ == "__main__":
    # 1) Load & clean
    df = load_and_clean()

    # 2) Split features/target
    X = df.drop("price", axis=1)
    y = df["price"]

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) Identify column groups
    numeric_feats = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # Example categoricals: adjust if you added more in load_and_clean()
    categorical_feats = ["zipcode", "view", "condition", "waterfront", "grade", "sale_year", "sale_month"]

    # 5) Build & train pipeline
    pipe = build_pipeline(numeric_feats, categorical_feats)
    pipe.fit(X_train, y_train)

    # 6) Evaluate
    preds = pipe.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", root_mean_squared_error(y_test, preds))
    print("R²:", r2_score(y_test, preds))