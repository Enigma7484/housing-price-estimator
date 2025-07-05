from clean_data import load_and_clean
from train_pipeline import build_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # --- 1. Load & clean ---
    df = load_and_clean()

    # --- 2. Features & target ---
    X = df.drop("price", axis=1)
    y = df["price"]

    # --- 3. Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- 4. Identify columns ---
    numeric_feats = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_feats = [
        "zipcode","view","condition","waterfront",
        "grade","sale_year","sale_month"
    ]

    # --- 5. Build & fit pipeline ---
    pipe = build_pipeline(numeric_feats, categorical_feats)
    pipe.fit(X_train, y_train)

    # --- 6. Predictions & metrics ---
    preds = pipe.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²  : {r2:.4f}")

    # --- 7. Residual plot ---
    residuals = y_test - preds
    plt.figure(figsize=(6,4))
    plt.scatter(preds, residuals, alpha=0.3)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.show()

    # --- 8. Feature importances ---
    # 8.1 Retrieve feature names
    num_names = numeric_feats
    ohe = pipe.named_steps["preprocessor"] \
             .named_transformers_["cat"] \
             .named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(categorical_feats)
    feature_names = np.concatenate([num_names, cat_names])

    importances = pipe.named_steps["model"].feature_importances_
    df_imp = pd.Series(importances, index=feature_names)

    # 8.2 Top 10 features
    top10 = df_imp.sort_values(ascending=False).head(10)
    print("\nTop 10 features by importance:")
    for name, imp in top10.items():
        print(f"{name}: {imp:.4f}")

    # 8.3 Show importances for bathrooms, grade_*, zipcode_*
    print("\nSelected importances for bathrooms, grade, and zipcode:")
    for name, imp in df_imp.items():
        if name == "bathrooms" or name.startswith("grade_") or name.startswith("zipcode_"):
            print(f"{name:20s} → {imp:.4f}")

    # 8.4 Aggregate total importance for grade & zipcode
    total_grade = df_imp[df_imp.index.str.startswith("grade_")].sum()
    total_zip   = df_imp[df_imp.index.str.startswith("zipcode_")].sum()
    print(f"\nTotal importance of grade  : {total_grade:.4f}")
    print(f"Total importance of zipcode: {total_zip:.4f}")