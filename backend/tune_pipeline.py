import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

from clean_data import load_and_clean
from train_pipeline import build_pipeline

if __name__ == "__main__":
    # 1) Load & clean
    df = load_and_clean()

    # 2) Split features/target
    X = df.drop("price", axis=1)
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3) Column groups (same as before)
    numeric_feats = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_feats = X.select_dtypes(include=["object"]).columns.tolist()

    # 4) Build base pipeline
    base_pipe = build_pipeline(numeric_feats, categorical_feats)

    # 5) Define hyperparameter distributions for HistGradientBoostingRegressor
    param_dist = {
        # number of boosting iterations (analogous to n_estimators)
        "model__max_iter": [100, 200, 300, 500],
        # maximum tree depth
        "model__max_depth": [None, 10, 20, 30],
        # minimum samples per leaf
        "model__min_samples_leaf": [1, 2, 4],
        # learning rate for shrinking the contribution of each tree
        "model__learning_rate": [0.01, 0.1, 0.2],
        # maximum number of leaves per tree
        "model__max_leaf_nodes": [31, 50, 100],
        # L2 regularization strength
        "model__l2_regularization": [0.0, 0.1, 1.0],
    }

    # 6) Set up RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=base_pipe,
        param_distributions=param_dist,
        n_iter=20,               # number of parameter settings sampled
        scoring="neg_root_mean_squared_error",
        cv=5,                    # 5-fold CV
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    # 7) Run search
    print("Starting hyperparameter search...")
    search.fit(X_train, y_train)

    # 8) Show best parameters and CV score
    print("\nBest parameters found:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    print(f"Best CV RMSE: { -search.best_score_ :,.2f}")

    # 9) Evaluate on test set
    best_pipe = search.best_estimator_
    preds = best_pipe.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    print("\nTest set performance:")
    print(f"MAE : {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²  : {r2:.4f}")