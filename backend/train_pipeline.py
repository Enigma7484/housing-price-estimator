from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold

def build_tabnet_pipeline(numeric_features, categorical_features):
    # Numeric pipeline: median impute → standard scale
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical pipeline: constant impute → one-hot encode (dense output)
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", TabNetRegressor(
            optimizer_params={"lr":1e-3},
            scheduler_params={"step_size":50, "gamma":0.9},
            scheduler_fn=StepLR,      # <-- pass the callable, not a string
            mask_type="entmax",
            n_d=16, n_a=16, n_steps=5,
            lambda_sparse=0.0001,
            seed=42,
            verbose=0
        )),
    ])

    return pipeline

def build_pipeline(num_cols, cat_cols):
    num_pipe = Pipeline([
        ("imp",   SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imp",  SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    head = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=None,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )

    return Pipeline([
        ("preprocessor", pre),
        ("var_thresh",   VarianceThreshold(threshold=0.0)),  # drop const cols
        ("model",        head),
    ])