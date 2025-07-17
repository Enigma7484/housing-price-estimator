# backend/prepare_data.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics        import mean_absolute_error, r2_score, root_mean_squared_error
from clean_data             import load_and_clean
from train_pipeline         import build_pipeline
import joblib

# 1) Choose data
mode = sys.argv[1] if len(sys.argv)>1 else "socal"
if mode == "kc":
    csv = "data/kc_house_data.csv"
elif mode in ("aug","socal_aug"):
    csv = "data/train_aug.csv"
else:
    csv = "data/train.csv"

print("⏳ Loading", csv)
df = load_and_clean(csv)

# 2) Split
X = df.drop("price", axis=1)
y = df["price"]
Xtr, Xte, ytr, yte = train_test_split(X,y, test_size=0.2, random_state=42)

# 3) Auto-detect columns
num_cols = Xtr.select_dtypes(include="number").columns.tolist()
cat_cols = Xtr.select_dtypes(include="object").columns.tolist()
print("NUM:", num_cols)
print("CAT:", cat_cols)

# 4) Build & train
pipe = build_pipeline(num_cols, cat_cols)
pipe.fit(Xtr, ytr)

# 5) Predict & eval
pred = pipe.predict(Xte)
print("MAE :", f"{mean_absolute_error(yte,pred):,.0f}")
print("RMSE:", f"{root_mean_squared_error(yte,pred):,.0f}")
print("R²  :", f"{r2_score(yte,pred):.3f}")

# 6) Save model for serving
Path("models").mkdir(exist_ok=True)
joblib.dump(pipe, "models/best.pkl")
print("✅ Saved models/best.pkl")