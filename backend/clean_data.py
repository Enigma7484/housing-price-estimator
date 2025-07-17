# backend/clean_data.py

import pandas as pd
from pathlib import Path
from image_transformer import add_image_features
from text_transformer import add_text_features
from chatgpt_features import add_gpt_features

def load_and_clean(path: str):
    df = pd.read_csv(path)

    # 1) Unify ID column
    if "image_id" in df.columns:
        df = df.rename(columns={"image_id": "id"})
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})

    # 2) Embed images only for SoCal files (train or train_aug)
    stem = Path(path).stem
    if stem.startswith("train") and Path("data/images").exists():
        df = add_image_features(df)

    # 3) (optional) text embeddings
    if "description" in df.columns:
        df = add_text_features(df)

    # 4) GPT-suggested features (no-op until you fill it)
    df = add_gpt_features(df)

    # 5) Drop raw columns
    for c in ["id","description","street","citi"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # 6) Date parsing for KC
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S")
        df["sale_year"]  = df["date"].dt.year
        df["sale_month"] = df["date"].dt.month
        df = df.drop(columns=["date"])

    # 7) Cap top 1% prices
    top = df["price"].quantile(0.99)
    df.loc[df["price"] > top, "price"] = top

    # 8) Cast any remaining object columns to str
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)

    return df

if __name__ == "__main__":
    df = load_and_clean("data/train.csv")
    print(df.shape, df.columns.tolist())