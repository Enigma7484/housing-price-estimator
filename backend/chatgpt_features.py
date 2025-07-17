import pandas as pd
import numpy as np
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def suggest_features(schema_sample: pd.DataFrame, n_suggestions: int = 10) -> str:
    cols = ", ".join(schema_sample.columns.tolist())
    prompt = (
        f"I have a housing dataset with columns: {cols}. "
        f"Suggest {n_suggestions} useful feature engineering ideas (interactions, transforms)."
    )
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.7
    )
    return resp.choices[0].message.content

def add_gpt_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a small set of engineered features *only* if the source columns exist.
    Prevents KeyError on datasets with different schemas.
    """
    # --- bed/bath ratio ---
    if "bed" in df.columns and "bath" in df.columns:
        df["bed_bath_ratio"] = df["bed"] / (df["bath"] + 1e-3)
    elif "bedrooms" in df.columns and "bathrooms" in df.columns:
        df["bed_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 1e-3)

    # --- sqft per bed & log sqft ---
    sqft_col = None
    if "sqft" in df.columns:
        sqft_col = "sqft"
    elif "sqft_living" in df.columns:
        sqft_col = "sqft_living"

    # use whichever bedrooms column exists
    bed_col = None
    if "bed" in df.columns:
        bed_col = "bed"
    elif "bedrooms" in df.columns:
        bed_col = "bedrooms"

    if sqft_col:
        df["log_" + sqft_col] = np.log1p(df[sqft_col])
        if bed_col:
            df["sqft_per_bed"] = df[sqft_col] / (df[bed_col] + 1e-3)

    return df