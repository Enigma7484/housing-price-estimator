from __future__ import annotations

import pandas as pd


def clean_housing_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [column.strip() for column in cleaned.columns]
    if "date" in cleaned.columns:
        cleaned = cleaned.drop(columns=["date"])
    if "id" in cleaned.columns:
        cleaned = cleaned.drop(columns=["id"])
    cleaned = cleaned.dropna(subset=["price"])

    for column in cleaned.columns:
        if column == "price":
            continue
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        if cleaned[column].isna().any():
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    return cleaned
