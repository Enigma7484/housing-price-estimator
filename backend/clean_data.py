import pandas as pd

def load_and_clean(path="kc_house_data.csv"):
    df = pd.read_csv(path)

    # 1) Drop ID
    df = df.drop(columns=["id"])

    # 2) Convert 'date' to datetime and extract year/month
    df["date"] = pd.to_datetime(
        df["date"],
        format="%Y%m%dT%H%M%S"
    )
    df["sale_year"]  = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month
    df = df.drop(columns=["date"])

    # 3) (Later) handle missing values & outliers...
    # cap 'price' at the 99th percentile
    upper_price = df["price"].quantile(0.99)
    df.loc[df["price"] > upper_price, "price"] = upper_price

    # → cast numeric‐coded categories to strings
    cat_cols = [
        "zipcode", "view", "condition", "waterfront",
        "grade", "sale_year", "sale_month"
    ]
    df[cat_cols] = df[cat_cols].astype(str)

    return df

if __name__ == "__main__":
    clean_df = load_and_clean()
    print(clean_df.head())
    print("Shape after dropping id/date:", clean_df.shape)