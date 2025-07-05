import pandas as pd

def load_and_clean(path="kc_house_data.csv"):
    # 1) Load
    df = pd.read_csv(path)

    # 2) Drop ID
    df = df.drop(columns=["id"])

    # 3) Parse 'date' and extract year/month
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S")
    df["sale_year"]  = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month
    df = df.drop(columns=["date"])

    # 4) Outlier capping on target
    upper_price = df["price"].quantile(0.99)
    df.loc[df["price"] > upper_price, "price"] = upper_price

    # 5) Cast numeric‐coded categories to strings
    cat_cols = [
        "zipcode",
        "view",
        "condition",
        "waterfront",
        "grade",
        "sale_year",
        "sale_month",
    ]
    df[cat_cols] = df[cat_cols].astype(str)

    return df

if __name__ == "__main__":
    cleaned = load_and_clean()
    print(cleaned.head())
    print("Shape:", cleaned.shape)