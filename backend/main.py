# backend/main.py

import json
import joblib
import pandas as pd
import pydantic as pdt
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 1) Load model + feature defaults
model = joblib.load("models/best.pkl")

try:
    with open("models/feature_defaults.json", "r") as f:
        feature_defaults = json.load(f)
except FileNotFoundError:
    feature_defaults = {}
    print("⚠️  No feature_defaults.json found – you must run prepare_data.py first.")

app = FastAPI(title="Housing Price Estimator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class House(pdt.BaseModel):
    bed:    int
    bath:   int
    sqft:   int
    n_citi: str

@app.get("/")
def read_root():
    return {"message": "FastAPI is up and running!"}

@app.post("/predict")
def predict(house: House):
    # 2) Start from the saved defaults
    data = feature_defaults.copy()

    # 3) Map your UI fields into whatever names the model expects
    u = house.model_dump()
    # bedrooms
    if "bedrooms" in data:
        data["bedrooms"] = u["bed"]
    elif "bed" in data:
        data["bed"] = u["bed"]
    # bathrooms
    if "bathrooms" in data:
        data["bathrooms"] = u["bath"]
    elif "bath" in data:
        data["bath"] = u["bath"]
    # square footage
    if "sqft_living" in data:
        data["sqft_living"] = u["sqft"]
    elif "sqft" in data:
        data["sqft"] = u["sqft"]
    # city code / zipcode
    if "zipcode" in data:
        data["zipcode"] = u["n_citi"]
    elif "n_citi" in data:
        data["n_citi"] = u["n_citi"]

    # 4) Build a one-row DataFrame and predict
    df = pd.DataFrame([data])
    price = model.predict(df)[0]
    return {"predicted_price": round(float(price), 2)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)