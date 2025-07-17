# backend/main.py
import joblib, pandas as pd, pydantic as pdt
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Housing Price Estimator")
model = joblib.load("models/best.pkl")

class House(pdt.BaseModel):
    # minimal schema – include every feature your UI sends
    bed:  int
    bath: int
    sqft: int
    n_citi: str

@app.get("/")
def read_root():
    return {"message": "FastAPI is up and running!"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React’s dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(house: House):
    df = pd.DataFrame([house.model_dump()])
    price = model.predict(df)[0]
    return {"predicted_price": round(float(price), 2)}