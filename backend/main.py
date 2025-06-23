# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Housing Price Estimator")

@app.get("/")
def read_root():
    return {"message": "FastAPI is up and running!"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React’s dev server
    allow_methods=["*"],
    allow_headers=["*"],
)