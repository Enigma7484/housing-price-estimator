# AI Estimator Platform

AI Estimator Platform is a full-stack applied ML product for real-world prediction workflows. Phase 1 ships a Housing Price Estimator with a trained scikit-learn regression model, FastAPI inference service, and a polished React dashboard UI.

This is structured as an extensible estimator platform, not a notebook. New estimators can be added through dedicated training scripts, saved model artifacts, Pydantic schemas, FastAPI routes, services, and frontend pages.

## Features

- Housing price prediction from real property inputs
- Mobile price-range classification from device specifications
- Multiple model training and evaluation with MAE, RMSE, and R2
- Best-model artifact saved locally with metadata
- FastAPI inference endpoint with typed request and response schemas
- React + TypeScript + Vite frontend
- Dashboard-style UI with model status and metrics
- Docker support for local presentation
- Environment-based frontend API URL

## Tech Stack

- Frontend: React, TypeScript, Vite, Tailwind CSS, Axios, Lucide icons
- Backend: FastAPI, Pydantic, pandas, scikit-learn, joblib
- ML: Linear Regression baseline, RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
- Deployment path: local demo first, then Vercel frontend and Render/Railway/Fly.io backend

## Architecture

```text
frontend/
  React pages and reusable UI components
  estimatorApi.ts -> HTTP client

backend/
  app/main.py -> FastAPI app and CORS
  app/routes/ -> estimator routes
  app/schemas/ -> Pydantic contracts
  app/services/ -> model loading and inference logic
  app/models/ -> trained joblib artifacts
  training/ -> repeatable model training scripts
  data/ -> local datasets
```

## Local Setup

Backend:

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python training/train_housing_model.py
python training/train_mobile_model.py
uvicorn app.main:app --reload --port 8000
```

Windows PowerShell:

```powershell
cd backend
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python training/train_housing_model.py
python training/train_mobile_model.py
uvicorn app.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Use this frontend environment value:

```bash
VITE_API_BASE_URL=http://localhost:8000
```

## API Endpoints

- `GET /api/health`
- `GET /api/estimators`
- `GET /api/housing/metadata`
- `POST /api/housing/predict`
- `GET /api/mobile/metadata`
- `POST /api/mobile/predict`

Example housing request:

```json
{
  "square_footage": 2100,
  "lot_size": 6200,
  "bedrooms": 3,
  "bathrooms": 2.25,
  "floors": 2,
  "waterfront": false,
  "view": 0,
  "condition": 3,
  "grade": 8,
  "year_built": 1998,
  "year_renovated": 0,
  "zipcode": "98103",
  "parking": 1,
  "furnishing_status": "standard",
  "main_road_access": true,
  "basement": false,
  "air_conditioning": false
}
```

## Model Training

The housing model trains from `backend/data/housing/kc_house_data.csv`.

```bash
cd backend
python training/train_housing_model.py
```

The mobile model trains from `backend/data/mobile/train.csv` when present. If that file is missing, the trainer creates a deterministic bootstrap dataset so the local demo remains runnable.

```bash
cd backend
python training/train_mobile_model.py
```

The script cleans data, splits train/test, compares model candidates, selects the best R2 score, and saves:

- `backend/app/models/housing_model.joblib`
- `backend/app/models/housing_metrics.json`
- `backend/app/models/mobile_model.joblib`
- `backend/app/models/mobile_metrics.json`

## Docker

```bash
docker compose up --build
```

The backend runs on `http://localhost:8000` and the frontend runs on `http://localhost:5173`.

## Deployment

Recommended hosted setup:

- Frontend: Vercel project with root directory `frontend`
- Backend: Render web service from `render.yaml`

Render build command:

```bash
pip install -r requirements.txt && python training/train_housing_model.py && python training/train_mobile_model.py
```

Render start command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Vercel environment variable:

```bash
VITE_API_BASE_URL=https://ml-estimator-platform-api.onrender.com
```

If Render assigns a different service URL, update `VITE_API_BASE_URL` in Vercel and `ESTIMATOR_FRONTEND_ORIGINS` in Render.

## Screenshots

Add screenshots here after running the local demo:

- Homepage
- Housing estimator form and result card
- Model dashboard

## Roadmap

- Car, laptop, salary, rent, insurance, and used-phone estimators
- SHAP or feature-importance explainability
- Model registry and retraining metadata
- Authentication for business-facing demos
- Production Docker and cloud deployment profiles
- CI checks for backend and frontend

## Demo Positioning

This project is designed as a serious applied ML platform: modular model architecture, local trained artifacts, typed APIs, clean frontend workflows, and a direct path toward hosted demos with stronger infrastructure.
