# Backend

FastAPI service for the AI Estimator Platform.

## Local Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python training/train_housing_model.py
uvicorn app.main:app --reload --port 8000
```

PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python training/train_housing_model.py
uvicorn app.main:app --reload --port 8000
```

## Endpoints

- `GET /api/health`
- `GET /api/estimators`
- `GET /api/housing/metadata`
- `POST /api/housing/predict`
- `GET /api/mobile/metadata`
- `POST /api/mobile/predict`
- `GET /api/car/metadata`
- `POST /api/car/predict`
