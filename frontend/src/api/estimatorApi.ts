import axios from "axios";

import type {
  HousingPayload,
  HousingPrediction,
  EstimatorCatalogResponse,
  MobilePayload,
  MobilePrediction,
  ModelMetadata,
} from "../types/estimatorTypes";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || (import.meta.env.DEV ? "http://localhost:8000" : "");

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    "Content-Type": "application/json",
  },
});

export async function predictHousing(payload: HousingPayload): Promise<HousingPrediction> {
  const response = await client.post<HousingPrediction>("/api/housing/predict", payload);
  return response.data;
}

export async function getHousingMetadata(): Promise<ModelMetadata> {
  const response = await client.get<ModelMetadata>("/api/housing/metadata");
  return response.data;
}

export async function predictMobile(payload: MobilePayload): Promise<MobilePrediction> {
  const response = await client.post<MobilePrediction>("/api/mobile/predict", payload);
  return response.data;
}

export async function getMobileMetadata(): Promise<ModelMetadata> {
  const response = await client.get<ModelMetadata>("/api/mobile/metadata");
  return response.data;
}

export async function getHealth() {
  const response = await client.get("/api/health");
  return response.data;
}

export async function getEstimatorCatalog(): Promise<EstimatorCatalogResponse> {
  const response = await client.get<EstimatorCatalogResponse>("/api/estimators");
  return response.data;
}
