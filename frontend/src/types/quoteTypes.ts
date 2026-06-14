import type { MobilePayload, MobilePrediction } from "./estimatorTypes";

export type DeviceCondition = "fair" | "good" | "excellent";
export type CarrierStatus = "unlocked" | "locked" | "unknown";
export type QuoteStatus = "draft" | "sent" | "accepted";

export type DeviceQuoteForm = MobilePayload & {
  device_model: string;
  storage_gb: number;
  condition: DeviceCondition;
  battery_health: number;
  carrier_status: CarrierStatus;
  has_box: boolean;
  has_charger: boolean;
  needs_repair: boolean;
};

export type DeviceQuote = {
  id: string;
  created_at: string;
  customer_label: string;
  status: QuoteStatus;
  device_model: string;
  condition: DeviceCondition;
  buy_offer: number;
  list_price: number;
  expected_margin: number;
  margin_rate: number;
  confidence: number;
  risk_level: "Low" | "Medium" | "High";
  model_tier: string;
  model_prediction: MobilePrediction;
  factors: string[];
  customer_note: string;
  input: DeviceQuoteForm;
};

