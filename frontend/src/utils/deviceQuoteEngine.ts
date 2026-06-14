import type { DeviceQuote, DeviceQuoteForm } from "../types/quoteTypes";
import type { MobilePrediction } from "../types/estimatorTypes";

const tierBasePrices: Record<number, number> = {
  0: 125,
  1: 255,
  2: 440,
  3: 720,
};

const conditionMultipliers = {
  fair: 0.78,
  good: 0.92,
  excellent: 1.08,
};

const carrierAdjustments = {
  unlocked: 1.06,
  locked: 0.88,
  unknown: 0.95,
};

export function buildDeviceQuote(input: DeviceQuoteForm, prediction: MobilePrediction): DeviceQuote {
  const basePrice = tierBasePrices[prediction.predicted_price_range] ?? 275;
  const storageBoost = Math.min(Math.max((input.storage_gb - 64) * 0.55, -35), 120);
  const batteryAdjustment = input.battery_health >= 90 ? 32 : input.battery_health >= 80 ? 12 : input.battery_health >= 70 ? -28 : -72;
  const accessoryBoost = (input.has_box ? 12 : 0) + (input.has_charger ? 10 : 0);
  const repairPenalty = input.needs_repair ? 110 : 0;

  const listPrice = Math.max(
    65,
    Math.round(
      (basePrice + storageBoost + batteryAdjustment + accessoryBoost - repairPenalty) *
        conditionMultipliers[input.condition] *
        carrierAdjustments[input.carrier_status],
    ),
  );
  const targetMarginRate = input.needs_repair ? 0.42 : input.condition === "fair" ? 0.36 : 0.31;
  const buyOffer = Math.max(25, Math.round(listPrice * (1 - targetMarginRate)));
  const expectedMargin = listPrice - buyOffer;
  const marginRate = expectedMargin / listPrice;
  const confidence = Math.max(0.35, Math.min(0.96, prediction.confidence - (input.needs_repair ? 0.08 : 0) - (input.carrier_status === "unknown" ? 0.05 : 0)));

  const factors = [
    `${prediction.label} model tier from hardware profile`,
    `${capitalize(input.condition)} cosmetic condition`,
    `${input.battery_health}% battery health`,
    input.carrier_status === "unlocked" ? "Unlocked carrier status improves resale demand" : "Carrier status reduces offer certainty",
    input.needs_repair ? "Repair flag increases margin buffer" : "No repair flag entered",
  ];

  return {
    id: crypto.randomUUID(),
    created_at: new Date().toISOString(),
    customer_label: `${input.device_model} quote`,
    status: "draft",
    device_model: input.device_model,
    condition: input.condition,
    buy_offer: buyOffer,
    list_price: listPrice,
    expected_margin: expectedMargin,
    margin_rate: marginRate,
    confidence,
    risk_level: confidence >= 0.72 && marginRate >= 0.3 ? "Low" : confidence >= 0.55 && marginRate >= 0.24 ? "Medium" : "High",
    model_tier: prediction.label,
    model_prediction: prediction,
    factors,
    customer_note: `Based on the ${input.device_model} condition, storage, battery health, and market tier, we can offer ${formatCurrency(buyOffer)} today. Estimated resale target is ${formatCurrency(listPrice)} after inspection.`,
    input,
  };
}

export function getStoredQuotes(): DeviceQuote[] {
  try {
    const raw = localStorage.getItem("resaleiq.quotes");
    if (!raw) {
      return [];
    }
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function storeQuote(quote: DeviceQuote) {
  const quotes = [quote, ...getStoredQuotes()].slice(0, 50);
  localStorage.setItem("resaleiq.quotes", JSON.stringify(quotes));
  window.dispatchEvent(new Event("resaleiq:quotes-updated"));
}

export function formatCurrency(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

