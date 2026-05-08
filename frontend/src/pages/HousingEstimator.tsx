import { FormEvent, useMemo, useState } from "react";
import { Building2, Calculator } from "lucide-react";

import { predictHousing } from "../api/estimatorApi";
import ErrorMessage from "../components/ErrorMessage";
import InputField from "../components/InputField";
import LoadingSpinner from "../components/LoadingSpinner";
import ResultCard from "../components/ResultCard";
import type { HousingPayload, HousingPrediction } from "../types/estimatorTypes";

const initialPayload: HousingPayload = {
  square_footage: 2100,
  lot_size: 6200,
  bedrooms: 3,
  bathrooms: 2.25,
  floors: 2,
  waterfront: false,
  view: 0,
  condition: 3,
  grade: 8,
  year_built: 1998,
  year_renovated: 0,
  zipcode: "98103",
  latitude: null,
  longitude: null,
  parking: 1,
  furnishing_status: "standard",
  main_road_access: true,
  basement: false,
  air_conditioning: false,
};

export default function HousingEstimator() {
  const [form, setForm] = useState<HousingPayload>(initialPayload);
  const [result, setResult] = useState<HousingPrediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const propertyScore = useMemo(() => {
    let score = 40;
    score += Math.min(form.square_footage / 120, 32);
    score += (form.grade - 6) * 5;
    score += form.waterfront ? 12 : 0;
    score += form.basement ? 4 : 0;
    score += form.air_conditioning ? 3 : 0;
    return Math.max(0, Math.min(Math.round(score), 100));
  }, [form]);

  function updateField(name: string, value: string) {
    setForm((current) => ({
      ...current,
      [name]: name === "zipcode" ? value : Number(value),
    }));
  }

  function updateToggle(name: keyof HousingPayload) {
    setForm((current) => ({ ...current, [name]: !current[name] }));
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      const prediction = await predictHousing(form);
      setResult(prediction);
    } catch (predictionError) {
      setError(
        predictionError instanceof Error
          ? predictionError.message
          : "Unable to generate the housing estimate. Check that the backend is running.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <section className="flex flex-col justify-between gap-4 rounded-lg border border-line bg-white p-6 shadow-panel lg:flex-row lg:items-center">
        <div>
          <div className="mb-3 flex items-center gap-3">
            <span className="grid h-10 w-10 place-items-center rounded-md bg-signal/10">
              <Building2 className="h-5 w-5 text-signal" />
            </span>
            <span className="rounded-md bg-mist px-2.5 py-1 text-xs font-semibold text-graphite">Phase 1 live</span>
          </div>
          <h1 className="text-3xl font-semibold">Housing Price Estimator</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
            Submit a structured property profile and receive a model-powered price estimate with range, confidence, and
            valuation signals.
          </p>
        </div>
        <div className="rounded-lg border border-line bg-mist p-4">
          <p className="text-xs font-semibold uppercase text-slate-500">Property strength index</p>
          <div className="mt-3 h-3 w-56 overflow-hidden rounded-full bg-white">
            <div className="h-full rounded-full bg-signal" style={{ width: `${propertyScore}%` }} />
          </div>
          <p className="mt-2 text-sm font-semibold">{propertyScore}/100</p>
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-[0.98fr_1.02fr]">
        <form onSubmit={submit} className="rounded-lg border border-line bg-white p-6 shadow-panel">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold">Property Inputs</h2>
              <p className="mt-1 text-sm text-slate-500">Core housing signals used by the trained model.</p>
            </div>
            <Calculator className="h-5 w-5 text-signal" />
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <InputField label="Square footage" name="square_footage" value={form.square_footage} min={250} max={20000} onChange={updateField} />
            <InputField label="Lot size" name="lot_size" value={form.lot_size} min={500} max={200000} onChange={updateField} />
            <InputField label="Bedrooms" name="bedrooms" value={form.bedrooms} min={0} max={15} onChange={updateField} />
            <InputField label="Bathrooms" name="bathrooms" value={form.bathrooms} min={0} max={10} step={0.25} onChange={updateField} />
            <InputField label="Floors" name="floors" value={form.floors} min={1} max={4} step={0.5} onChange={updateField} />
            <InputField label="Parking spaces" name="parking" value={form.parking} min={0} max={8} onChange={updateField} />
            <InputField label="View score" name="view" value={form.view} min={0} max={4} onChange={updateField} />
            <InputField label="Condition" name="condition" value={form.condition} min={1} max={5} onChange={updateField} />
            <InputField label="Construction grade" name="grade" value={form.grade} min={1} max={13} onChange={updateField} />
            <InputField label="Year built" name="year_built" value={form.year_built} min={1900} max={2026} onChange={updateField} />
            <InputField label="Year renovated" name="year_renovated" value={form.year_renovated} min={0} max={2026} onChange={updateField} />
            <InputField label="Zipcode" name="zipcode" value={form.zipcode} type="text" onChange={updateField} />
          </div>

          <div className="mt-5">
            <span className="mb-2 block text-sm font-medium text-graphite">Furnishing status</span>
            <div className="grid grid-cols-3 gap-2">
              {(["basic", "standard", "premium"] as const).map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setForm((current) => ({ ...current, furnishing_status: option }))}
                  className={`h-10 rounded-md border text-sm font-medium capitalize ${
                    form.furnishing_status === option ? "border-signal bg-signal text-white" : "border-line bg-white text-graphite"
                  }`}
                >
                  {option}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            {[
              ["waterfront", "Waterfront"],
              ["main_road_access", "Main road access"],
              ["basement", "Basement"],
              ["air_conditioning", "Air conditioning"],
            ].map(([name, label]) => (
              <label key={name} className="flex items-center justify-between rounded-md border border-line px-3 py-3 text-sm font-medium">
                {label}
                <input
                  type="checkbox"
                  checked={Boolean(form[name as keyof HousingPayload])}
                  onChange={() => updateToggle(name as keyof HousingPayload)}
                  className="h-4 w-4 accent-signal"
                />
              </label>
            ))}
          </div>

          {error ? <div className="mt-5"><ErrorMessage message={error} /></div> : null}

          <button
            type="submit"
            disabled={loading}
            className="mt-6 inline-flex h-11 w-full items-center justify-center rounded-md bg-ink px-4 text-sm font-semibold text-white transition hover:bg-graphite disabled:cursor-not-allowed disabled:opacity-70"
          >
            {loading ? <LoadingSpinner /> : "Generate estimate"}
          </button>
        </form>

        <ResultCard result={result} />
      </div>
    </div>
  );
}
