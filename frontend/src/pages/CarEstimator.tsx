import { FormEvent, useMemo, useState } from "react";
import { Car, Wrench } from "lucide-react";

import { predictCar } from "../api/estimatorApi";
import CarResultCard from "../components/CarResultCard";
import ErrorMessage from "../components/ErrorMessage";
import InputField from "../components/InputField";
import LoadingSpinner from "../components/LoadingSpinner";
import type { CarPayload, CarPrediction } from "../types/estimatorTypes";

const initialPayload: CarPayload = {
  make: "Toyota",
  model: "Camry",
  body_type: "sedan",
  fuel_type: "gasoline",
  transmission: "automatic",
  year: 2021,
  mileage: 42000,
  engine_size_l: 2.5,
  horsepower: 203,
  owners: 1,
  accident_history: false,
  condition_score: 4,
};

export default function CarEstimator() {
  const [form, setForm] = useState<CarPayload>(initialPayload);
  const [result, setResult] = useState<CarPrediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const resaleIndex = useMemo(() => {
    const age = 2026 - form.year;
    let score = 82;
    score -= Math.min(age * 4, 42);
    score -= Math.min(form.mileage / 6000, 32);
    score += form.condition_score * 6;
    score += form.fuel_type === "electric" || form.fuel_type === "hybrid" ? 6 : 0;
    score -= form.accident_history ? 16 : 0;
    return Math.max(0, Math.min(Math.round(score), 100));
  }, [form]);

  function updateField(name: string, value: string) {
    setForm((current) => ({
      ...current,
      [name]:
        name === "make" || name === "model"
          ? value
          : name === "engine_size_l"
            ? Number(value)
            : Number.parseInt(value || "0", 10),
    }));
  }

  function updateToggle(name: keyof CarPayload) {
    setForm((current) => ({ ...current, [name]: !current[name] }));
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      setResult(await predictCar(form));
    } catch (predictionError) {
      setError(
        predictionError instanceof Error
          ? predictionError.message
          : "Unable to estimate this vehicle. Check that the backend is running and the car model is trained.",
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
              <Car className="h-5 w-5 text-signal" />
            </span>
            <span className="rounded-md bg-signal/10 px-2.5 py-1 text-xs font-semibold text-signal">Phase 4 live</span>
          </div>
          <h1 className="text-3xl font-semibold">Car Price Estimator</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
            Estimate used vehicle resale value from make, model year, mileage, condition, ownership, and powertrain signals.
          </p>
        </div>
        <div className="rounded-lg border border-line bg-mist p-4">
          <p className="text-xs font-semibold uppercase text-slate-500">Resale strength index</p>
          <div className="mt-3 h-3 w-56 overflow-hidden rounded-full bg-white">
            <div className="h-full rounded-full bg-signal" style={{ width: `${resaleIndex}%` }} />
          </div>
          <p className="mt-2 text-sm font-semibold">{resaleIndex}/100</p>
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-[0.98fr_1.02fr]">
        <form onSubmit={submit} className="rounded-lg border border-line bg-white p-6 shadow-panel">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold">Vehicle Inputs</h2>
              <p className="mt-1 text-sm text-slate-500">Core resale signals used by the model.</p>
            </div>
            <Wrench className="h-5 w-5 text-signal" />
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <InputField label="Make" name="make" value={form.make} type="text" onChange={updateField} />
            <InputField label="Model" name="model" value={form.model} type="text" onChange={updateField} />
            <InputField label="Year" name="year" value={form.year} min={1995} max={2026} onChange={updateField} />
            <InputField label="Mileage" name="mileage" value={form.mileage} min={0} max={350000} onChange={updateField} />
            <InputField label="Engine size (L)" name="engine_size_l" value={form.engine_size_l} min={0} max={8.5} step={0.1} onChange={updateField} />
            <InputField label="Horsepower" name="horsepower" value={form.horsepower} min={60} max={1000} onChange={updateField} />
            <InputField label="Owners" name="owners" value={form.owners} min={1} max={8} onChange={updateField} />
            <InputField label="Condition score" name="condition_score" value={form.condition_score} min={1} max={5} onChange={updateField} />
          </div>

          <SegmentedControl
            label="Body type"
            value={form.body_type}
            options={["sedan", "suv", "truck", "hatchback", "coupe", "wagon"]}
            onChange={(value) => setForm((current) => ({ ...current, body_type: value as CarPayload["body_type"] }))}
          />
          <SegmentedControl
            label="Fuel type"
            value={form.fuel_type}
            options={["gasoline", "diesel", "hybrid", "electric"]}
            onChange={(value) => setForm((current) => ({ ...current, fuel_type: value as CarPayload["fuel_type"] }))}
          />
          <SegmentedControl
            label="Transmission"
            value={form.transmission}
            options={["automatic", "manual"]}
            onChange={(value) => setForm((current) => ({ ...current, transmission: value as CarPayload["transmission"] }))}
          />

          <div className="mt-5">
            <label className="flex items-center justify-between rounded-md border border-line px-3 py-3 text-sm font-medium">
              Accident history
              <input
                type="checkbox"
                checked={form.accident_history}
                onChange={() => updateToggle("accident_history")}
                className="h-4 w-4 accent-signal"
              />
            </label>
          </div>

          {error ? <div className="mt-5"><ErrorMessage message={error} /></div> : null}

          <button
            type="submit"
            disabled={loading}
            className="mt-6 inline-flex h-11 w-full items-center justify-center rounded-md bg-ink px-4 text-sm font-semibold text-white transition hover:bg-graphite disabled:cursor-not-allowed disabled:opacity-70"
          >
            {loading ? <LoadingSpinner /> : "Estimate resale value"}
          </button>
        </form>

        <CarResultCard result={result} />
      </div>
    </div>
  );
}

function SegmentedControl({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <div className="mt-5">
      <span className="mb-2 block text-sm font-medium text-graphite">{label}</span>
      <div className="grid gap-2 sm:grid-cols-3">
        {options.map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => onChange(option)}
            className={`h-10 rounded-md border text-sm font-medium capitalize ${
              value === option ? "border-signal bg-signal text-white" : "border-line bg-white text-graphite"
            }`}
          >
            {option}
          </button>
        ))}
      </div>
    </div>
  );
}
