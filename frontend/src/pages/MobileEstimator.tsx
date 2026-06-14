import { FormEvent, useMemo, useState } from "react";
import { ClipboardCheck, Cpu, Smartphone } from "lucide-react";

import { predictMobile } from "../api/estimatorApi";
import ErrorMessage from "../components/ErrorMessage";
import InputField from "../components/InputField";
import LoadingSpinner from "../components/LoadingSpinner";
import MobileResultCard from "../components/MobileResultCard";
import type { MobilePayload } from "../types/estimatorTypes";
import type { DeviceQuote, DeviceQuoteForm } from "../types/quoteTypes";
import { buildDeviceQuote, storeQuote } from "../utils/deviceQuoteEngine";

const initialPayload: DeviceQuoteForm = {
  device_model: "iPhone 13",
  storage_gb: 128,
  condition: "good",
  battery_health: 87,
  carrier_status: "unlocked",
  has_box: false,
  has_charger: true,
  needs_repair: false,
  battery_power: 1800,
  clock_speed: 2.4,
  ram: 4096,
  internal_memory: 128,
  mobile_weight: 165,
  n_cores: 8,
  primary_camera_mp: 48,
  front_camera_mp: 16,
  pixel_height: 1920,
  pixel_width: 1080,
  screen_height_cm: 15,
  screen_width_cm: 7,
  talk_time: 20,
  mobile_depth_cm: 0.8,
  bluetooth: true,
  dual_sim: true,
  four_g: true,
  three_g: true,
  touch_screen: true,
  wifi: true,
};

const modelFields: Array<keyof MobilePayload> = [
  "battery_power",
  "clock_speed",
  "ram",
  "internal_memory",
  "mobile_weight",
  "n_cores",
  "primary_camera_mp",
  "front_camera_mp",
  "pixel_height",
  "pixel_width",
  "screen_height_cm",
  "screen_width_cm",
  "talk_time",
  "mobile_depth_cm",
  "bluetooth",
  "dual_sim",
  "four_g",
  "three_g",
  "touch_screen",
  "wifi",
];

export default function MobileEstimator() {
  const [form, setForm] = useState<DeviceQuoteForm>(initialPayload);
  const [quote, setQuote] = useState<DeviceQuote | null>(null);
  const [savedMessage, setSavedMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const specIndex = useMemo(() => {
    let score = 20;
    score += Math.min(form.ram / 120, 36);
    score += Math.min(form.battery_power / 90, 24);
    score += Math.min(form.internal_memory / 8, 18);
    score += form.four_g ? 8 : 0;
    score += form.primary_camera_mp >= 48 ? 8 : 0;
    score += form.condition === "excellent" ? 6 : form.condition === "fair" ? -4 : 0;
    return Math.max(0, Math.min(Math.round(score), 100));
  }, [form]);

  function updateField(name: string, value: string) {
    setForm((current) => ({
      ...current,
      [name]: parseFieldValue(name, value),
      ...(name === "storage_gb" ? { internal_memory: Number.parseInt(value || "0", 10) } : {}),
    }));
  }

  function updateSelect(name: string, value: string) {
    setForm((current) => ({ ...current, [name]: value }));
  }

  function updateToggle(name: keyof DeviceQuoteForm) {
    setForm((current) => ({ ...current, [name]: !current[name] }));
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setSavedMessage("");
    try {
      const prediction = await predictMobile(toMobilePayload(form));
      setQuote(buildDeviceQuote(form, prediction));
    } catch (predictionError) {
      setError(
        predictionError instanceof Error
          ? predictionError.message
          : "Unable to generate this quote. Check that the backend is running and the model is trained.",
      );
    } finally {
      setLoading(false);
    }
  }

  function saveQuote() {
    if (!quote) {
      return;
    }
    storeQuote(quote);
    setSavedMessage("Quote saved to owner history.");
  }

  return (
    <div className="space-y-6">
      <section className="flex flex-col justify-between gap-4 rounded-lg border border-line bg-white p-6 shadow-panel lg:flex-row lg:items-center">
        <div>
          <div className="mb-3 flex items-center gap-3">
            <span className="grid h-10 w-10 place-items-center rounded-md bg-signal/10">
              <Smartphone className="h-5 w-5 text-signal" />
            </span>
            <span className="rounded-md bg-signal/10 px-2.5 py-1 text-xs font-semibold text-signal">Quote desk</span>
          </div>
          <h1 className="text-3xl font-semibold">Device Trade-In Quote</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
            Turn device condition and hardware signals into a recommended buy offer, resale target, margin, and customer-ready explanation.
          </p>
        </div>
        <div className="rounded-lg border border-line bg-mist p-4">
          <p className="text-xs font-semibold uppercase text-slate-500">Resale strength</p>
          <div className="mt-3 h-3 w-56 overflow-hidden rounded-full bg-white">
            <div className="h-full rounded-full bg-signal" style={{ width: `${specIndex}%` }} />
          </div>
          <p className="mt-2 text-sm font-semibold">{specIndex}/100</p>
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-[0.98fr_1.02fr]">
        <form onSubmit={submit} className="rounded-lg border border-line bg-white p-6 shadow-panel">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold">Quote Inputs</h2>
              <p className="mt-1 text-sm text-slate-500">Business-facing details first, model signals underneath.</p>
            </div>
            <ClipboardCheck className="h-5 w-5 text-signal" />
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <InputField label="Device model" name="device_model" value={form.device_model} type="text" onChange={updateField} />
            <InputField label="Storage (GB)" name="storage_gb" value={form.storage_gb} min={16} max={1024} onChange={updateField} />
            <SelectField label="Condition" name="condition" value={form.condition} onChange={updateSelect} options={["fair", "good", "excellent"]} />
            <InputField label="Battery health (%)" name="battery_health" value={form.battery_health} min={40} max={100} onChange={updateField} />
            <SelectField label="Carrier status" name="carrier_status" value={form.carrier_status} onChange={updateSelect} options={["unlocked", "locked", "unknown"]} />
            <InputField label="RAM (MB)" name="ram" value={form.ram} min={256} max={8192} onChange={updateField} />
            <InputField label="Battery power" name="battery_power" value={form.battery_power} min={500} max={2200} onChange={updateField} />
            <InputField label="Primary camera (MP)" name="primary_camera_mp" value={form.primary_camera_mp} min={0} max={108} onChange={updateField} />
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            {[
              ["has_charger", "Charger included"],
              ["has_box", "Original box"],
              ["needs_repair", "Needs repair"],
              ["four_g", "4G capable"],
              ["touch_screen", "Touch screen"],
              ["wifi", "Wi-Fi"],
            ].map(([name, label]) => (
              <label key={name} className="flex items-center justify-between rounded-md border border-line px-3 py-3 text-sm font-medium">
                {label}
                <input
                  type="checkbox"
                  checked={Boolean(form[name as keyof DeviceQuoteForm])}
                  onChange={() => updateToggle(name as keyof DeviceQuoteForm)}
                  className="h-4 w-4 accent-signal"
                />
              </label>
            ))}
          </div>

          <details className="mt-5 rounded-md border border-line bg-mist p-4">
            <summary className="flex cursor-pointer items-center gap-2 text-sm font-semibold text-graphite">
              <Cpu className="h-4 w-4 text-signal" />
              Advanced model signals
            </summary>
            <div className="mt-4 grid gap-4 sm:grid-cols-2">
              <InputField label="Clock speed (GHz)" name="clock_speed" value={form.clock_speed} min={0.5} max={3.5} step={0.1} onChange={updateField} />
              <InputField label="CPU cores" name="n_cores" value={form.n_cores} min={1} max={12} onChange={updateField} />
              <InputField label="Weight (g)" name="mobile_weight" value={form.mobile_weight} min={80} max={260} onChange={updateField} />
              <InputField label="Front camera (MP)" name="front_camera_mp" value={form.front_camera_mp} min={0} max={64} onChange={updateField} />
              <InputField label="Pixel height" name="pixel_height" value={form.pixel_height} min={240} max={3200} onChange={updateField} />
              <InputField label="Pixel width" name="pixel_width" value={form.pixel_width} min={240} max={3200} onChange={updateField} />
              <InputField label="Screen height (cm)" name="screen_height_cm" value={form.screen_height_cm} min={5} max={25} onChange={updateField} />
              <InputField label="Screen width (cm)" name="screen_width_cm" value={form.screen_width_cm} min={3} max={15} onChange={updateField} />
              <InputField label="Talk time (hours)" name="talk_time" value={form.talk_time} min={2} max={32} onChange={updateField} />
              <InputField label="Depth (cm)" name="mobile_depth_cm" value={form.mobile_depth_cm} min={0.1} max={1.5} step={0.1} onChange={updateField} />
            </div>
          </details>

          {error ? <div className="mt-5"><ErrorMessage message={error} /></div> : null}
          {savedMessage ? <p className="mt-5 rounded-md bg-signal/10 px-3 py-2 text-sm font-semibold text-signal">{savedMessage}</p> : null}

          <button
            type="submit"
            disabled={loading}
            className="mt-6 inline-flex h-11 w-full items-center justify-center rounded-md bg-ink px-4 text-sm font-semibold text-white transition hover:bg-graphite disabled:cursor-not-allowed disabled:opacity-70"
          >
            {loading ? <LoadingSpinner /> : "Generate quote"}
          </button>
        </form>

        <MobileResultCard quote={quote} onSave={saveQuote} />
      </div>
    </div>
  );
}
function toMobilePayload(form: DeviceQuoteForm): MobilePayload {
  return modelFields.reduce((payload, field) => {
    return { ...payload, [field]: form[field] };
  }, {} as MobilePayload);
}

function parseFieldValue(name: string, value: string) {
  if (name === "device_model") {
    return value;
  }
  if (name === "clock_speed" || name === "mobile_depth_cm") {
    return Number(value);
  }
  return Number.parseInt(value || "0", 10);
}

function SelectField({
  label,
  name,
  value,
  options,
  onChange,
}: {
  label: string;
  name: string;
  value: string;
  options: string[];
  onChange: (name: string, value: string) => void;
}) {
  return (
    <label className="block">
      <span className="mb-2 block text-sm font-medium text-graphite">{label}</span>
      <select
        className="h-11 w-full rounded-md border border-line bg-white px-3 text-sm capitalize text-ink outline-none transition focus:border-signal focus:ring-2 focus:ring-signal/20"
        name={name}
        value={value}
        onChange={(event) => onChange(name, event.target.value)}
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}
