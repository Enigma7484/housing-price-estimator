import { FormEvent, useMemo, useState } from "react";
import { Cpu, Smartphone } from "lucide-react";

import { predictMobile } from "../api/estimatorApi";
import ErrorMessage from "../components/ErrorMessage";
import InputField from "../components/InputField";
import LoadingSpinner from "../components/LoadingSpinner";
import MobileResultCard from "../components/MobileResultCard";
import type { MobilePayload, MobilePrediction } from "../types/estimatorTypes";

const initialPayload: MobilePayload = {
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

export default function MobileEstimator() {
  const [form, setForm] = useState<MobilePayload>(initialPayload);
  const [result, setResult] = useState<MobilePrediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const specIndex = useMemo(() => {
    let score = 20;
    score += Math.min(form.ram / 120, 36);
    score += Math.min(form.battery_power / 90, 24);
    score += Math.min(form.internal_memory / 8, 18);
    score += form.four_g ? 8 : 0;
    score += form.primary_camera_mp >= 48 ? 8 : 0;
    return Math.max(0, Math.min(Math.round(score), 100));
  }, [form]);

  function updateField(name: string, value: string) {
    setForm((current) => ({
      ...current,
      [name]: name === "clock_speed" || name === "mobile_depth_cm" ? Number(value) : Number.parseInt(value || "0", 10),
    }));
  }

  function updateToggle(name: keyof MobilePayload) {
    setForm((current) => ({ ...current, [name]: !current[name] }));
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    try {
      setResult(await predictMobile(form));
    } catch (predictionError) {
      setError(
        predictionError instanceof Error
          ? predictionError.message
          : "Unable to classify this mobile profile. Check that the backend is running and the model is trained.",
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
              <Smartphone className="h-5 w-5 text-signal" />
            </span>
            <span className="rounded-md bg-signal/10 px-2.5 py-1 text-xs font-semibold text-signal">Phase 2 live</span>
          </div>
          <h1 className="text-3xl font-semibold">Mobile Price Range Estimator</h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
            Classify mobile devices into low, medium, high, or very high cost ranges using hardware and connectivity
            specifications.
          </p>
        </div>
        <div className="rounded-lg border border-line bg-mist p-4">
          <p className="text-xs font-semibold uppercase text-slate-500">Spec strength index</p>
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
              <h2 className="text-xl font-semibold">Device Inputs</h2>
              <p className="mt-1 text-sm text-slate-500">Core specification signals used by the classifier.</p>
            </div>
            <Cpu className="h-5 w-5 text-signal" />
          </div>

          <div className="grid gap-4 sm:grid-cols-2">
            <InputField label="Battery power" name="battery_power" value={form.battery_power} min={500} max={2200} onChange={updateField} />
            <InputField label="RAM (MB)" name="ram" value={form.ram} min={256} max={8192} onChange={updateField} />
            <InputField label="Internal memory (GB)" name="internal_memory" value={form.internal_memory} min={2} max={256} onChange={updateField} />
            <InputField label="Clock speed (GHz)" name="clock_speed" value={form.clock_speed} min={0.5} max={3.5} step={0.1} onChange={updateField} />
            <InputField label="CPU cores" name="n_cores" value={form.n_cores} min={1} max={12} onChange={updateField} />
            <InputField label="Weight (g)" name="mobile_weight" value={form.mobile_weight} min={80} max={260} onChange={updateField} />
            <InputField label="Primary camera (MP)" name="primary_camera_mp" value={form.primary_camera_mp} min={0} max={108} onChange={updateField} />
            <InputField label="Front camera (MP)" name="front_camera_mp" value={form.front_camera_mp} min={0} max={64} onChange={updateField} />
            <InputField label="Pixel height" name="pixel_height" value={form.pixel_height} min={240} max={3200} onChange={updateField} />
            <InputField label="Pixel width" name="pixel_width" value={form.pixel_width} min={240} max={3200} onChange={updateField} />
            <InputField label="Screen height (cm)" name="screen_height_cm" value={form.screen_height_cm} min={5} max={25} onChange={updateField} />
            <InputField label="Screen width (cm)" name="screen_width_cm" value={form.screen_width_cm} min={3} max={15} onChange={updateField} />
            <InputField label="Talk time (hours)" name="talk_time" value={form.talk_time} min={2} max={32} onChange={updateField} />
            <InputField label="Depth (cm)" name="mobile_depth_cm" value={form.mobile_depth_cm} min={0.1} max={1.5} step={0.1} onChange={updateField} />
          </div>

          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            {[
              ["bluetooth", "Bluetooth"],
              ["dual_sim", "Dual SIM"],
              ["four_g", "4G"],
              ["three_g", "3G"],
              ["touch_screen", "Touch screen"],
              ["wifi", "Wi-Fi"],
            ].map(([name, label]) => (
              <label key={name} className="flex items-center justify-between rounded-md border border-line px-3 py-3 text-sm font-medium">
                {label}
                <input
                  type="checkbox"
                  checked={Boolean(form[name as keyof MobilePayload])}
                  onChange={() => updateToggle(name as keyof MobilePayload)}
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
            {loading ? <LoadingSpinner /> : "Classify device"}
          </button>
        </form>

        <MobileResultCard result={result} />
      </div>
    </div>
  );
}
