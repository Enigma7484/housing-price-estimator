import { Cpu, Gauge, Signal } from "lucide-react";

import type { MobilePrediction } from "../types/estimatorTypes";

type MobileResultCardProps = {
  result: MobilePrediction | null;
};

export default function MobileResultCard({ result }: MobileResultCardProps) {
  if (!result) {
    return (
      <section className="rounded-lg border border-dashed border-line bg-white p-6 shadow-panel">
        <div className="flex items-center gap-3 text-graphite">
          <Cpu className="h-5 w-5 text-signal" />
          <p className="text-sm">Submit mobile specifications to classify the device price range.</p>
        </div>
      </section>
    );
  }

  const probabilityRows = Object.entries(result.probabilities);

  return (
    <section className="rounded-lg border border-line bg-white p-6 shadow-panel">
      <div className="flex flex-col gap-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm font-medium text-slate-500">Predicted Price Range</p>
            <h2 className="mt-1 text-4xl font-semibold tracking-normal text-ink">{result.label}</h2>
          </div>
          <span className="rounded-md bg-signal/10 px-3 py-1 text-sm font-medium text-signal">
            {(result.confidence * 100).toFixed(1)}% confidence
          </span>
        </div>

        <div className="grid gap-3 sm:grid-cols-3">
          <div className="rounded-md border border-line p-4">
            <Signal className="mb-3 h-5 w-5 text-signal" />
            <p className="text-xs uppercase text-slate-500">Class index</p>
            <p className="mt-1 text-sm font-semibold">{result.predicted_price_range}</p>
          </div>
          <div className="rounded-md border border-line p-4 sm:col-span-2">
            <Gauge className="mb-3 h-5 w-5 text-amberline" />
            <p className="text-xs uppercase text-slate-500">Model</p>
            <p className="mt-1 text-sm font-semibold">{result.model_name}</p>
          </div>
        </div>

        <div>
          <p className="mb-3 text-sm font-semibold text-graphite">Class probabilities</p>
          <div className="space-y-3">
            {probabilityRows.map(([label, probability]) => (
              <div key={label}>
                <div className="mb-1 flex justify-between text-xs font-medium text-slate-600">
                  <span>{label}</span>
                  <span>{(probability * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-mist">
                  <div className="h-full rounded-full bg-signal" style={{ width: `${probability * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <p className="mb-3 text-sm font-semibold text-graphite">Classification signals</p>
          <div className="space-y-2">
            {result.explanation.map((item) => (
              <p key={item} className="rounded-md bg-mist px-3 py-2 text-sm text-graphite">
                {item}
              </p>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
