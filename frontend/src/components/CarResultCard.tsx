import { BadgeDollarSign, Car, Gauge } from "lucide-react";

import type { CarPrediction } from "../types/estimatorTypes";

type CarResultCardProps = {
  result: CarPrediction | null;
};

export default function CarResultCard({ result }: CarResultCardProps) {
  if (!result) {
    return (
      <section className="rounded-lg border border-dashed border-line bg-white p-6 shadow-panel">
        <div className="flex items-center gap-3 text-graphite">
          <Car className="h-5 w-5 text-signal" />
          <p className="text-sm">Submit a vehicle profile to estimate resale value.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="rounded-lg border border-line bg-white p-6 shadow-panel">
      <div className="flex flex-col gap-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm font-medium text-slate-500">Estimated Resale Value</p>
            <h2 className="mt-1 text-4xl font-semibold tracking-normal text-ink">{result.formatted_price}</h2>
          </div>
          <span className="rounded-md bg-signal/10 px-3 py-1 text-sm font-medium text-signal">
            {result.confidence} confidence
          </span>
        </div>

        <div className="grid gap-3 sm:grid-cols-3">
          <div className="rounded-md border border-line p-4">
            <BadgeDollarSign className="mb-3 h-5 w-5 text-signal" />
            <p className="text-xs uppercase text-slate-500">Estimated range</p>
            <p className="mt-1 text-sm font-semibold">
              {result.price_range.formatted_low} - {result.price_range.formatted_high}
            </p>
          </div>
          <div className="rounded-md border border-line p-4">
            <Gauge className="mb-3 h-5 w-5 text-amberline" />
            <p className="text-xs uppercase text-slate-500">Model</p>
            <p className="mt-1 text-sm font-semibold">{result.model_name}</p>
          </div>
          <div className="rounded-md border border-line p-4">
            <Car className="mb-3 h-5 w-5 text-graphite" />
            <p className="text-xs uppercase text-slate-500">Market badge</p>
            <p className="mt-1 text-sm font-semibold">{result.value_badge}</p>
          </div>
        </div>

        <div>
          <p className="mb-3 text-sm font-semibold text-graphite">Resale value signals</p>
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
