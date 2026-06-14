import { Clipboard, Gauge, LineChart, ShieldCheck } from "lucide-react";

import type { DeviceQuote } from "../types/quoteTypes";
import { formatCurrency } from "../utils/deviceQuoteEngine";

type MobileResultCardProps = {
  quote: DeviceQuote | null;
  onSave: () => void;
};

export default function MobileResultCard({ quote, onSave }: MobileResultCardProps) {
  if (!quote) {
    return (
      <section className="rounded-lg border border-dashed border-line bg-white p-6 shadow-panel">
        <div className="flex items-center gap-3 text-graphite">
          <Clipboard className="h-5 w-5 text-signal" />
          <p className="text-sm">Enter a device profile to generate a trade-in offer and resale target.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="rounded-lg border border-line bg-white p-6 shadow-panel">
      <div className="flex flex-col gap-5">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <p className="text-sm font-medium text-slate-500">Recommended Buy Offer</p>
            <h2 className="mt-1 text-4xl font-semibold tracking-normal text-ink">{formatCurrency(quote.buy_offer)}</h2>
            <p className="mt-2 text-sm text-slate-600">{quote.customer_label}</p>
          </div>
          <span className="w-fit rounded-md bg-signal/10 px-3 py-1 text-sm font-medium text-signal">
            {(quote.confidence * 100).toFixed(0)}% confidence
          </span>
        </div>

        <div className="grid gap-3 sm:grid-cols-3">
          <Metric icon={LineChart} label="Resale target" value={formatCurrency(quote.list_price)} />
          <Metric icon={Gauge} label="Expected margin" value={formatCurrency(quote.expected_margin)} />
          <Metric icon={ShieldCheck} label="Risk level" value={quote.risk_level} />
        </div>

        <div className="rounded-md border border-line bg-mist p-4">
          <p className="text-xs font-semibold uppercase text-slate-500">Customer-ready note</p>
          <p className="mt-2 text-sm leading-6 text-graphite">{quote.customer_note}</p>
        </div>

        <div>
          <p className="mb-3 text-sm font-semibold text-graphite">Pricing factors</p>
          <div className="space-y-2">
            {quote.factors.map((item) => (
              <p key={item} className="rounded-md bg-mist px-3 py-2 text-sm text-graphite">
                {item}
              </p>
            ))}
          </div>
        </div>

        <div>
          <p className="mb-3 text-sm font-semibold text-graphite">Model probabilities</p>
          <div className="space-y-3">
            {Object.entries(quote.model_prediction.probabilities).map(([label, probability]) => (
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

        <button
          type="button"
          onClick={onSave}
          className="inline-flex h-11 w-full items-center justify-center gap-2 rounded-md bg-ink px-4 text-sm font-semibold text-white transition hover:bg-graphite"
        >
          <Clipboard className="h-4 w-4" />
          Save quote to history
        </button>
      </div>
    </section>
  );
}
function Metric({ icon: Icon, label, value }: { icon: typeof LineChart; label: string; value: string }) {
  return (
    <div className="rounded-md border border-line p-4">
      <Icon className="mb-3 h-5 w-5 text-signal" />
      <p className="text-xs uppercase text-slate-500">{label}</p>
      <p className="mt-1 text-sm font-semibold">{value}</p>
    </div>
  );
}
