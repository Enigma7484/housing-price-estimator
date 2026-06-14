import { useEffect, useMemo, useState } from "react";
import { Activity, AlertTriangle, Database, DollarSign, LineChart, Server } from "lucide-react";

import { getEstimatorCatalog } from "../api/estimatorApi";
import ErrorMessage from "../components/ErrorMessage";
import type { EstimatorCatalogItem } from "../types/estimatorTypes";
import type { DeviceQuote } from "../types/quoteTypes";
import { formatCurrency, getStoredQuotes } from "../utils/deviceQuoteEngine";

export default function Dashboard() {
  const [estimators, setEstimators] = useState<EstimatorCatalogItem[]>([]);
  const [quotes, setQuotes] = useState<DeviceQuote[]>(() => getStoredQuotes());
  const [error, setError] = useState("");

  useEffect(() => {
    getEstimatorCatalog()
      .then((response) => setEstimators(response.estimators))
      .catch(() => setError("Estimator catalog is unavailable. Start FastAPI and confirm models are trained."));
  }, []);

  useEffect(() => {
    function refreshQuotes() {
      setQuotes(getStoredQuotes());
    }
    window.addEventListener("storage", refreshQuotes);
    window.addEventListener("resaleiq:quotes-updated", refreshQuotes);
    return () => {
      window.removeEventListener("storage", refreshQuotes);
      window.removeEventListener("resaleiq:quotes-updated", refreshQuotes);
    };
  }, []);

  const liveEstimators = estimators.filter((item) => item.status === "ready");
  const totalRows = liveEstimators.reduce((total, item) => total + (item.metadata?.dataset_rows ?? 0), 0);
  const averageMargin = quotes.length ? quotes.reduce((total, quote) => total + quote.margin_rate, 0) / quotes.length : 0;
  const totalOfferValue = quotes.reduce((total, quote) => total + quote.buy_offer, 0);
  const highRiskQuotes = quotes.filter((quote) => quote.risk_level === "High").length;
  const bestClassification = useMemo(
    () =>
      liveEstimators
        .filter((item) => item.problem_type === "Classification")
        .map((item) => item.metadata?.metrics.f1_macro ?? item.metadata?.metrics.accuracy ?? 0)
        .sort((a, b) => b - a)[0],
    [liveEstimators],
  );

  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-line bg-white p-6 shadow-panel">
        <h1 className="text-3xl font-semibold">Owner Dashboard</h1>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
          Track quote volume, margin quality, risk, and the model systems powering the resale workflow.
        </p>
      </section>

      {error ? <ErrorMessage message={error} /> : null}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard icon={DollarSign} label="Open offer value" value={formatCurrency(totalOfferValue)} />
        <MetricCard icon={LineChart} label="Average margin" value={quotes.length ? `${Math.round(averageMargin * 100)}%` : "-"} />
        <MetricCard icon={AlertTriangle} label="High-risk quotes" value={highRiskQuotes.toString()} />
        <MetricCard icon={Server} label="Live models" value={`${liveEstimators.length}/${estimators.length || "-"}`} />
      </div>

      <section className="rounded-lg border border-line bg-white p-6 shadow-panel">
        <div className="mb-5 flex items-center justify-between gap-4">
          <div>
            <h2 className="text-xl font-semibold">Quote History</h2>
            <p className="mt-1 text-sm text-slate-500">Saved quote decisions from the trade-in desk.</p>
          </div>
          <span className="rounded-md bg-signal/10 px-3 py-1 text-sm font-semibold text-signal">{quotes.length} saved</span>
        </div>
        {quotes.length ? (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] text-left text-sm">
              <thead className="border-b border-line text-xs uppercase text-slate-500">
                <tr>
                  <th className="py-3 pr-4">Device</th>
                  <th className="py-3 pr-4">Offer</th>
                  <th className="py-3 pr-4">Resale target</th>
                  <th className="py-3 pr-4">Margin</th>
                  <th className="py-3 pr-4">Risk</th>
                  <th className="py-3 pr-4">Confidence</th>
                  <th className="py-3 pr-4">Created</th>
                </tr>
              </thead>
              <tbody>
                {quotes.map((quote) => (
                  <tr key={quote.id} className="border-b border-line">
                    <td className="py-4 pr-4">
                      <p className="font-semibold">{quote.device_model}</p>
                      <p className="mt-1 text-xs capitalize text-slate-500">{quote.condition} · {quote.model_tier}</p>
                    </td>
                    <td className="py-4 pr-4">{formatCurrency(quote.buy_offer)}</td>
                    <td className="py-4 pr-4">{formatCurrency(quote.list_price)}</td>
                    <td className="py-4 pr-4">{Math.round(quote.margin_rate * 100)}%</td>
                    <td className="py-4 pr-4">
                      <span className={`rounded-md px-2.5 py-1 text-xs font-semibold ${riskClass(quote.risk_level)}`}>
                        {quote.risk_level}
                      </span>
                    </td>
                    <td className="py-4 pr-4">{Math.round(quote.confidence * 100)}%</td>
                    <td className="py-4 pr-4">{new Date(quote.created_at).toLocaleDateString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="rounded-md border border-dashed border-line bg-mist p-6 text-sm text-slate-600">
            No saved quotes yet. Generate a quote and save it to start building owner history.
          </div>
        )}
      </section>

      <section className="rounded-lg border border-line bg-white p-6 shadow-panel">
        <div className="mb-5 flex items-center justify-between">
          <h2 className="text-xl font-semibold">Model Registry</h2>
          <span className="rounded-md bg-signal/10 px-3 py-1 text-sm font-semibold text-signal">Operations</span>
        </div>
        <div className="mb-4 grid gap-4 md:grid-cols-3">
          <MetricCard icon={Database} label="Training rows" value={totalRows ? totalRows.toLocaleString() : "-"} />
          <MetricCard icon={Activity} label="Best classifier" value={bestClassification ? bestClassification.toFixed(3) : "-"} />
          <MetricCard icon={Server} label="Model families" value={uniqueCount(liveEstimators.map((item) => item.problem_type)).toString()} />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[860px] text-left text-sm">
            <thead className="border-b border-line text-xs uppercase text-slate-500">
              <tr>
                <th className="py-3 pr-4">Estimator</th>
                <th className="py-3 pr-4">Category</th>
                <th className="py-3 pr-4">Type</th>
                <th className="py-3 pr-4">Model</th>
                <th className="py-3 pr-4">Primary metric</th>
                <th className="py-3 pr-4">Rows</th>
                <th className="py-3 pr-4">Status</th>
              </tr>
            </thead>
            <tbody>
              {estimators.map((item) => (
                <tr key={item.key} className="border-b border-line">
                  <td className="py-4 pr-4">
                    <p className="font-semibold">{item.name}</p>
                    <p className="mt-1 text-xs text-slate-500">{item.phase}</p>
                  </td>
                  <td className="py-4 pr-4">{item.category}</td>
                  <td className="py-4 pr-4">{item.problem_type}</td>
                  <td className="py-4 pr-4">{item.metadata?.model_name ?? "Planned"}</td>
                  <td className="py-4 pr-4">{formatPrimaryMetric(item)}</td>
                  <td className="py-4 pr-4">{item.metadata?.dataset_rows ? item.metadata.dataset_rows.toLocaleString() : "-"}</td>
                  <td className="py-4 pr-4">
                    <span className={`rounded-md px-2.5 py-1 text-xs font-semibold ${item.status === "ready" ? "bg-signal/10 text-signal" : "bg-slate-100 text-slate-500"}`}>
                      {item.status === "ready" ? "Ready" : item.status === "planned" ? "Roadmap" : "Needs artifact"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function MetricCard({ icon: Icon, label, value }: { icon: typeof Server; label: string; value: string }) {
  return (
    <div className="rounded-lg border border-line bg-white p-5 shadow-panel">
      <Icon className="mb-4 h-5 w-5 text-signal" />
      <p className="text-xs font-semibold uppercase text-slate-500">{label}</p>
      <p className="mt-2 break-words text-lg font-semibold">{value}</p>
    </div>
  );
}

function uniqueCount(values: string[]) {
  return new Set(values).size;
}

function riskClass(riskLevel: DeviceQuote["risk_level"]) {
  if (riskLevel === "Low") {
    return "bg-signal/10 text-signal";
  }
  if (riskLevel === "Medium") {
    return "bg-amber-100 text-amber-700";
  }
  return "bg-red-100 text-red-700";
}

function formatPrimaryMetric(item: EstimatorCatalogItem) {
  const metrics = item.metadata?.metrics;
  if (!metrics) {
    return "-";
  }
  if (typeof metrics.r2 === "number") {
    return `R2 ${metrics.r2.toFixed(3)}`;
  }
  if (typeof metrics.f1_macro === "number") {
    return `F1 ${metrics.f1_macro.toFixed(3)}`;
  }
  if (typeof metrics.accuracy === "number") {
    return `Acc ${metrics.accuracy.toFixed(3)}`;
  }
  return "-";
}
