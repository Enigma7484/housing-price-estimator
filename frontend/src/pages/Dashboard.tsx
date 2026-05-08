import { useEffect, useMemo, useState } from "react";
import { Activity, Database, LineChart, Server } from "lucide-react";

import { getEstimatorCatalog } from "../api/estimatorApi";
import ErrorMessage from "../components/ErrorMessage";
import type { EstimatorCatalogItem } from "../types/estimatorTypes";

export default function Dashboard() {
  const [estimators, setEstimators] = useState<EstimatorCatalogItem[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    getEstimatorCatalog()
      .then((response) => setEstimators(response.estimators))
      .catch(() => setError("Estimator catalog is unavailable. Start FastAPI and confirm models are trained."));
  }, []);

  const liveEstimators = estimators.filter((item) => item.status === "ready");
  const totalRows = liveEstimators.reduce((total, item) => total + (item.metadata?.dataset_rows ?? 0), 0);
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
        <h1 className="text-3xl font-semibold">Model Dashboard</h1>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
          Registry-driven operational summary for available estimators, model artifacts, dataset coverage, and roadmap modules.
        </p>
      </section>

      {error ? <ErrorMessage message={error} /> : null}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard icon={Server} label="Live estimators" value={`${liveEstimators.length}/${estimators.length || "-"}`} />
        <MetricCard icon={LineChart} label="Model families" value={uniqueCount(liveEstimators.map((item) => item.problem_type)).toString()} />
        <MetricCard icon={Database} label="Training rows" value={totalRows ? totalRows.toLocaleString() : "-"} />
        <MetricCard icon={Activity} label="Best classifier" value={bestClassification ? bestClassification.toFixed(3) : "-"} />
      </div>

      <section className="rounded-lg border border-line bg-white p-6 shadow-panel">
        <div className="mb-5 flex items-center justify-between">
          <h2 className="text-xl font-semibold">Estimator Registry</h2>
          <span className="rounded-md bg-signal/10 px-3 py-1 text-sm font-semibold text-signal">Phase 3</span>
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
                <th className="py-3 pr-4">Route</th>
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
                    <code className="rounded-md bg-mist px-2 py-1 text-xs">{item.route ?? "TBD"}</code>
                  </td>
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
