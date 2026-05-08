import { useEffect, useState } from "react";
import { ArrowRight, Building2, Car, Laptop, Shield, Smartphone, WalletCards } from "lucide-react";
import { Link } from "react-router-dom";

import { getEstimatorCatalog } from "../api/estimatorApi";
import EstimatorCard from "../components/EstimatorCard";
import type { EstimatorCatalogItem } from "../types/estimatorTypes";

export default function Home() {
  const [estimators, setEstimators] = useState<EstimatorCatalogItem[]>([]);

  useEffect(() => {
    getEstimatorCatalog()
      .then((response) => setEstimators(response.estimators))
      .catch(() => {
        setEstimators([
          {
            key: "housing",
            name: "Housing Price Estimator",
            category: "Real Estate",
            problem_type: "Regression",
            route: "/api/housing/predict",
            frontend_path: "/estimators/housing",
            phase: "Phase 1",
            description: "Regression model trained on real housing sales data with a dedicated FastAPI prediction route.",
            status: "ready",
            metadata: null,
          },
          {
            key: "mobile",
            name: "Mobile Price Range Estimator",
            category: "Consumer Devices",
            problem_type: "Classification",
            route: "/api/mobile/predict",
            frontend_path: "/estimators/mobile",
            phase: "Phase 2",
            description: "Classification model for device market tier using hardware, display, camera, and connectivity specs.",
            status: "ready",
            metadata: null,
          },
        ]);
      });
  }, []);

  const liveEstimators = estimators.filter((item) => item.status === "ready");
  const plannedEstimators = estimators.filter((item) => item.status !== "ready");

  return (
    <div className="space-y-10">
      <section className="grid gap-8 rounded-lg border border-line bg-white p-6 shadow-panel lg:grid-cols-[1.1fr_0.9fr] lg:p-10">
        <div className="flex flex-col justify-center">
          <span className="mb-5 w-fit rounded-md bg-signal/10 px-3 py-1 text-sm font-semibold text-signal">
            Full-stack ML estimator platform
          </span>
          <h1 className="max-w-3xl text-4xl font-semibold tracking-normal text-ink sm:text-5xl">
            AI Estimator Platform
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-600">
            A modular prediction product for real-world valuation workflows, starting with a trained housing price
            estimator and designed to expand into device, vehicle, compensation, rent, and insurance models.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <Link
              to="/estimators/housing"
              className="inline-flex h-11 items-center gap-2 rounded-md bg-ink px-4 text-sm font-semibold text-white transition hover:bg-graphite"
            >
              Open housing estimator
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              to="/dashboard"
              className="inline-flex h-11 items-center gap-2 rounded-md border border-line bg-white px-4 text-sm font-semibold text-graphite transition hover:border-signal"
            >
              View model dashboard
            </Link>
          </div>
        </div>
        <div className="grid content-center gap-4">
          {[
            ["Live model", "Housing Price Estimator", Building2],
            ["Architecture", "FastAPI + React + scikit-learn", Shield],
            ["Roadmap", "Multiple estimators in one platform", WalletCards],
          ].map(([eyebrow, title, Icon]) => (
            <div key={title as string} className="rounded-lg border border-line bg-mist p-5">
              <div className="mb-4 flex items-center gap-3">
                <span className="grid h-10 w-10 place-items-center rounded-md bg-white">
                  <Icon className="h-5 w-5 text-signal" />
                </span>
                <span className="text-xs font-semibold uppercase text-slate-500">{eyebrow as string}</span>
              </div>
              <p className="text-lg font-semibold">{title as string}</p>
            </div>
          ))}
        </div>
      </section>

      <section>
        <div className="mb-5 flex items-end justify-between gap-4">
          <div>
            <h2 className="text-2xl font-semibold">Estimator Catalog</h2>
            <p className="mt-2 text-sm text-slate-600">Live modules and planned expansion paths.</p>
          </div>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {liveEstimators.map((item) => (
            <EstimatorCard
              key={item.key}
              title={item.name}
              description={item.description}
              href={item.frontend_path ?? undefined}
              status="Live"
              meta={`${item.problem_type} · ${item.phase}`}
            />
          ))}
          {plannedEstimators.map((item, index) => {
            const Icon = index % 2 === 0 ? Smartphone : index % 3 === 0 ? Car : Laptop;
            return (
              <div key={item.key} className="relative">
                <EstimatorCard
                  title={item.name}
                  description={item.description}
                  status="Coming soon"
                  meta={`${item.problem_type} · ${item.category}`}
                />
                <Icon className="pointer-events-none absolute bottom-5 right-5 h-5 w-5 text-slate-300" />
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
