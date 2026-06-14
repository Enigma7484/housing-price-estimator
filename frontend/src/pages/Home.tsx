import { ArrowRight, ClipboardCheck, Gauge, History, ShieldCheck, Smartphone } from "lucide-react";
import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div className="space-y-8">
      <section className="grid gap-8 rounded-lg border border-line bg-white p-6 shadow-panel lg:grid-cols-[1.05fr_0.95fr] lg:p-10">
        <div className="flex flex-col justify-center">
          <span className="mb-5 w-fit rounded-md bg-signal/10 px-3 py-1 text-sm font-semibold text-signal">
            Resale quote copilot
          </span>
          <h1 className="max-w-3xl text-4xl font-semibold tracking-normal text-ink sm:text-5xl">
            ResaleIQ
          </h1>
          <p className="mt-5 max-w-2xl text-lg leading-8 text-slate-600">
            Generate consistent trade-in offers, resale targets, margin checks, and customer-ready quote notes from a single device workflow.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <Link
              to="/quote"
              className="inline-flex h-11 items-center gap-2 rounded-md bg-ink px-4 text-sm font-semibold text-white transition hover:bg-graphite"
            >
              Start quote
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              to="/dashboard"
              className="inline-flex h-11 items-center gap-2 rounded-md border border-line bg-white px-4 text-sm font-semibold text-graphite transition hover:border-signal"
            >
              Owner dashboard
            </Link>
          </div>
        </div>
        <div className="grid content-center gap-4">
          {[
            ["Offer", "Recommended buy price with confidence", ClipboardCheck],
            ["Margin", "Resale target and gross margin guardrail", Gauge],
            ["History", "Saved quotes for owner review", History],
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

      <section className="grid gap-4 md:grid-cols-3">
        <WorkflowCard
          icon={Smartphone}
          title="Device Intake"
          copy="Capture model, storage, condition, battery health, carrier status, accessories, and repair risk."
        />
        <WorkflowCard
          icon={ShieldCheck}
          title="Offer Guardrails"
          copy="Translate the ML tier into buy offer, resale target, confidence, and risk level."
        />
        <WorkflowCard
          icon={History}
          title="Owner Review"
          copy="Save quotes locally so owners can inspect margin, overrides, and high-risk pricing behavior."
        />
      </section>
    </div>
  );
}
function WorkflowCard({ icon: Icon, title, copy }: { icon: typeof Smartphone; title: string; copy: string }) {
  return (
    <div className="rounded-lg border border-line bg-white p-5 shadow-panel">
      <Icon className="mb-4 h-5 w-5 text-signal" />
      <h2 className="text-xl font-semibold text-ink">{title}</h2>
      <p className="mt-2 text-sm leading-6 text-slate-600">{copy}</p>
    </div>
  );
}
