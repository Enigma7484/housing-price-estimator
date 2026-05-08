import { ArrowRight, Lock } from "lucide-react";
import { Link } from "react-router-dom";

type EstimatorCardProps = {
  title: string;
  description: string;
  href?: string;
  status: "Live" | "Coming soon";
  meta?: string;
};

export default function EstimatorCard({ title, description, href, status, meta }: EstimatorCardProps) {
  const content = (
    <div className="flex h-full flex-col justify-between rounded-lg border border-line bg-white p-5 shadow-panel transition hover:-translate-y-0.5 hover:border-signal/50">
      <div>
        <div className="mb-4 flex items-center justify-between">
          <span className="rounded-md bg-mist px-2.5 py-1 text-xs font-semibold text-graphite">{status}</span>
          {status === "Live" ? <ArrowRight className="h-4 w-4 text-signal" /> : <Lock className="h-4 w-4 text-slate-400" />}
        </div>
        <h3 className="text-lg font-semibold">{title}</h3>
        <p className="mt-2 text-sm leading-6 text-slate-600">{description}</p>
        {meta ? <p className="mt-4 text-xs font-semibold uppercase text-slate-500">{meta}</p> : null}
      </div>
    </div>
  );

  return href ? <Link to={href}>{content}</Link> : content;
}
