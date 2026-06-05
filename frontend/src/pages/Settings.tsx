import { Check, Moon, Palette, Sun } from "lucide-react";

import type { DisplayMode, VisualTheme } from "../types/themeTypes";
import { themeOptions } from "../types/themeTypes";

type SettingsProps = {
  mode: DisplayMode;
  theme: VisualTheme;
  onChangeMode: (mode: DisplayMode) => void;
  onChangeTheme: (theme: VisualTheme) => void;
};

export default function Settings({ mode, theme, onChangeMode, onChangeTheme }: SettingsProps) {
  return (
    <div className="space-y-6">
      <section className="rounded-lg border border-line bg-white p-6 shadow-panel">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <span className="mb-4 inline-flex w-fit items-center gap-2 rounded-md bg-signal/10 px-3 py-1 text-sm font-semibold text-signal">
              <Palette className="h-4 w-4" />
              Interface preferences
            </span>
            <h1 className="text-3xl font-semibold text-ink">Settings</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
              Choose the product theme separately from day or night mode. The selected visual system is applied across
              estimators, dashboards, forms, and result cards.
            </p>
          </div>
          <div className="flex rounded-lg border border-line bg-mist p-1">
            <button
              type="button"
              onClick={() => onChangeMode("light")}
              className={`inline-flex h-10 items-center gap-2 rounded-md px-4 text-sm font-semibold transition ${
                mode === "light" ? "bg-white text-ink shadow-panel" : "text-slate-500 hover:bg-white"
              }`}
            >
              <Sun className="h-4 w-4" />
              Day
            </button>
            <button
              type="button"
              onClick={() => onChangeMode("dark")}
              className={`inline-flex h-10 items-center gap-2 rounded-md px-4 text-sm font-semibold transition ${
                mode === "dark" ? "bg-ink text-white shadow-panel" : "text-slate-500 hover:bg-white"
              }`}
            >
              <Moon className="h-4 w-4" />
              Night
            </button>
          </div>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-3">
        {themeOptions.map((option) => {
          const isActive = theme === option.id;
          return (
            <button
              key={option.id}
              type="button"
              onClick={() => onChangeTheme(option.id)}
              className={`group rounded-lg border bg-white p-5 text-left shadow-panel transition hover:-translate-y-0.5 hover:border-signal ${
                isActive ? "border-signal" : "border-line"
              }`}
            >
              <div className="mb-5 flex items-center justify-between gap-4">
                <span className="grid h-11 w-11 place-items-center rounded-md border border-line bg-mist">
                  <span className="h-5 w-5 rounded-full" style={{ backgroundColor: option.accent }} />
                </span>
                {isActive ? (
                  <span className="inline-flex items-center gap-1 rounded-md bg-signal/10 px-2.5 py-1 text-xs font-semibold text-signal">
                    <Check className="h-3.5 w-3.5" />
                    Active
                  </span>
                ) : null}
              </div>
              <h2 className="text-xl font-semibold text-ink">{option.name}</h2>
              <p className="mt-2 min-h-16 text-sm leading-6 text-slate-600">{option.description}</p>
              <div className="mt-5 grid grid-cols-3 gap-2">
                <span className="h-10 rounded-md border border-line bg-mist" />
                <span className="h-10 rounded-md border border-line bg-white" />
                <span className="h-10 rounded-md" style={{ backgroundColor: option.accent }} />
              </div>
            </button>
          );
        })}
      </section>
    </div>
  );
}
