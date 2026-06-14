import { BarChart3, Building2, Car, ClipboardList, LayoutDashboard, Moon, Settings, Smartphone, Sun } from "lucide-react";
import { NavLink } from "react-router-dom";
import type { DisplayMode } from "../types/themeTypes";

const navItems = [
  { label: "Quote", href: "/quote", icon: ClipboardList },
  { label: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  { label: "Devices", href: "/estimators/mobile", icon: Smartphone },
  { label: "Auto", href: "/estimators/car", icon: Car },
  { label: "Property", href: "/estimators/housing", icon: Building2 },
  { label: "Settings", href: "/settings", icon: Settings },
];

type NavbarProps = {
  mode: DisplayMode;
  onToggleMode: () => void;
};

export default function Navbar({ mode, onToggleMode }: NavbarProps) {
  const ModeIcon = mode === "dark" ? Sun : Moon;
  const nextMode = mode === "dark" ? "day" : "night";

  return (
    <header className="border-b border-line bg-white">
      <nav className="mx-auto flex w-full max-w-7xl flex-col gap-4 px-4 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-8">
        <NavLink to="/quote" className="flex items-center gap-3">
          <span className="grid h-10 w-10 place-items-center rounded-md bg-ink text-white">
            <BarChart3 className="h-5 w-5" />
          </span>
          <div>
            <p className="text-base font-semibold">ResaleIQ</p>
            <p className="text-xs text-slate-500">Trade-in quote copilot</p>
          </div>
        </NavLink>
        <div className="flex flex-wrap items-center gap-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.href}
                to={item.href}
                className={({ isActive }) =>
                  `inline-flex h-10 items-center gap-2 rounded-md px-3 text-sm font-medium transition ${
                    isActive ? "bg-ink text-white" : "text-graphite hover:bg-mist"
                  }`
                }
              >
                <Icon className="h-4 w-4" />
                {item.label}
              </NavLink>
            );
          })}
          <button
            type="button"
            onClick={onToggleMode}
            aria-label={`Switch to ${nextMode} mode`}
            title={`Switch to ${nextMode} mode`}
            className="inline-flex h-10 w-10 items-center justify-center rounded-md border border-line bg-white text-graphite transition hover:border-signal hover:text-signal"
          >
            <ModeIcon className="h-4 w-4" />
          </button>
        </div>
      </nav>
    </header>
  );
}
