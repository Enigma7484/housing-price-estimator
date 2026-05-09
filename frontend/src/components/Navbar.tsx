import { BarChart3, Building2, Car, Home, LayoutDashboard, Moon, Smartphone, Sun } from "lucide-react";
import { NavLink } from "react-router-dom";

const navItems = [
  { label: "Home", href: "/", icon: Home },
  { label: "Housing", href: "/estimators/housing", icon: Building2 },
  { label: "Mobile", href: "/estimators/mobile", icon: Smartphone },
  { label: "Car", href: "/estimators/car", icon: Car },
  { label: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
];

type NavbarProps = {
  theme: "light" | "dark";
  onToggleTheme: () => void;
};

export default function Navbar({ theme, onToggleTheme }: NavbarProps) {
  const ThemeIcon = theme === "dark" ? Sun : Moon;

  return (
    <header className="border-b border-line bg-white">
      <nav className="mx-auto flex w-full max-w-7xl flex-col gap-4 px-4 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-8">
        <NavLink to="/" className="flex items-center gap-3">
          <span className="grid h-10 w-10 place-items-center rounded-md bg-ink text-white">
            <BarChart3 className="h-5 w-5" />
          </span>
          <div>
            <p className="text-base font-semibold">AI Estimator Platform</p>
            <p className="text-xs text-slate-500">Applied ML prediction workflows</p>
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
            onClick={onToggleTheme}
            aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            className="inline-flex h-10 w-10 items-center justify-center rounded-md border border-line bg-white text-graphite transition hover:border-signal hover:text-signal"
          >
            <ThemeIcon className="h-4 w-4" />
          </button>
        </div>
      </nav>
    </header>
  );
}
