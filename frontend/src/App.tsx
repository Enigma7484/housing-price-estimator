import { useEffect, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import Navbar from "./components/Navbar";
import CarEstimator from "./pages/CarEstimator";
import Dashboard from "./pages/Dashboard";
import Home from "./pages/Home";
import HousingEstimator from "./pages/HousingEstimator";
import MobileEstimator from "./pages/MobileEstimator";

export type AppTheme = "terminal" | "dark" | "light";

export default function App() {
  const [theme, setTheme] = useState<AppTheme>(() => {
    const savedTheme = localStorage.getItem("estimator.theme");
    if (savedTheme === "terminal" || savedTheme === "light" || savedTheme === "dark") {
      return savedTheme;
    }
    return "terminal";
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark" || theme === "terminal");
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme === "light" ? "light" : "dark";
    document.documentElement.style.setProperty("--theme-panel-alpha", "0.66");
    document.documentElement.style.setProperty("--theme-blur", "18px");
    localStorage.setItem("estimator.theme", theme);
  }, [theme]);

  function toggleTheme() {
    setTheme((currentTheme) => {
      if (currentTheme === "terminal") {
        return "light";
      }
      if (currentTheme === "light") {
        return "dark";
      }
      return "terminal";
    });
  }

  return (
    <div className="min-h-screen bg-mist text-ink transition-colors">
      <Navbar theme={theme} onToggleTheme={toggleTheme} />
      <main className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/estimators/housing" element={<HousingEstimator />} />
          <Route path="/estimators/mobile" element={<MobileEstimator />} />
          <Route path="/estimators/car" element={<CarEstimator />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
