import { useEffect, useState } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import Navbar from "./components/Navbar";
import CarEstimator from "./pages/CarEstimator";
import Dashboard from "./pages/Dashboard";
import Home from "./pages/Home";
import HousingEstimator from "./pages/HousingEstimator";
import MobileEstimator from "./pages/MobileEstimator";
import Settings from "./pages/Settings";
import type { DisplayMode, VisualTheme } from "./types/themeTypes";

export default function App() {
  const [mode, setMode] = useState<DisplayMode>(() => {
    const savedMode = localStorage.getItem("estimator.mode");
    if (savedMode === "light" || savedMode === "dark") {
      return savedMode;
    }
    const savedTheme = localStorage.getItem("estimator.theme");
    if (savedTheme === "light" || savedTheme === "dark") {
      return savedTheme;
    }
    return "dark";
  });
  const [theme, setTheme] = useState<VisualTheme>(() => {
    const savedTheme = localStorage.getItem("estimator.theme");
    if (savedTheme === "terminal" || savedTheme === "graphite" || savedTheme === "atlas") {
      return savedTheme;
    }
    return "terminal";
  });

  useEffect(() => {
    document.documentElement.classList.toggle("dark", mode === "dark");
    document.documentElement.dataset.mode = mode;
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = mode;
    document.documentElement.style.setProperty("--theme-panel-alpha", "0.66");
    document.documentElement.style.setProperty("--theme-blur", "18px");
    localStorage.setItem("estimator.mode", mode);
    localStorage.setItem("estimator.theme", theme);
  }, [mode, theme]);

  function toggleMode() {
    setMode((currentMode) => (currentMode === "dark" ? "light" : "dark"));
  }

  return (
    <div className="min-h-screen bg-mist text-ink transition-colors">
      <Navbar mode={mode} onToggleMode={toggleMode} />
      <main className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/quote" element={<MobileEstimator />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/settings" element={<Settings mode={mode} theme={theme} onChangeMode={setMode} onChangeTheme={setTheme} />} />
          <Route path="/estimators/housing" element={<HousingEstimator />} />
          <Route path="/estimators/mobile" element={<MobileEstimator />} />
          <Route path="/estimators/car" element={<CarEstimator />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
