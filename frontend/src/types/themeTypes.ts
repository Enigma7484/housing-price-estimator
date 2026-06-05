export type DisplayMode = "light" | "dark";

export type VisualTheme = "terminal" | "graphite" | "atlas";

export type ThemeOption = {
  id: VisualTheme;
  name: string;
  description: string;
  accent: string;
};

export const themeOptions: ThemeOption[] = [
  {
    id: "terminal",
    name: "Terminal",
    description: "BillsAgent-inspired command center with scanlines, glass panels, and green model-status accents.",
    accent: "#22c55e",
  },
  {
    id: "graphite",
    name: "Graphite",
    description: "Quiet executive dashboard styling for serious portfolio and demo conversations.",
    accent: "#14b8a6",
  },
  {
    id: "atlas",
    name: "Atlas",
    description: "Sharper product analytics theme with blue/teal emphasis and crisp valuation surfaces.",
    accent: "#2563eb",
  },
];
