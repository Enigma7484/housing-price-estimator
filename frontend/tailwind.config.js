/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      colors: {
        ink: "#15171c",
        mist: "#f5f7fb",
        line: "#d9dee8",
        signal: "#0f766e",
        graphite: "#2f3542",
        amberline: "#b7791f",
      },
      boxShadow: {
        panel: "0 18px 50px rgba(22, 31, 46, 0.08)",
      },
    },
  },
  plugins: [],
};
