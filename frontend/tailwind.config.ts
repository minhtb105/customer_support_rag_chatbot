import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: "#eff6ff",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
        },
        diabetes: {
          low: "#f59e0b",
          normal: "#10b981",
          elevated: "#f59e0b",
          high: "#ef4444",
          critical: "#991b1b",
        },
      },
    },
  },
  plugins: [],
};
export default config;
