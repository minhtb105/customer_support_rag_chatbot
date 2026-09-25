"use client";
import { useEffect, useState } from "react";
import { Moon, Sun } from "lucide-react";

export function useIsDark(): boolean {
  const [dark, setDark] = useState(false);
  useEffect(() => {
    const read = () => setDark(document.documentElement.classList.contains("dark"));
    read();
    const obs = new MutationObserver(read);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] });
    return () => obs.disconnect();
  }, []);
  return dark;
}

export default function ThemeToggle() {
  const [dark, setDark] = useState(false);
  useEffect(() => {
    try {
      const t = localStorage.getItem("med-theme");
      setDark(t === "dark" || (!t && window.matchMedia("(prefers-color-scheme: dark)").matches));
    } catch {}
  }, []);
  const toggle = () => {
    const next = !dark;
    setDark(next);
    try {
      localStorage.setItem("med-theme", next ? "dark" : "light");
      document.documentElement.classList.toggle("dark", next);
    } catch {}
  };
  return (
    <button
      onClick={toggle}
      aria-label={dark ? "Chuyển sang giao diện sáng" : "Chuyển sang giao diện tối"}
      className="inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-full border px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:border-slate-700 dark:hover:bg-slate-800"
    >
      {dark ? <Sun size={16} /> : <Moon size={16} />}
    </button>
  );
}
