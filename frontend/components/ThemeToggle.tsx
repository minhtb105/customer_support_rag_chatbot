"use client";
import { useEffect, useState } from "react";
import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";

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
    <Button
      variant="outline"
      size="icon"
      onClick={toggle}
      aria-label={dark ? "Chuyển sang giao diện sáng" : "Chuyển sang giao diện tối"}
      className="rounded-full min-w-11 text-slate-600 dark:text-slate-300 dark:border-slate-700"
    >
      {dark ? <Sun size={16} /> : <Moon size={16} />}
    </Button>
  );
}
