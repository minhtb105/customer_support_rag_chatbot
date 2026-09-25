"use client";
import { AlertTriangle, TrendingUp, Eye, CheckCircle2 } from "lucide-react";
import { cn } from "./ui/cn";

// Non-color cue (D6): always icon + text label, never color-only.
const MAP: Record<string, { icon: any; cls: string; label: string }> = {
  critical: { icon: AlertTriangle, cls: "bg-red-100 text-red-800 border-red-200 dark:bg-red-950 dark:text-red-200 dark:border-red-800", label: "critical" },
  low: { icon: AlertTriangle, cls: "bg-orange-100 text-orange-800 border-orange-200 dark:bg-orange-950 dark:text-orange-200 dark:border-orange-800", label: "low" },
  trend: { icon: TrendingUp, cls: "bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-950 dark:text-amber-200 dark:border-amber-800", label: "trend" },
  watch: { icon: Eye, cls: "bg-blue-100 text-blue-800 border-blue-200 dark:bg-blue-950 dark:text-blue-200 dark:border-blue-800", label: "watch" },
  high: { icon: TrendingUp, cls: "bg-red-100 text-red-800 border-red-200 dark:bg-red-950 dark:text-red-200 dark:border-red-800", label: "high" },
  elevated: { icon: Eye, cls: "bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-950 dark:text-amber-200 dark:border-amber-800", label: "elevated" },
  normal: { icon: CheckCircle2, cls: "bg-emerald-100 text-emerald-800 border-emerald-200 dark:bg-emerald-950 dark:text-emerald-200 dark:border-emerald-800", label: "normal" },
  safe: { icon: CheckCircle2, cls: "bg-slate-100 text-slate-600 border-slate-200 dark:bg-slate-800 dark:text-slate-300 dark:border-slate-700", label: "safe" },
};

export default function StatusBadge({ level, text }: { level: string; text?: string }) {
  const m = MAP[level] || MAP.safe;
  const Icon = m.icon;
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full border px-2 py-1 text-xs font-medium", m.cls)}>
      <Icon size={13} aria-hidden />
      {text ?? m.label}
    </span>
  );
}
