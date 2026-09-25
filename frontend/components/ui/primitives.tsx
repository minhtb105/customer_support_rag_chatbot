"use client";
import React from "react";
import { cn } from "./cn";

// Frozen minimal API (D3): Button variant/size, Alert tone + role=alert, Card, Input, EmptyState.
// No clsx/cva/tailwind-merge — internal cn() only. No per-page variants.

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "dark" | "outline" | "danger";
  size?: "sm" | "md";
};
export function Button({ variant = "primary", size = "md", className, ...rest }: ButtonProps) {
  const v =
    variant === "primary"
      ? "bg-blue-600 text-white hover:bg-blue-700"
      : variant === "dark"
        ? "bg-slate-900 text-white hover:bg-black dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-white"
        : variant === "danger"
          ? "bg-red-600 text-white hover:bg-red-700"
          : "border bg-white hover:bg-slate-50 dark:bg-slate-900 dark:border-slate-700 dark:hover:bg-slate-800";
  const s = size === "sm" ? "px-3 py-1.5 text-sm min-h-[36px]" : "px-5 py-2.5 text-sm min-h-[44px]";
  return <button className={cn("rounded-lg font-medium transition focus-visible:outline-2 disabled:opacity-50", v, s, className)} {...rest} />;
}

export function Card({ className, ...rest }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-2xl border bg-white p-5 shadow-sm dark:bg-slate-900 dark:border-slate-700",
        className
      )}
      {...rest}
    />
  );
}

type AlertProps = React.HTMLAttributes<HTMLDivElement> & { tone?: "critical" | "warning" | "info" | "success" };
export function Alert({ tone = "info", className, children, ...rest }: AlertProps) {
  const t =
    tone === "critical"
      ? "border-red-600 bg-red-600 text-white dark:bg-red-700"
      : tone === "warning"
        ? "border-amber-200 bg-amber-50 text-amber-800 dark:bg-amber-950 dark:border-amber-800 dark:text-amber-200"
        : tone === "success"
          ? "border-emerald-200 bg-emerald-50 text-emerald-800 dark:bg-emerald-950 dark:border-emerald-800 dark:text-emerald-200"
          : "border-blue-100 bg-blue-50 text-blue-800 dark:bg-blue-950 dark:border-blue-800 dark:text-blue-200";
  return (
    <div role="alert" className={cn("rounded-xl border p-3 text-sm leading-relaxed", t, className)} {...rest}>
      {children}
    </div>
  );
}

export function Input(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input
      {...props}
      className={cn(
        "w-full rounded-lg border px-3 py-2 text-sm bg-white dark:bg-slate-900 dark:border-slate-700 min-h-[44px]",
        props.className
      )}
    />
  );
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="rounded-2xl border border-dashed bg-slate-50 p-8 text-center text-sm text-slate-600 dark:bg-slate-900 dark:border-slate-700 dark:text-slate-300">
      <div className="font-medium">{title}</div>
      {hint && <div className="mt-1 text-xs opacity-80">{hint}</div>}
    </div>
  );
}
