"use client";
import React from "react";

// Shared filter-table-pagination shell for admin list pages (tracing/chunks/memory).
// Filters stay per-page (different fields); table + pagination are shared.
export function AdminTable({ columns, children, emptyText }: {
  columns: string[];
  children: React.ReactNode;
  emptyText: string;
}) {
  return (
    <div className="rounded-2xl border bg-white dark:bg-slate-900 dark:border-slate-700 shadow-sm overflow-hidden">
      <div className="overflow-x-auto">
      <table className="w-full text-xs min-w-[640px]">
        <thead className="bg-slate-50 dark:bg-slate-800 border-b text-left">
          <tr>{columns.map((c) => <th key={c} className="p-2 text-slate-600 dark:text-slate-300">{c}</th>)}</tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
      </div>
    </div>
  );
}

export function AdminEmpty({ colSpan, text }: { colSpan: number; text: string }) {
  return <tr><td colSpan={colSpan} className="p-6 text-center text-slate-500 dark:text-slate-400">{text}</td></tr>;
}

export function AdminSkeletonRows({ cols, rows = 3 }: { cols: number; rows?: number }) {
  return <>{Array.from({ length: rows }, (_, i) => (
    <tr key={i} className="border-b"><td colSpan={cols} className="p-2"><div className="h-4 w-full animate-pulse rounded bg-slate-200 dark:bg-slate-700" /></td></tr>
  ))}</>;
}

export function AdminPagination({ page, totalPages, total, label, go }: {
  page: number; totalPages: number; total: number; label: string; go: (p: number) => void;
}) {
  return (
    <div className="flex items-center justify-between text-xs">
      <div className="dark:text-slate-300">Trang {page}/{totalPages || 1} · {total} {label}</div>
      <div className="flex gap-1">
        <button disabled={page <= 1} onClick={() => go(page - 1)} className="rounded-full border px-3 py-1 disabled:opacity-50 min-h-[44px] min-w-[44px]">Trước</button>
        {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => i + 1).map((p) => (
          <button key={p} onClick={() => go(p)} aria-current={p === page ? "page" : undefined} className={`rounded-full px-3 py-1 min-h-[44px] min-w-[44px] ${p === page ? "bg-slate-900 text-white" : "border"}`}>{p}</button>
        ))}
        <button disabled={page >= totalPages} onClick={() => go(page + 1)} className="rounded-full border px-3 py-1 disabled:opacity-50 min-h-[44px] min-w-[44px]">Sau</button>
      </div>
    </div>
  );
}

export function AdminFilterBar({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-4 shadow-sm flex flex-wrap gap-2 items-end">
      {children}
    </div>
  );
}
