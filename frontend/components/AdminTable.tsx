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
      <table className="w-full text-xs">
        <thead className="bg-slate-50 border-b text-left">
          <tr>{columns.map((c) => <th key={c} className="p-2">{c}</th>)}</tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export function AdminEmpty({ colSpan, text }: { colSpan: number; text: string }) {
  return <tr><td colSpan={colSpan} className="p-6 text-center text-slate-500">{text}</td></tr>;
}

export function AdminPagination({ page, totalPages, total, label, go }: {
  page: number; totalPages: number; total: number; label: string; go: (p: number) => void;
}) {
  return (
    <div className="flex items-center justify-between text-xs">
      <div>Trang {page}/{totalPages || 1} · {total} {label}</div>
      <div className="flex gap-1">
        <button disabled={page <= 1} onClick={() => go(page - 1)} className="rounded-full border px-3 py-1 disabled:opacity-50">Trước</button>
        {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => i + 1).map((p) => (
          <button key={p} onClick={() => go(p)} className={`rounded-full px-3 py-1 ${p === page ? "bg-slate-900 text-white" : "border"}`}>{p}</button>
        ))}
        <button disabled={page >= totalPages} onClick={() => go(page + 1)} className="rounded-full border px-3 py-1 disabled:opacity-50">Sau</button>
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
