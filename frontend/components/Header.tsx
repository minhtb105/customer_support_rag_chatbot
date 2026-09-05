"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const nav = [
  { href: "/", label: "Tổng quan", labelEn: "Overview" },
  { href: "/tracker", label: "Nhật ký ĐH", labelEn: "Tracker A" },
  { href: "/previsit", label: "Tái khám", labelEn: "Pre-visit B" },
  { href: "/api-playground", label: "WHO-RAG API", labelEn: "API C" },
];

export default function Header() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b bg-white/80 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
        <Link href="/" className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600 text-white font-bold text-sm">ĐTĐ</div>
          <div>
            <div className="text-sm font-semibold leading-none">Diabetes RAG</div>
            <div className="text-[11px] text-slate-500">WHO • ADA • BYT</div>
          </div>
        </Link>
        <nav className="flex items-center gap-1">
          {nav.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`rounded-full px-3 py-1.5 text-sm transition ${active ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"}`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
        <div className="hidden sm:flex items-center gap-2 text-xs">
          <span className="rounded-full bg-emerald-50 px-2 py-1 text-emerald-700 border border-emerald-200">● FastAPI :8000</span>
        </div>
      </div>
    </header>
  );
}
