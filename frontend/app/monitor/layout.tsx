"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

export default function MonitorLayout({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  const tabs = [
    { href: "/monitor/guidelines", label: "Guidelines" },
    { href: "/monitor/alerts", label: "Cảnh báo thuốc" },
  ];
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold">Giám sát — Guidelines & An toàn thuốc</h1>
        <p className="text-sm text-slate-600">
          Agent tìm → tóm tắt VI → gắn cờ pending_review → chuyên gia duyệt → mới vào corpus (kèm version/ngày). Xem{" "}
          <Link href="/docs/monitoring" className="underline">
            docs/monitoring.md
          </Link>
        </p>
      </div>
      <div className="flex gap-2">
        {tabs.map((t) => (
          <Link
            key={t.href}
            href={t.href}
            className={`rounded-full px-4 py-2 text-xs font-medium ${path === t.href ? "bg-slate-900 text-white" : "border bg-white hover:bg-slate-50"}`}
          >
            {t.label}
          </Link>
        ))}
      </div>
      {children}
    </div>
  );
}
