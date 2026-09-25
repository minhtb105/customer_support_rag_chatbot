"use client";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { useEffect, useState } from "react";
import { getNotifications } from "@/lib/api";
import ThemeToggle from "./ThemeToggle";

const baseNav = [
  { href: "/", label: "Tổng quan" },
  { href: "/tracker", label: "Nhật ký ĐH" },
  { href: "/previsit", label: "Tái khám" },
  { href: "/api-playground", label: "WHO-RAG API" },
];

export default function Header() {
  const pathname = usePathname();
  const router = useRouter();
  const { user, loading, logout, isAdmin, isExpert } = useAuth();
  const [unread, setUnread] = useState(0);

  useEffect(() => {
    if (!user) return;
    const fetchNotif = async () => {
      try {
        const data = await getNotifications();
        setUnread(data.unread_count || 0);
      } catch {}
    };
    fetchNotif();
    const id = setInterval(fetchNotif, 10000);
    return () => clearInterval(id);
  }, [user]);

  const nav = [...baseNav];
  if (user) nav.push({ href: "/my/reviews", label: "Lịch sử" });
  if (isExpert || isAdmin) nav.push({ href: "/expert/queue", label: "Duyệt" + (unread ? ` (${unread})` : "") });
  if (isExpert || isAdmin) nav.push({ href: "/expert/patients", label: "Bệnh nhân" });
  if (isAdmin) {
    nav.push({ href: "/admin/tracing", label: "Tracing" });
    nav.push({ href: "/admin/prompts", label: "Prompts" });
    nav.push({ href: "/admin/users", label: "Quản trị" });
  }
  // Giám sát: admin + specialist/doctor (guidelines) + pharmacist (alerts)
  if (user && (isAdmin || ["specialist", "doctor", "pharmacist"].includes(user.role))) {
    nav.push({ href: "/monitor/guidelines", label: "Giám sát" });
  }

  // Grouped by role (calm nav): patient / doctor / admin
  const patientNav = nav.filter((n) => ["/", "/tracker", "/previsit", "/api-playground", "/my/reviews"].includes(n.href));
  const doctorNav = nav.filter((n) => ["/expert/queue", "/expert/patients", "/monitor/guidelines"].includes(n.href));
  const adminNav = nav.filter((n) => n.href.startsWith("/admin"));
  const renderGroup = (items: { href: string; label: string }[]) =>
    items.map((item) => {
      const active = pathname === item.href;
      return (
        <Link key={item.href} href={item.href} className={`rounded-full px-3 py-1.5 text-sm transition min-h-[44px] inline-flex items-center ${active ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900" : "text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"}`}>
          {item.label}
        </Link>
      );
    });

  return (
    <header className="sticky top-0 z-40 border-b bg-white/80 backdrop-blur dark:bg-slate-950/80 dark:border-slate-800">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
        <Link href="/" className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-600 text-white font-bold text-sm">ĐTĐ</div>
          <div>
            <div className="text-sm font-semibold leading-none">Diabetes RAG</div>
            <div className="text-[11px] text-slate-500 dark:text-slate-400">WHO • ADA • BYT</div>
          </div>
        </Link>
        <nav className="flex items-center gap-1" aria-label="Điều hướng chính">
          {renderGroup(patientNav)}
          {doctorNav.length > 0 && <span aria-hidden className="mx-1 h-5 w-px bg-slate-200 dark:bg-slate-700" />}
          {renderGroup(doctorNav)}
          {adminNav.length > 0 && <span aria-hidden className="mx-1 h-5 w-px bg-slate-200 dark:bg-slate-700" />}
          {renderGroup(adminNav)}
        </nav>
        <div className="flex items-center gap-2 text-xs">
          <ThemeToggle />
          {loading ? <span className="text-slate-400">…</span> : user ? (
            <>
              <span className="hidden sm:inline rounded-full bg-slate-100 px-2 py-1 border">{user.username} <span className="text-slate-500">· {user.role}</span></span>
              {unread > 0 && <span className="rounded-full bg-red-500 text-white px-2 py-1">{unread}</span>}
              <button onClick={async () => { await logout(); router.push("/login"); }} className="rounded-full bg-slate-900 text-white px-3 py-1.5 hover:bg-black">Đăng xuất</button>
            </>
          ) : (
            <>
              <Link href="/login" className="rounded-full bg-slate-900 text-white px-3 py-1.5 hover:bg-black">Đăng nhập</Link>
              <Link href="/register" className="rounded-full border px-3 py-1.5 hover:bg-slate-50">Đăng ký</Link>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
