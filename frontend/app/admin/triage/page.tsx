"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import { adminListTriageEvents } from "@/lib/api";

export default function TriageMonitorPage() {
  const { user, loading, isAdmin } = useAuth();
  const [events, setEvents] = useState<any[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [specialty, setSpecialty] = useState("");
  const [emg, setEmg] = useState("");
  const [q, setQ] = useState("");
  const [err, setErr] = useState("");
  const limit = 20;

  const load = async () => {
    setErr("");
    try {
      const d = await adminListTriageEvents({
        page, limit,
        specialty: specialty || undefined,
        emergency: emg === "" ? undefined : emg === "true",
        q: q.trim() || undefined,
      });
      setEvents(d.events || []);
      setTotal(d.total ?? 0);
    } catch (e: any) {
      setErr(e.message);
    }
  };
  useEffect(() => { if (user && isAdmin) load(); }, [user, isAdmin, page]);

  if (loading) return <div className="rounded-2xl border bg-white p-8 text-center text-sm text-slate-500">Đang tải...</div>;
  if (!user) return <div className="rounded-2xl border bg-white p-8 text-center">Cần đăng nhập <Link href="/login" className="underline">Đăng nhập</Link></div>;
  if (!isAdmin) return <div className="rounded-2xl border bg-white p-8 text-center text-sm text-red-700">403 — Chỉ admin (lễ tân demo) xem được monitor triage.</div>;

  return (
    <div className="space-y-4">
      <div><h1 className="text-xl font-bold">Triage monitor (lễ tân)</h1><p className="text-sm text-slate-600">Log các ca agent đã xử lý — read-only, highlight ca phức tạp/khẩn cấp.</p></div>
      <div className="flex flex-wrap gap-2 rounded-2xl border bg-white p-4 shadow-sm">
        <select value={specialty} onChange={(e) => setSpecialty(e.target.value)} className="rounded-lg border px-3 py-2 text-sm">
          <option value="">Mọi chuyên khoa</option>
          <option value="Endocrinology_Complication">Biến chứng ĐTĐ</option>
          <option value="Endocrinology_General">Nội tiết chung</option>
          <option value="Nutrition_Lifestyle">Dinh dưỡng &amp; Lối sống</option>
        </select>
        <select value={emg} onChange={(e) => setEmg(e.target.value)} className="rounded-lg border px-3 py-2 text-sm">
          <option value="">Mọi mức</option>
          <option value="true">Khẩn cấp</option>
          <option value="false">Thường</option>
        </select>
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Tìm trong excerpt..." className="rounded-lg border px-3 py-2 text-sm" />
        <button onClick={() => { setPage(1); load(); }} className="rounded-lg bg-slate-900 px-4 py-2 text-sm text-white">Lọc</button>
      </div>
      {err && <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-xs text-red-700">{err}</div>}
      <div className="grid gap-2">
        {events.map((e: any, i: number) => {
          const hot = e.emergency;
          const complex = (e.specialty || "") === "Endocrinology_Complication";
          return (
            <div key={i} className={`rounded-xl border bg-white p-3 text-xs shadow-sm ${hot ? "border-red-500 ring-2 ring-red-300" : complex ? "border-amber-400 bg-amber-50/50" : ""}`}>
              <div className="flex flex-wrap items-center gap-2">
                {hot && <span className="rounded-full bg-red-600 px-2 py-0.5 font-bold text-white">🚨 KHẨN CẤP</span>}
                {complex && <span className="rounded-full border border-amber-400 bg-amber-100 px-2 py-0.5 text-amber-800">Biến chứng ĐTĐ</span>}
                {e.solver_used && <span className="rounded-full bg-slate-100 px-2 py-0.5 text-slate-600">{e.solver_used}</span>}
                <span className="text-slate-500">{e.ts?.slice(0, 19)?.replace("T", " ") || ""}</span>
              </div>
              <div className="mt-1 whitespace-pre-wrap">{e.excerpt || ""}</div>
            </div>
          );
        })}
        {!events.length && !err && <div className="rounded-2xl border border-dashed bg-slate-50 p-8 text-center text-sm text-slate-600">Chưa có event nào.</div>}
      </div>
      <div className="flex items-center gap-3">
        <button disabled={page <= 1} onClick={() => setPage((p) => Math.max(1, p - 1))} className="rounded-lg border bg-white px-4 py-2 text-sm disabled:opacity-50">Trước</button>
        <span className="text-xs text-slate-600">Trang {page} · {total} events</span>
        <button disabled={page * limit >= total} onClick={() => setPage((p) => p + 1)} className="rounded-lg border bg-white px-4 py-2 text-sm disabled:opacity-50">Sau</button>
      </div>
    </div>
  );
}
