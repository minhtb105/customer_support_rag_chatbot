"use client";
import { useEffect, useState } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import { getDoctorPatients } from "@/lib/api";
import StatusBadge from "@/components/StatusBadge";

const badge: Record<string,string> = {
  critical: "bg-red-100 text-red-800 border-red-200",
  trend: "bg-amber-100 text-amber-800 border-amber-200",
  watch: "bg-blue-100 text-blue-800 border-blue-200",
  safe: "bg-slate-100 text-slate-600 border-slate-200",
};

export default function DoctorPatientsPage(){
  const { user, loading, isExpert, isAdmin } = useAuth();
  const [patients,setPatients]=useState<any[]>([]);
  const [q,setQ]=useState("");
  const [err,setErr]=useState("");
  const allowed = !!user && (isExpert || isAdmin);
  useEffect(()=>{
    if(!allowed) return;
    getDoctorPatients().then(d=>setPatients(d.patients || [])).catch((e:any)=>setErr(e.message));
  },[allowed]);
  if(loading) return <div className="rounded-2xl border bg-white p-8 text-center text-sm text-slate-500 dark:bg-slate-900 dark:border-slate-700">Đang tải...</div>;
  if(!user) return <div className="rounded-2xl border bg-white p-8 text-center dark:bg-slate-900 dark:border-slate-700">Cần đăng nhập <Link href="/login" className="underline">Đăng nhập</Link></div>;
  if(!allowed) return <div role="alert" className="rounded-2xl border bg-white p-8 text-center text-sm text-red-700 dark:bg-slate-900 dark:border-slate-700">403 — Chỉ bác sĩ mới xem được hàng đợi bệnh nhân.</div>;
  const filtered = patients.filter(p=>!q || String(p.username || "").toLowerCase().includes(q.toLowerCase()));
  return (
    <div className="space-y-4">
      <div><h1 className="text-xl font-bold">Hàng đợi bệnh nhân (Glucose)</h1><p className="text-sm text-slate-600 dark:text-slate-300">Sắp xếp theo mức ưu tiên: critical → trend → watch → safe.</p></div>
      <div className="rounded-2xl border bg-white p-4 shadow-sm dark:bg-slate-900 dark:border-slate-700">
        <input value={q} onChange={(e)=>setQ(e.target.value)} placeholder="Lọc theo username..." className="w-full rounded-lg border px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-700 min-h-[44px]" />
      </div>
      {err && <div role="alert" className="rounded-lg bg-red-50 border border-red-200 p-3 text-sm text-red-700">{err}</div>}
      <div className="grid gap-3">
        {filtered.map((p:any)=>(
          <Link key={p.user_id} data-testid="queue-row" href={`/expert/patients/${p.user_id}`} className="rounded-2xl border bg-white p-4 shadow-sm hover:border-blue-300 dark:bg-slate-900 dark:border-slate-700 overflow-x-auto">
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <div className="text-sm font-semibold">{p.username}</div>
                <div className="text-[11px] text-slate-500 dark:text-slate-400">Gần nhất: {p.last_value ?? "—"} mg/dL ({p.last_classification || "—"}) · {p.anomaly_count ?? 0} anomaly</div>
              </div>
              <StatusBadge level={p.level} text={p.level} />
            </div>
          </Link>
        ))}
        {!filtered.length && !err && <div className="rounded-2xl border border-dashed bg-slate-50 p-8 text-center text-sm text-slate-600 dark:bg-slate-900 dark:border-slate-700">Chưa có bệnh nhân có log đường huyết.</div>}
      </div>
    </div>
  );
}
