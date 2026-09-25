"use client";
import { useEffect, useState } from "react";
import { useAuth } from "@/lib/auth";
import { listAlerts, decideAlert, triggerSafetyCheck } from "@/lib/api";

type AItem = {
  id: string;
  source: string;
  alert_type: string;
  severity: string;
  drug_name?: string;
  alert_title: string;
  alert_url?: string;
  ai_summary?: string;
  published_date?: string;
  fetched_at: string;
  status: string;
};

function sevColor(s: string) {
  if (s === "critical") return "bg-red-600 text-white border-red-700";
  if (s === "high") return "bg-amber-100 text-amber-800 border-amber-200";
  if (s === "medium") return "bg-slate-100 text-slate-700 border-slate-200";
  return "bg-slate-50 text-slate-600 border-slate-200";
}

export default function AlertsPage() {
  const { user } = useAuth();
  const [items, setItems] = useState<AItem[]>([]);
  const [total, setTotal] = useState(0);
  const [severity, setSeverity] = useState("");
  const [msg, setMsg] = useState("");
  const [loading, setLoading] = useState(false);

  const fetchList = async () => {
    setLoading(true);
    try {
      const j = await listAlerts({ status: "pending_review", limit: 20, severity: severity || undefined });
      setItems(j.items || []);
      setTotal(j.total || 0);
    } catch (e: any) {
      setMsg(e.message);
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    fetchList();
  }, [severity]);

  const decide = async (aid: string, decision: "approved" | "dismissed") => {
    const notes = prompt(`Ghi chú cho ${decision}:`) || "";
    try {
      await decideAlert(aid, decision, notes || undefined);
      setMsg(`Đã ${decision} ${aid.slice(0, 8)}`);
      fetchList();
    } catch (e: any) {
      setMsg(e.message);
    }
  };

  const trigger = async (src: "fda" | "byt" | "all") => {
    try {
      const j = await triggerSafetyCheck(src);
      setMsg(`Trigger ${src}: found ${j.total_found ?? j.found_new ?? 0}`);
      fetchList();
    } catch (e: any) {
      setMsg(e.message);
    }
  };

  if (!user) return <div className="rounded-xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-6 text-sm">Cần đăng nhập (pharmacist/admin).</div>;
  if (!["pharmacist", "admin"].includes(user.role)) return <div className="rounded-xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-6 text-sm">Cần role pharmacist/admin. Bạn là {user.role}.</div>;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <select value={severity} onChange={(e) => setSeverity(e.target.value)} className="rounded-lg border px-3 py-2 text-sm">
          <option value="">Tất cả mức</option>
          <option value="critical">critical</option>
          <option value="high">high</option>
          <option value="medium">medium</option>
          <option value="low">low</option>
        </select>
        <button onClick={fetchList} className="rounded-lg border bg-white px-4 py-2 text-sm hover:bg-slate-50">
          Làm mới
        </button>
        <button onClick={() => trigger("fda")} className="rounded-lg bg-slate-900 px-3 py-2 text-xs text-white hover:bg-black">
          FDA daily
        </button>
        <button onClick={() => trigger("byt")} className="rounded-lg border bg-white px-3 py-2 text-xs hover:bg-slate-50">
          BYT weekly
        </button>
        <span className="py-2 text-xs text-slate-500">Tổng pending: {total}</span>
      </div>
      {msg && <div className="rounded-lg border bg-slate-50 p-3 text-xs">{msg}</div>}
      {loading ? (
        <div className="text-sm text-slate-500">Đang tải…</div>
      ) : items.length ? (
        <div className="space-y-3">
          {items.map((it) => {
            let sum: any = null;
            try {
              sum = it.ai_summary ? JSON.parse(it.ai_summary) : null;
            } catch {}
            return (
              <div key={it.id} className="rounded-xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-4 shadow-sm">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className={`rounded-full border px-2 py-1 text-[11px] font-medium ${sevColor(it.severity)}`}>{it.severity}</span>
                      <span className="text-xs text-slate-500">
                        {it.source} · {it.alert_type} · {it.fetched_at.slice(0, 16).replace("T", " ")}
                      </span>
                    </div>
                    <div className="mt-2 text-sm font-semibold">{it.drug_name ? `${it.drug_name} — ` : ""}{it.alert_title}</div>
                    {sum?.tom_tat_vi && <div className="mt-2 rounded-lg bg-amber-50 p-3 text-xs text-amber-900">{sum.tom_tat_vi}</div>}
                    {it.alert_url && (
                      <a href={it.alert_url} target="_blank" className="mt-2 inline-block text-xs text-blue-600 underline">
                        Nguồn
                      </a>
                    )}
                    <div className="mt-1 text-[11px] font-mono text-slate-500">{it.id.slice(0, 8)} · {it.status}</div>
                  </div>
                  <div className="flex gap-2">
                    <button onClick={() => decide(it.id, "approved")} className="rounded-full bg-emerald-600 px-4 py-2 text-xs text-white hover:bg-emerald-700">
                      Duyệt
                    </button>
                    <button onClick={() => decide(it.id, "dismissed")} className="rounded-full border bg-white px-4 py-2 text-xs hover:bg-slate-50">
                      Bỏ qua
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="rounded-xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-6 text-sm text-slate-500">Không có cảnh báo pending.</div>
      )}
    </div>
  );
}
