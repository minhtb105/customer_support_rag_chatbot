"use client";
import { useState } from "react";
import { useAuth } from "@/lib/auth";
import { listAlerts, decideAlert, triggerSafetyCheck, fmtDT } from "@/lib/api";
import { useMonitorList, MonitorMsg, MonitorLoading } from "@/components/MonitorList";
import { Button } from "@/components/ui/button";

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
  if (s === "medium") return "bg-slate-100 text-slate-700 border-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-700";
  return "bg-slate-50 text-slate-600 border-slate-200 dark:bg-slate-800 dark:text-slate-300 dark:border-slate-700";
}

export default function AlertsPage() {
  const { user } = useAuth();
  const [severity, setSeverity] = useState("");

  const { items, total, msg, setMsg, loading, fetchList } = useMonitorList<AItem>(
    () => listAlerts({ status: "pending_review", limit: 20, severity: severity || undefined }),
    [severity]
  );

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
        <Button variant="outline" onClick={fetchList}>
          Làm mới
        </Button>
        <Button variant="dark" size="sm" onClick={() => trigger("fda")}>
          FDA daily
        </Button>
        <Button variant="outline" size="sm" onClick={() => trigger("byt")}>
          BYT weekly
        </Button>
        <span className="py-2 text-xs text-slate-500 dark:text-slate-400">Tổng pending: {total}</span>
      </div>
      <MonitorMsg msg={msg} />
      <MonitorLoading loading={loading} empty={!items.length} emptyText="Không có cảnh báo pending.">
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
                      <span className="text-xs text-slate-500 dark:text-slate-400">
                        {it.source} · {it.alert_type} · {fmtDT(it.fetched_at)}
                      </span>
                    </div>
                    <div className="mt-2 text-sm font-semibold">{it.drug_name ? `${it.drug_name} — ` : ""}{it.alert_title}</div>
                    {sum?.tom_tat_vi && <div className="mt-2 rounded-lg bg-amber-50 p-3 text-xs text-amber-900 dark:bg-amber-950 dark:text-amber-100">{sum.tom_tat_vi}</div>}
                    {it.alert_url && (
                      <a href={it.alert_url} target="_blank" className="mt-2 inline-block text-xs text-blue-600 underline">
                        Nguồn
                      </a>
                    )}
                    <div className="mt-1 text-[11px] font-mono text-slate-500 dark:text-slate-400">{it.id.slice(0, 8)} · {it.status}</div>
                  </div>
                  <div className="flex gap-2">
                    <Button variant="success" size="sm" onClick={() => decide(it.id, "approved")}>
                      Duyệt
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => decide(it.id, "dismissed")}>
                      Bỏ qua
                    </Button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </MonitorLoading>
    </div>
  );
}
