"use client";
import { useEffect, useState } from "react";
import { useAuth } from "@/lib/auth";
import { listGuidelines, decideGuideline, triggerGuidelineCheck } from "@/lib/api";

type GItem = {
  id: string;
  source: string;
  title: string;
  version_label?: string;
  status: string;
  sha256?: string;
  staging_path?: string;
  change_summary_json?: string;
  fetched_at: string;
};

export default function GuidelinesPage() {
  const { user } = useAuth();
  const [items, setItems] = useState<GItem[]>([]);
  const [total, setTotal] = useState(0);
  const [status, setStatus] = useState("pending_review");
  const [msg, setMsg] = useState("");
  const [loading, setLoading] = useState(false);

  const fetchList = async () => {
    setLoading(true);
    try {
      const j = await listGuidelines({ status, limit: 20 });
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
  }, [status]);

  const decide = async (gid: string, decision: "approved" | "rejected") => {
    const notes = prompt(`Ghi chú cho ${decision} (để trống nếu không cần):`) || "";
    try {
      await decideGuideline(gid, decision, notes || undefined);
      setMsg(`Đã ${decision} ${gid.slice(0, 8)}`);
      fetchList();
    } catch (e: any) {
      setMsg(e.message);
    }
  };

  const trigger = async () => {
    const src = prompt("source_key (gold,gina,ada_soc,who_mhgap,who_diabetes,aha_acc_htn,byt_diabetes) hoặc để trống = all:", "gold") || "gold";
    try {
      const j = await triggerGuidelineCheck(src, true);
      setMsg(`Trigger ${src}: ${JSON.stringify(j).slice(0, 300)}`);
      fetchList();
    } catch (e: any) {
      setMsg(e.message);
    }
  };

  if (!user) return <div className="rounded-xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-6 text-sm">Cần đăng nhập (specialist/doctor/admin).</div>;
  if (!["specialist", "doctor", "admin"].includes(user.role)) return <div className="rounded-xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-6 text-sm">Cần role specialist/doctor/admin. Bạn là {user.role}.</div>;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <select value={status} onChange={(e) => setStatus(e.target.value)} className="rounded-lg border px-3 py-2 text-sm">
          <option value="pending_review">pending_review</option>
          <option value="approved">approved</option>
          <option value="rejected">rejected</option>
          <option value="superseded">superseded</option>
        </select>
        <button onClick={fetchList} className="rounded-lg border bg-white px-4 py-2 text-sm hover:bg-slate-50">
          Làm mới
        </button>
        <button onClick={trigger} className="rounded-lg bg-slate-900 px-4 py-2 text-sm text-white hover:bg-black">
          Kiểm tra ngay (HEAD)
        </button>
        <span className="py-2 text-xs text-slate-500">Tổng: {total}</span>
      </div>
      {msg && <div className="rounded-lg border bg-slate-50 p-3 text-xs">{msg}</div>}
      {loading ? (
        <div className="text-sm text-slate-500">Đang tải…</div>
      ) : items.length ? (
        <div className="space-y-3">
          {items.map((it) => {
            let summary: any = null;
            try {
              summary = it.change_summary_json ? JSON.parse(it.change_summary_json) : null;
            } catch {}
            return (
              <div key={it.id} className="rounded-xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-4 shadow-sm">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold">
                      [{it.source}] {it.title}
                    </div>
                    <div className="text-xs text-slate-500">
                      v: {it.version_label || "—"} · {it.status} · {it.fetched_at.slice(0, 16).replace("T", " ")} · {it.id.slice(0, 8)}
                    </div>
                    {summary?.tom_tat_tieng_viet && <div className="mt-2 rounded-lg bg-blue-50 p-3 text-xs text-blue-900">{summary.tom_tat_tieng_viet}</div>}
                    {it.sha256 && <div className="mt-1 text-[11px] font-mono text-slate-500">sha256 {it.sha256.slice(0, 16)}…</div>}
                  </div>
                  {it.status === "pending_review" && (
                    <div className="flex gap-2">
                      <button onClick={() => decide(it.id, "approved")} className="rounded-full bg-emerald-600 px-4 py-2 text-xs text-white hover:bg-emerald-700">
                        Duyệt
                      </button>
                      <button onClick={() => decide(it.id, "rejected")} className="rounded-full border bg-white px-4 py-2 text-xs hover:bg-slate-50">
                        Từ chối
                      </button>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="rounded-xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-6 text-sm text-slate-500">Không có bản ghi {status}.</div>
      )}
    </div>
  );
}
