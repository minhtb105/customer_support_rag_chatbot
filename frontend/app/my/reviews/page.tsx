"use client";
import { useEffect, useState } from "react";
import { getMyReviews, getQueryHistory, getNotifications, API_BASE } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import Link from "next/link";

export default function MyReviewsPage() {
  const { user } = useAuth();
  const [data, setData] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [notif, setNotif] = useState<any>(null);

  const load = async () => {
    try {
      const d = await getMyReviews();
      setData(d);
      const h = await getQueryHistory();
      setHistory(h.history || []);
      const n = await getNotifications();
      setNotif(n);
    } catch(e:any) { console.error(e); }
  };
  useEffect(()=>{ load(); const id=setInterval(load, 7000); return()=>clearInterval(id); },[]);

  if (!user) return <div className="text-sm">Cần đăng nhập <Link href="/login" className="underline">Đăng nhập</Link></div>;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold">Lịch sử & Review của tôi</h1>
        <p className="text-xs text-slate-600">Push notification polling 7s. Khi AI pending → expert duyệt → cập nhật vào lịch sử và notification.</p>
        <div className="text-xs text-slate-500">{API_BASE} · user {user.username} ({user.role})</div>
      </div>
      {notif && (
        <div className="rounded-2xl border bg-white p-4 shadow-sm">
          <h2 className="text-sm font-semibold">Notifications ({notif.unread_count} chưa đọc)</h2>
          <div className="mt-2 space-y-1 max-h-40 overflow-auto">
            {(notif.notifications||[]).slice(0,10).map((n:any)=><div key={n.id} className={`text-xs p-2 rounded ${n.is_read? "bg-white border":"bg-amber-50 border border-amber-200"}`}>{n.title}: {n.body}</div>)}
          </div>
        </div>
      )}
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-2xl border bg-white p-4 shadow-sm">
          <h2 className="text-sm font-semibold">Review requests ({data?.reviews?.length||0})</h2>
          <div className="mt-3 space-y-2 max-h-[500px] overflow-auto">
            {(data?.reviews||[]).map((r:any)=>(
              <div key={r.id} className="rounded-xl border p-3 bg-slate-50">
                <div className="text-xs font-mono">{r.status} · routed {r.routed_role} · {r.created_at?.slice(0,16)}</div>
                <div className="text-sm font-medium">{r.query}</div>
                <div className="text-xs">Draft: {r.draft_answer?.slice(0,150)}</div>
                {r.final_answer && <div className="text-xs bg-emerald-50 border border-emerald-200 rounded p-2 mt-1"><b>Final:</b> {r.final_answer.slice(0,300)}</div>}
                <div className="text-[11px] text-slate-500">Failed: {JSON.stringify(r.evaluation?.failed_metrics || r.failed_metrics)}</div>
              </div>
            ))}
            {!data?.reviews?.length && <div className="text-xs text-slate-500">Chưa có review</div>}
          </div>
        </div>
        <div className="rounded-2xl border bg-white p-4 shadow-sm">
          <h2 className="text-sm font-semibold">Query history ({history.length}) — cập nhật sau duyệt</h2>
          <div className="mt-3 space-y-2 max-h-[500px] overflow-auto">
            {history.map((h:any)=>(
              <div key={h.id} className={`rounded-xl border p-3 ${h.status==="pending_review"? "bg-amber-50 border-amber-200":"bg-white"}`}>
                <div className="text-xs font-mono">{h.status} · {h.created_at?.slice(0,16)} {h.confidence!==null? `conf ${h.confidence}`:""}</div>
                <div className="text-sm">{h.query}</div>
                <div className="text-xs text-slate-600 line-clamp-3">{h.answer?.slice(0,300)}</div>
                {h.review_id && <div className="text-[11px] text-slate-500">review {h.review_id.slice(0,8)}</div>}
              </div>
            ))}
            {!history.length && <div className="text-xs text-slate-500">Chưa có history</div>}
          </div>
        </div>
      </div>
    </div>
  );
}
