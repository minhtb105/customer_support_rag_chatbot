"use client";
import { useEffect, useState } from "react";
import { getReviews, getReviewDetail, decideReview, API_BASE, parseFailedMetrics, fmtDT } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export default function ExpertQueuePage() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [reviews, setReviews] = useState<any[]>([]);
  const [selected, setSelected] = useState<any>(null);
  const [finalAns, setFinalAns] = useState("");
  const [notes, setNotes] = useState("");
  const [msg, setMsg] = useState("");

  useEffect(() => {
    if (!loading && !user) router.push("/login");
    if (user && !["doctor","pharmacist","specialist","admin"].includes(user.role)) router.push("/");
  }, [user, loading]);

  const load = async () => {
    try {
      const data = await getReviews("pending");
      setReviews(data.reviews || []);
    } catch (e:any) { setMsg(e.message); }
  };
  useEffect(()=>{ load(); const id=setInterval(load, 8000); return()=>clearInterval(id); },[]);

  const open = async (id:string) => {
    const d = await getReviewDetail(id);
    setSelected(d);
    setFinalAns(d.draft_answer);
    setNotes("");
  };
  const decide = async (decision:string) => {
    if (!selected) return;
    try {
      await decideReview(selected.id, { decision, final_answer: decision==="revised"? finalAns: undefined, expert_notes: notes || undefined });
      setMsg(`Đã ${decision}`);
      setSelected(null);
      load();
    } catch(e:any){ setMsg(e.message); }
  };

  if (loading) return <div className="text-sm">Đang tải…</div>;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold">Hàng đợi duyệt HILT</h1>
        <p className="text-xs text-slate-600 dark:text-slate-300">AI thiếu tự tin (faithfulness / precision / recall / answer_relevance &lt; ngưỡng) → chuyển tới {user?.role}. Polling 8s + xem toàn bộ vitals.</p>
        <div className="text-xs text-slate-500 dark:text-slate-400">API: {API_BASE} · Auto route: faithfulness→doctor, precision/recall→specialist, drug queries→pharmacist</div>
      </div>
      {msg && <div className="rounded-lg border bg-amber-50 p-2 text-xs dark:bg-amber-950 dark:border-amber-800 dark:text-amber-100">{msg}</div>}
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-2xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-4 shadow-sm">
          <h2 className="text-sm font-semibold">Pending ({reviews.length})</h2>
          <div className="mt-3 space-y-2 max-h-[600px] overflow-auto">
            {reviews.length===0 && <div className="text-xs text-slate-500 dark:text-slate-400">Không có yêu cầu pending</div>}
            {reviews.map((r:any)=>(
              <div key={r.id} onClick={()=>open(r.id)} className={`rounded-xl border p-3 cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800 ${selected?.id===r.id? "bg-blue-50 border-blue-200 dark:bg-blue-950 dark:border-blue-800":"bg-white dark:bg-slate-900 dark:border-slate-700"}`}>
                <div className="text-xs font-mono text-slate-500 dark:text-slate-400">{r.routed_role} · {r.status} · {fmtDT(r.created_at)}</div>
                <div className="text-sm font-medium line-clamp-2">{r.query}</div>
                <div className="text-xs text-slate-600 dark:text-slate-300 line-clamp-2">{r.draft_answer?.slice(0,120)}</div>
                <div className="text-[11px] text-slate-500 dark:text-slate-400">Failed: {parseFailedMetrics(r.failed_metrics).join(", ") || JSON.stringify(r.evaluation?.failed_metrics)}</div>
                <div className="text-[11px] text-slate-500 dark:text-slate-400">Requester: {r.requester_username || r.requester_id?.slice(0,8)}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="rounded-2xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-4 shadow-sm">
          {!selected ? <div className="text-sm text-slate-500 dark:text-slate-400">Chọn một yêu cầu để xem chi tiết + vitals + duyệt.</div> : (
            <div className="space-y-4">
              <div>
                <div className="text-xs text-slate-500 dark:text-slate-400">Query</div>
                <div className="text-sm font-medium">{selected.query}</div>
                <div className="text-xs">Routed: {selected.routed_role} · Failed: {JSON.stringify(selected.evaluation?.failed_metrics || selected.failed_metrics)}</div>
                <div className="text-xs">Confidence: {selected.confidence} · Evaluation: {JSON.stringify(selected.evaluation?.metrics)}</div>
              </div>
              <div>
                <div className="text-xs font-semibold">Draft answer (AI)</div>
                <div className="rounded-lg bg-slate-50 border p-3 text-xs whitespace-pre-wrap dark:bg-slate-800 dark:border-slate-700">{selected.draft_answer}</div>
              </div>
              <div>
                <div className="text-xs font-semibold">Contexts (trích)</div>
                <div className="max-h-40 overflow-auto rounded-lg border p-2 text-xs bg-slate-50 dark:bg-slate-800 dark:border-slate-700">
                  {selected.contexts?.map((c:any,i:number)=><div key={i} className="mb-2"><b>[Source {c.source_id}]</b> {c.content?.slice(0,300)}</div>)}
                </div>
              </div>
              <div>
                <div className="text-xs font-semibold">Vitals của user (toàn bộ)</div>
                <div className="rounded-lg border p-2 text-xs bg-white max-h-60 overflow-auto dark:bg-slate-900 dark:border-slate-700">
                  <div>Requester: {selected.requester?.username} ({selected.requester?.role})</div>
                  <div className="mt-1">Glucose: {(selected.vitals?.glucose_logs||[]).length} logs · avg {selected.vitals?.glucose_stats?.avg_mgdl}</div>
                  {(selected.vitals?.glucose_logs||[]).slice(0,3).map((l:any)=><div key={l.id} className="text-[11px] text-slate-600 dark:text-slate-300">{fmtDT(l.measured_at)} {l.value_mgdl} mg/dL {l.classification}</div>)}
                  <div className="mt-1">BP: {(selected.vitals?.bp_logs||[]).length} logs</div>
                  <div>Mood: {(selected.vitals?.mood_logs||[]).length} logs</div>
                  <div>Respiratory: {(selected.vitals?.respiratory_logs||[]).length} logs</div>
                </div>
              </div>
              <div>
                <label className="text-xs">Final answer (chỉnh sửa nếu revised)</label>
                <textarea value={finalAns} onChange={(e)=>setFinalAns(e.target.value)} rows={6} className="w-full rounded-lg border p-2 text-xs" />
              </div>
              <div>
                <label className="text-xs">Expert notes</label>
                <Input value={notes} onChange={(e)=>setNotes(e.target.value)} className="text-xs" placeholder="ghi chú duyệt" />
              </div>
              <div className="flex gap-2">
                <Button variant="success" size="sm" onClick={()=>decide("approved")}>Duyệt (approved)</Button>
                <Button size="sm" onClick={()=>decide("revised")}>Duyệt có sửa (revised)</Button>
                <Button variant="destructive" size="sm" onClick={()=>decide("rejected")}>Từ chối</Button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
