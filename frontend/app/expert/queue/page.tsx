"use client";
import { useEffect, useState } from "react";
import { getReviews, getReviewDetail, decideReview, API_BASE } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useRouter } from "next/navigation";

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
        <p className="text-xs text-slate-600">AI thiếu tự tin (faithfulness / precision / recall / answer_relevance &lt; ngưỡng) → chuyển tới {user?.role}. Polling 8s + xem toàn bộ vitals.</p>
        <div className="text-xs text-slate-500">API: {API_BASE} · Auto route: faithfulness→doctor, precision/recall→specialist, drug queries→pharmacist</div>
      </div>
      {msg && <div className="rounded-lg border bg-amber-50 p-2 text-xs">{msg}</div>}
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-2xl border bg-white p-4 shadow-sm">
          <h2 className="text-sm font-semibold">Pending ({reviews.length})</h2>
          <div className="mt-3 space-y-2 max-h-[600px] overflow-auto">
            {reviews.length===0 && <div className="text-xs text-slate-500">Không có yêu cầu pending</div>}
            {reviews.map((r:any)=>(
              <div key={r.id} onClick={()=>open(r.id)} className={`rounded-xl border p-3 cursor-pointer hover:bg-slate-50 ${selected?.id===r.id? "bg-blue-50 border-blue-200":"bg-white"}`}>
                <div className="text-xs font-mono text-slate-500">{r.routed_role} · {r.status} · {r.created_at?.slice(0,16)}</div>
                <div className="text-sm font-medium line-clamp-2">{r.query}</div>
                <div className="text-xs text-slate-600 line-clamp-2">{r.draft_answer?.slice(0,120)}</div>
                <div className="text-[11px] text-slate-500">Failed: {(r.failed_metrics? JSON.parse(r.failed_metrics):[]).join(", ") || JSON.stringify(r.evaluation?.failed_metrics)}</div>
                <div className="text-[11px] text-slate-500">Requester: {r.requester_id?.slice(0,8)}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="rounded-2xl border bg-white p-4 shadow-sm">
          {!selected ? <div className="text-sm text-slate-500">Chọn một yêu cầu để xem chi tiết + vitals + duyệt.</div> : (
            <div className="space-y-4">
              <div>
                <div className="text-xs text-slate-500">Query</div>
                <div className="text-sm font-medium">{selected.query}</div>
                <div className="text-xs">Routed: {selected.routed_role} · Failed: {JSON.stringify(selected.evaluation?.failed_metrics || selected.failed_metrics)}</div>
                <div className="text-xs">Confidence: {selected.confidence} · Evaluation: {JSON.stringify(selected.evaluation?.metrics)}</div>
              </div>
              <div>
                <div className="text-xs font-semibold">Draft answer (AI)</div>
                <div className="rounded-lg bg-slate-50 border p-3 text-xs whitespace-pre-wrap">{selected.draft_answer}</div>
              </div>
              <div>
                <div className="text-xs font-semibold">Contexts (trích)</div>
                <div className="max-h-40 overflow-auto rounded-lg border p-2 text-xs bg-slate-50">
                  {selected.contexts?.map((c:any,i:number)=><div key={i} className="mb-2"><b>[Source {c.source_id}]</b> {c.content?.slice(0,300)}</div>)}
                </div>
              </div>
              <div>
                <div className="text-xs font-semibold">Vitals của user (toàn bộ)</div>
                <div className="rounded-lg border p-2 text-xs bg-white max-h-60 overflow-auto">
                  <div>Requester: {selected.requester?.username} ({selected.requester?.role})</div>
                  <div className="mt-1">Glucose: {(selected.vitals?.glucose_logs||[]).length} logs · avg {selected.vitals?.glucose_stats?.avg_mgdl}</div>
                  {(selected.vitals?.glucose_logs||[]).slice(0,3).map((l:any)=><div key={l.id} className="text-[11px] text-slate-600">{l.measured_at?.slice(0,16)} {l.value_mgdl} mg/dL {l.classification}</div>)}
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
                <input value={notes} onChange={(e)=>setNotes(e.target.value)} className="w-full rounded-lg border p-2 text-xs" placeholder="ghi chú duyệt" />
              </div>
              <div className="flex gap-2">
                <button onClick={()=>decide("approved")} className="rounded-full bg-emerald-600 text-white px-4 py-2 text-xs">Duyệt (approved)</button>
                <button onClick={()=>decide("revised")} className="rounded-full bg-blue-600 text-white px-4 py-2 text-xs">Duyệt có sửa (revised)</button>
                <button onClick={()=>decide("rejected")} className="rounded-full bg-red-600 text-white px-4 py-2 text-xs">Từ chối</button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
