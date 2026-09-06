"use client";
import { useEffect, useState } from "react";
import { adminGetTrace, adminTriggerRagas } from "@/lib/api";
import { useParams } from "next/navigation";

export default function TraceDetailPage(){
  const params = useParams();
  const traceId = params.traceId as string;
  const [data,setData]=useState<any>(null);
  const [msg,setMsg]=useState("");
  const [loading,setLoading]=useState(false);
  const load=async()=>{
    try{ const d=await adminGetTrace(traceId); setData(d);}catch(e:any){ setMsg(e.message); }
  };
  useEffect(()=>{ load(); },[traceId]);
  const trigger=async()=>{
    setLoading(true);
    try{ await adminTriggerRagas(traceId); await load(); setMsg("Đã tính RAGAS"); }catch(e:any){ setMsg(e.message); }finally{ setLoading(false); }
  };
  if(!data) return <div className="p-6 text-sm">{msg || "Đang tải trace..."}</div>;
  const { trace, spans, chunks, ragas, review } = data;
  return (
    <div className="space-y-4">
      <div className="rounded-2xl border bg-white p-4 shadow-sm">
        <div className="text-xs text-slate-500">Trace {trace.id} · {trace.created_at?.slice(0,19).replace("T"," ")} · user {trace.username || trace.user_id.slice(0,8)}</div>
        <h1 className="text-sm font-bold mt-1">{trace.query}</h1>
        <div className="mt-2 text-xs">Tone <span className="rounded-full bg-slate-100 border px-2 py-0.5">{trace.tone}</span> · Prompt {trace.prompt_version} · Model {trace.model} · Embedding {trace.embedding_model} · TopK {trace.top_k} · {trace.total_latency_ms? Math.round(trace.total_latency_ms)+"ms":""}</div>
        <div className="mt-3 rounded-lg bg-slate-50 border p-3 text-sm whitespace-pre-wrap">{trace.answer || "—"}</div>
        {review && <div className="mt-2 text-xs">Review {review.id} — {review.status} · routed {review.routed_role}</div>}
      </div>

      <div className="rounded-2xl border bg-white p-4 shadow-sm">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">RAGAS (on-demand)</h2>
          <button onClick={trigger} disabled={loading} className="rounded-full bg-violet-600 text-white px-3 py-1 text-xs disabled:opacity-50">{loading?"Đang tính...":"Tính RAGAS"}</button>
        </div>
        {ragas ? (
          <div className="mt-3 grid grid-cols-2 sm:grid-cols-5 gap-2">
            <Metric label="faithfulness" v={ragas.faithfulness} />
            <Metric label="precision" v={ragas.context_precision} />
            <Metric label="recall" v={ragas.context_recall} />
            <Metric label="relevance" v={ragas.answer_relevance} />
            <Metric label="fluency" v={ragas.fluency} />
          </div>
        ) : <div className="mt-2 text-xs text-slate-500">Chưa có RAGAS — bấm “Tính RAGAS” (admin on-demand) để đánh giá 5 metrics, tốn 1 LLM call.</div>}
        {ragas && <div className="mt-2 text-xs">Failed: {(JSON.parse(ragas.failed_metrics||"[]")).join(", ") || "none"} · Confidence {ragas.confidence} · {ragas.evaluated_at?.slice(0,19)}</div>}
        {ragas?.raw_json && <details className="mt-2 text-xs"><summary>Raw</summary><pre className="mt-1 bg-slate-900 text-slate-100 p-3 rounded overflow-auto">{JSON.stringify(JSON.parse(ragas.raw_json), null,2)}</pre></details>}
      </div>

      <div className="rounded-2xl border bg-white p-4 shadow-sm">
        <h2 className="text-sm font-semibold">Waterfall — Spans</h2>
        <div className="mt-3 space-y-2">
          {spans?.map((s:any)=>(
            <div key={s.id} className="rounded-xl border p-3 bg-slate-50">
              <div className="flex justify-between text-xs"><span className="font-bold">{s.name}</span><span>{s.duration_ms? Math.round(s.duration_ms)+"ms":"—"} · {s.start_at?.slice(11,19)}</span></div>
              {s.inputs_json && <div className="text-[11px] text-slate-600">in: {s.inputs_json.slice(0,200)}</div>}
              {s.outputs_json && <div className="text-[11px] text-slate-600">out: {s.outputs_json.slice(0,200)}</div>}
              {s.metadata_json && <div className="text-[11px] text-slate-500">meta: {s.metadata_json.slice(0,300)}</div>}
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-2xl border bg-white p-4 shadow-sm">
        <h2 className="text-sm font-semibold">Chunks retrieved — đầy đủ metadata (snippet 500, page đầu tiên)</h2>
        <div className="mt-3 overflow-auto">
          <table className="w-full text-xs">
            <thead className="bg-slate-50 border-b text-left"><tr><th>#</th><th>Source</th><th>File</th><th>Trang</th><th>Strategy</th><th>Embedding</th><th>Score</th><th>Snippet 500</th></tr></thead>
            <tbody>
              {chunks?.map((c:any)=>(
                <tr key={c.id} className="border-b">
                  <td className="p-2">{c.rank+1}</td>
                  <td className="p-2 font-mono">{c.source_id}</td>
                  <td className="p-2">{c.file_name || "—"}<div className="text-[11px] text-slate-500">{c.dataset}</div></td>
                  <td className="p-2">{c.page_numbers ? JSON.parse(c.page_numbers)[0] : (c.first_page || "—")}<div className="text-[11px] text-slate-500">{c.page_numbers}</div></td>
                  <td className="p-2">{c.chunking_strategy}</td>
                  <td className="p-2 text-[11px]">{c.embedding_model}</td>
                  <td className="p-2">{c.score? Number(c.score).toFixed(3):"—"}</td>
                  <td className="p-2 max-w-[400px]">{c.content_snippet?.slice(0,500)}<div className="text-[11px] text-slate-500">updated {c.updated_at?.slice(0,10) || "—"}</div></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      {msg && <div className="rounded-lg bg-amber-50 border p-2 text-xs">{msg}</div>}
    </div>
  );
}
function Metric({label,v}:{label:string, v:any}){
  const val = Number(v);
  const ok = val>=3.5;
  return <div className={`rounded-lg border p-2 ${ok?"bg-emerald-50 border-emerald-200":"bg-red-50 border-red-200"}`}><div className="text-[11px] font-bold">{label}</div><div className="text-sm">{val?.toFixed(1) ?? "—"}/5</div><div className="text-[11px]">{ok?"✅":"❌"}</div></div>;
}
