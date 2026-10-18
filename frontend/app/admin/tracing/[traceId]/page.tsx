"use client";
import { useEffect, useState } from "react";
import { adminGetTrace, adminTriggerRagas, adminAddTraceFeedback, parseFailedMetrics, formatRawJson, fmtDT } from "@/lib/api";
import { useParams } from "next/navigation";
import { Check, X, Copy, Star, ChevronRight, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

const TABS = ["Tổng quan", "Chunks", "Memory", "RAGAS", "Raw"] as const;

// A5: clipboard + textarea fallback (http / quyền clipboard bị chặn).
async function copyText(t: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(t);
    return true;
  } catch {
    try {
      const ta = document.createElement("textarea");
      ta.value = t;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
      return true;
    } catch {
      return false;
    }
  }
}

// A6: JSON viewer đơn giản, không thêm dep (dùng formatRawJson sẵn có).
function JsonBlock({ text }: { text: string }) {
  return <pre className="mt-1 max-h-64 overflow-auto rounded bg-slate-900 p-3 text-[11px] text-slate-100">{formatRawJson(text)}</pre>;
}

function CopyBtn({ text, label }: { text: string; label: string }) {
  const [ok, setOk] = useState<boolean | null>(null);
  return (
    <button
      onClick={async (e) => { e.stopPropagation(); setOk(await copyText(text)); setTimeout(() => setOk(null), 1500); }}
      title={label}
      aria-label={label}
      className="inline-flex min-h-[32px] min-w-[32px] items-center gap-1 rounded border px-1.5 py-0.5 text-[11px] text-slate-500 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-700"
    >
      <Copy size={13} aria-hidden />{ok === true ? "Đã copy" : ok === false ? "Copy lỗi" : ""}
    </button>
  );
}

export default function TraceDetailPage(){
  const params = useParams();
  const traceId = params.traceId as string;
  const [data,setData]=useState<any>(null);
  const [msg,setMsg]=useState("");
  const [loading,setLoading]=useState(false);
  const [expired,setExpired]=useState(false);
  const [tab,setTab]=useState<string>("Tổng quan");
  const [selSpan,setSelSpan]=useState<string | null>(null);
  const [expanded,setExpanded]=useState<Record<string,boolean>>({});
  // Phase B: cây span theo parent_id (rows cũ flat → toàn roots, view như cũ).
  const [treeClosed,setTreeClosed]=useState<Record<string,boolean>>({});
  const spanKids = (() => {
    const m: Record<string, any[]> = { ROOT: [] };
    for (const s of (spans || [])) {
      const p = (s as any).parent_id || "ROOT";
      (m[p] = m[p] || []).push(s);
    }
    return m;
  })();
  const [fbScore,setFbScore]=useState(0);
  const [fbComment,setFbComment]=useState("");
  const [fbBusy,setFbBusy]=useState(false);
  const load=async()=>{
    try{ const d=await adminGetTrace(traceId); setData(d); setExpired(false); }catch(e:any){ setMsg(e.message); }
  };
  useEffect(()=>{
    load();
    const id=setTimeout(()=>{ setExpired((v)=>{ if(!v) setMsg((m)=>m || "Tải quá 10s — có thể trace lớn hoặc backend chậm. Bấm Tải lại."); return true; }); },10000);
    return ()=>clearTimeout(id);
  },[traceId]);
  const trigger=async()=>{
    setLoading(true);
    try{ await adminTriggerRagas(traceId); await load(); setMsg("Đã tính RAGAS"); }catch(e:any){ setMsg(e.message); }finally{ setLoading(false); }
  };
  if(!data) return <div className="p-6 text-sm space-y-2"><div>{msg || "Đang tải trace..."}</div>{!data && !msg && <div className="h-4 w-2/3 animate-pulse rounded bg-slate-200 dark:bg-slate-700" />}{!data && !msg && <div className="h-4 w-1/2 animate-pulse rounded bg-slate-200 dark:bg-slate-700" />}<Button variant="outline" size="sm" onClick={load}>Tải lại</Button></div>;
  const { trace, spans, chunks, ragas, review } = data;
  const MEMORY_DATASETS = ["short_term_memory", "episodic_memory", "long_term_memory"];
  const memoryRows = (chunks || []).filter((c: any) => MEMORY_DATASETS.includes(c.dataset));
  const hasBuildSpan = (spans || []).some((s: any) => s.name === "build_llm_input");
  // A1: thanh duration tương đối theo span dài nhất.
  const maxDur = Math.max(1, ...(spans || []).map((s: any) => Number(s.duration_ms) || 0));
  const toggleRow=(id:string)=>setExpanded(s=>({ ...s, [id]: !s[id] }));
  // Phase B: node đệ quy — con thụt đầu dòng, collapse theo parent; flat cũ = roots.
  const SpanNode = ({ s, depth }: { s: any; depth: number }) => {
    const pct = Math.max(2, Math.round(((Number(s.duration_ms) || 0) / maxDur) * 100));
    const open = selSpan === s.id;
    const kids = spanKids[s.id] || [];
    const closed = !!treeClosed[s.id];
    const toggleTree = (e: any) => { e.stopPropagation(); setTreeClosed((t) => ({ ...t, [s.id]: !t[s.id] })); };
    const toggleOpen = () => setSelSpan(open ? null : s.id);
    return (
      <div style={{ marginLeft: depth ? depth * 16 : 0 }}>
        <div onClick={toggleOpen} role="button" tabIndex={0} aria-expanded={open} onKeyDown={(e)=>{ if(e.key==="Enter"||e.key===" "){ e.preventDefault(); toggleOpen(); } }} className="cursor-pointer rounded-xl border p-3 bg-slate-50 dark:bg-slate-800 dark:border-slate-700 hover:border-violet-300">
          <div className="flex justify-between text-xs">
            <span className="font-bold flex items-center gap-1">
              {kids.length > 0 && (
                <button onClick={toggleTree} aria-expanded={!closed} aria-label={closed ? "Mở rộng cây con" : "Thu gọn cây con"} className="rounded p-0.5 hover:bg-slate-200 dark:hover:bg-slate-700 min-h-[32px] min-w-[32px] inline-flex items-center justify-center">
                  {closed ? <ChevronRight size={14} aria-hidden /> : <ChevronDown size={14} aria-hidden />}
                </button>
              )}
              {s.name}
            </span>
            <span className="flex items-center gap-2">{s.duration_ms? Math.round(s.duration_ms)+"ms":"—"} · {s.start_at?.slice(11,19)} <CopyBtn text={s.id} label="Copy span ID" /></span>
          </div>
          <div className="mt-1 h-1.5 rounded bg-slate-200 dark:bg-slate-700"><div className="h-1.5 rounded bg-violet-500" style={{ width: `${pct}%` }} /></div>
          {s.inputs_json && <div className="text-[11px] text-slate-600 dark:text-slate-300">in: {s.inputs_json.slice(0,200)}</div>}
          {s.outputs_json && <div className="text-[11px] text-slate-600 dark:text-slate-300">out: {s.outputs_json.slice(0,200)}</div>}
          {s.metadata_json && <div className="text-[11px] text-slate-500 dark:text-slate-400">meta: {s.metadata_json.slice(0,300)}</div>}
        </div>
        {open && (
          <div className="ml-4 mt-1 rounded-xl border border-violet-200 p-3 dark:border-violet-800">
            <div className="flex items-center gap-2 text-[11px] text-slate-500 dark:text-slate-400"><span className="font-mono">{s.id}</span><CopyBtn text={s.id} label="Copy span ID" /></div>
            {s.inputs_json ? <><div className="mt-2 text-xs font-semibold">Input</div><JsonBlock text={s.inputs_json} /></> : <div className="mt-2 text-xs text-slate-500">Không có input</div>}
            {s.outputs_json ? <><div className="mt-2 text-xs font-semibold">Output</div><JsonBlock text={s.outputs_json} /></> : <div className="mt-2 text-xs text-slate-500">Không có output</div>}
          </div>
        )}
        {!closed && kids.map((k: any) => <SpanNode key={k.id} s={k} depth={depth + 1} />)}
      </div>
    );
  };
  return (
    <div className="space-y-4">
      <div className="rounded-2xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-4 shadow-sm">
        <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400"><span>Trace {trace.id} · {fmtDT(trace.created_at)} · user {trace.username || trace.user_id.slice(0,8)}</span><CopyBtn text={trace.id} label="Copy trace ID" /></div>
        <h1 className="text-sm font-bold mt-1">{trace.query}</h1>
        <div className="mt-2 text-xs">Tone <span className="rounded-full bg-slate-100 border px-2 py-0.5 dark:bg-slate-800 dark:border-slate-700">{trace.tone}</span> · Prompt {trace.prompt_version} · Model {trace.model} · Embedding {trace.embedding_model} · TopK {trace.top_k} · {trace.total_latency_ms? Math.round(trace.total_latency_ms)+"ms":""}</div>
        <div className="mt-3 rounded-lg bg-slate-50 border p-3 text-sm whitespace-pre-wrap dark:bg-slate-800 dark:border-slate-700">{trace.answer || "—"}</div>
        {review && <div className="mt-2 text-xs">Review {review.id} — {review.status} · routed {review.routed_role}</div>}
      </div>

      <div className="flex gap-1 overflow-x-auto" role="tablist" aria-label="Chi tiết trace">
        {TABS.map((t)=>(
          <button key={t} role="tab" aria-selected={tab===t} onClick={()=>setTab(t)} className={`rounded-full px-4 py-2 text-xs min-h-[44px] whitespace-nowrap ${tab===t ? "bg-slate-900 text-white" : "border hover:bg-slate-50 dark:hover:bg-slate-800"}`}>{t}</button>
        ))}
      </div>

      {tab==="Tổng quan" && (
      <div className="rounded-2xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-4 shadow-sm">
        <h2 className="text-sm font-semibold">Waterfall — Spans</h2>
        <div className="mt-3 space-y-2">
          {spanKids.ROOT?.map((s:any)=>(
            <SpanNode key={s.id} s={s} depth={0} />
          ))}
        </div>
      </div>
      )}

      {tab==="Memory" && (
      <div className="rounded-2xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-4 shadow-sm">
        <h2 className="text-sm font-semibold">Memory context <span className="font-normal text-slate-500 dark:text-slate-400">(truncated 500)</span></h2>
        {!hasBuildSpan ? (
          <div className="mt-2 text-xs text-slate-500 dark:text-slate-400">— (cache hit)</div>
        ) : memoryRows.length === 0 ? (
          <div className="mt-2 text-xs text-slate-500 dark:text-slate-400">không có memory context</div>
        ) : (
          <div className="mt-3 overflow-auto">
            <table className="w-full text-xs">
              <thead className="bg-slate-50 dark:bg-slate-800 border-b text-left"><tr><th className="p-2">Dataset</th><th className="p-2">Source</th><th className="p-2">Score</th><th className="p-2">Snippet (truncated 500)</th></tr></thead>
              <tbody>
                {memoryRows.map((c: any) => (
                  <tr key={c.id} className="border-b">
                    <td className="p-2"><span className="rounded-full bg-violet-50 border border-violet-200 px-2 py-0.5">{c.dataset}</span></td>
                    <td className="p-2 font-mono">{c.source_id}</td>
                    <td className="p-2">{c.score ? Number(c.score).toFixed(3) : "—"}</td>
                    <td className="p-2 max-w-[480px] whitespace-pre-wrap">{expanded[c.id] && c.content_full ? c.content_full : c.content_snippet?.slice(0, 500)}{c.content_full && <div><button onClick={()=>toggleRow(c.id)} aria-expanded={!!expanded[c.id]} className="text-blue-700 hover:underline min-h-[44px] px-1 dark:text-blue-300">{expanded[c.id] ? "Thu gọn" : "Mở rộng"}</button></div>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">Nguồn span_chunks: cắt 500 ký tự (snippet = full).</div>
          </div>
        )}
      </div>
      )}

      {tab==="Chunks" && (
      <div className="rounded-2xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-4 shadow-sm">
        <h2 className="text-sm font-semibold">Chunks retrieved — đầy đủ metadata (snippet 500, page đầu tiên)</h2>
        <div className="mt-3 overflow-auto">
          <table className="w-full text-xs">
            <thead className="bg-slate-50 dark:bg-slate-800 border-b text-left"><tr><th>#</th><th>Source</th><th>File</th><th>Trang</th><th>Strategy</th><th>Embedding</th><th>Score</th><th>Snippet 500</th></tr></thead>
            <tbody>
              {chunks?.map((c:any)=>(
                <tr key={c.id} className="border-b">
                  <td className="p-2">{c.rank+1}</td>
                  <td className="p-2 font-mono">{c.source_id}</td>
                  <td className="p-2">{c.file_name || "—"}<div className="text-[11px] text-slate-500 dark:text-slate-400">{c.dataset}</div></td>
                  <td className="p-2">{c.page_numbers ? JSON.parse(c.page_numbers)[0] : (c.first_page || "—")}<div className="text-[11px] text-slate-500 dark:text-slate-400">{c.page_numbers}</div></td>
                  <td className="p-2">{c.chunking_strategy}</td>
                  <td className="p-2 text-[11px]">{c.embedding_model}</td>
                  <td className="p-2">{c.score? Number(c.score).toFixed(3):"—"}</td>
                  <td className="p-2 max-w-[400px]">{expanded[c.id] && c.content_full ? c.content_full : c.content_snippet?.slice(0,500)}{c.content_full && <div><button onClick={()=>toggleRow(c.id)} aria-expanded={!!expanded[c.id]} className="text-blue-700 hover:underline min-h-[44px] px-1 dark:text-blue-300">{expanded[c.id] ? "Thu gọn" : "Mở rộng"}</button></div>}<div className="text-[11px] text-slate-500 dark:text-slate-400">updated {c.updated_at?.slice(0,10) || "—"}</div></td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-1 text-[11px] text-slate-500 dark:text-slate-400">Nguồn span_chunks: cắt 500 ký tự (snippet = full).</div>
        </div>
      </div>
      )}

      {tab==="RAGAS" && (
      <div className="rounded-2xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-4 shadow-sm">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">RAGAS (on-demand)</h2>
          <Button variant="accent" size="sm" onClick={trigger} disabled={loading}>{loading?"Đang tính...":"Tính RAGAS"}</Button>
        </div>
        {ragas ? (
          <div className="mt-3 grid grid-cols-2 sm:grid-cols-5 gap-2">
            <Metric label="faithfulness" v={ragas.faithfulness} />
            <Metric label="precision" v={ragas.context_precision} />
            <Metric label="recall" v={ragas.context_recall} />
            <Metric label="relevance" v={ragas.answer_relevance} />
            <Metric label="fluency" v={ragas.fluency} />
          </div>
        ) : <div className="mt-2 text-xs text-slate-500 dark:text-slate-400">Chưa có RAGAS — bấm “Tính RAGAS” (admin on-demand) để đánh giá 5 metrics, tốn 1 LLM call.</div>}
        {ragas && <div className="mt-2 text-xs">Failed: {parseFailedMetrics(ragas.failed_metrics).join(", ") || "none"} · Confidence {ragas.confidence} · {fmtDT(ragas.evaluated_at)}</div>}
        {ragas?.raw_json && <details className="mt-2 text-xs"><summary>Raw</summary><pre className="mt-1 bg-slate-900 text-slate-100 p-3 rounded overflow-auto">{formatRawJson(ragas.raw_json)}</pre></details>}
      </div>
      )}

      {tab==="Raw" && (
      <div className="rounded-2xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-4 shadow-sm space-y-2">
        <details className="text-xs"><summary className="cursor-pointer font-semibold">Trace JSON</summary><JsonBlock text={JSON.stringify(trace, null, 2)} /></details>
        <details className="text-xs"><summary className="cursor-pointer font-semibold">Spans JSON ({(spans || []).length})</summary><JsonBlock text={JSON.stringify(spans, null, 2)} /></details>
        <details className="text-xs"><summary className="cursor-pointer font-semibold">Chunks JSON ({(chunks || []).length})</summary><JsonBlock text={JSON.stringify(chunks, null, 2)} /></details>
      </div>
      )}
      {msg && <div className="rounded-lg bg-amber-50 border p-2 text-xs dark:bg-amber-950 dark:border-amber-800 dark:text-amber-100">{msg}</div>}

      <div className="rounded-2xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-4 shadow-sm">
        <h2 className="text-sm font-semibold">Phản hồi chất lượng</h2>
        <div className="mt-2 flex items-center gap-1" role="radiogroup" aria-label="Chấm điểm trace (1-5 sao)">
          {[1,2,3,4,5].map((s)=>(
            <button key={s} role="radio" aria-checked={fbScore===s} aria-label={`Chấm ${s} sao`} onClick={()=>setFbScore(s)} className="min-h-[44px] min-w-[44px] rounded p-1 hover:bg-slate-100 dark:hover:bg-slate-800">
              <Star size={20} aria-hidden className={s<=fbScore ? "fill-amber-400 text-amber-400" : "text-slate-400"} />
            </button>
          ))}
        </div>
        <div className="mt-2 flex flex-col gap-2 sm:flex-row">
          <Input value={fbComment} onChange={(e)=>setFbComment(e.target.value)} placeholder="Nhận xét (tùy chọn)" aria-label="Nhận xét phản hồi" className="flex-1 text-xs dark:bg-slate-800 dark:border-slate-700" />
          <Button variant="accent" size="sm" disabled={fbBusy || fbScore<1} onClick={async()=>{ setFbBusy(true); try{ await adminAddTraceFeedback(traceId, { score: fbScore, comment: fbComment || undefined }); setFbScore(0); setFbComment(""); await load(); setMsg("Đã lưu phản hồi"); }catch(e:any){ setMsg(e.message); }finally{ setFbBusy(false); } }}>Gửi phản hồi</Button>
        </div>
        <div className="mt-3 space-y-2">
          {((data as any).feedback || []).map((f:any)=>(
            <div key={f.id} className="rounded-xl border p-2 text-xs dark:border-slate-700">
              <div className="flex items-center gap-1" aria-label={`Điểm ${f.score}/5`}>
                {[1,2,3,4,5].map((s)=>(<Star key={s} size={13} aria-hidden className={s<=(f.score||0) ? "fill-amber-400 text-amber-400" : "text-slate-300"} />))}
                <span className="ml-1 text-slate-500 dark:text-slate-400">{fmtDT(f.created_at)}</span>
              </div>
              {f.comment && <div className="mt-1 whitespace-pre-wrap">{f.comment}</div>}
            </div>
          ))}
          {!(data as any).feedback?.length && <div className="text-xs text-slate-500 dark:text-slate-400">Chưa có phản hồi.</div>}
        </div>
      </div>
    </div>
  );
}
function Metric({label,v}:{label:string, v:any}){
  const val = Number(v);
  const ok = val>=3.5;
  return <div className={`rounded-lg border p-2 ${ok?"bg-emerald-50 border-emerald-200":"bg-red-50 border-red-200"}`}><div className="text-[11px] font-bold">{label}</div><div className="text-sm">{val?.toFixed(1) ?? "—"}/5</div><div className="text-[11px]">{ok?<span className="inline-flex items-center gap-1"><Check size={13} aria-hidden /> đạt</span>:<span className="inline-flex items-center gap-1"><X size={13} aria-hidden /> chưa đạt</span>}</div></div>;
}
