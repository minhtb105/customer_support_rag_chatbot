"use client";
import { useState } from "react";
import { generateSoap, generateSoapMarkdown, getGlucose } from "@/lib/api";
import Link from "next/link";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, ReferenceLine } from "recharts";
import { useIsDark } from "./ThemeToggle";

function isAnomalyPoint(v: number){ return v>250 || v<70; }

// Render [Xem log #id] as clickable link to /tracker?highlight=id
function renderWithLogLinks(text: string){
  if(!text) return text;
  const parts = String(text).split(/(\[Xem log #\d+\])/g);
  return parts.map((p,i)=>{
    const m = p.match(/\[Xem log #(\d+)\]/);
    if(m) return <Link key={i} data-testid="log-link" href={`/tracker?highlight=${m[1]}`} className="font-medium text-blue-700 underline hover:text-blue-900 dark:text-blue-300">{p}</Link>;
    return <span key={i}>{p}</span>;
  });
}

function AnomalyDot(props:any){
  const { cx, cy, payload } = props;
  if(!payload) return <g/>;
  // Server list first (covers trend windows), local spike check as fallback.
  const anomaly = payload.anomaly ?? isAnomalyPoint(payload.value);
  return <circle cx={cx} cy={cy} r={anomaly?5:3} fill={anomaly?"#ef4444":"#0ea5e9"} stroke="#fff" strokeWidth={1} />;
}

export default function PrevisitView({ patientId, title, subtitle }: { patientId: string; title?: string; subtitle?: string }){
  const [days,setDays]=useState(14);
  const [data,setData]=useState<any>(null);
  const [md,setMd]=useState("");
  const [glucose,setGlucose]=useState<any>(null);
  const [loading,setLoading]=useState(false);
  const [err,setErr]=useState("");
  const [sEdit,setSEdit]=useState<string | null>(null);
  const hasTriageSource = Array.isArray((glucose as any)?.logs)
    ? (glucose as any).logs.some((l: any) => String(l.notes || "").split(" | ").some((s: string) => s.trim() === "[AI Triage]"))
    : false;
  const gen = async (asMd=false)=>{
    if(!patientId){ setErr("Thiếu patient id"); return; }
    setLoading(true); setErr(""); setData(null); setMd("");
    try{
      const g = await getGlucose(patientId,200,days).catch(()=>null);
      setGlucose(g);
      if(asMd){ const text=await generateSoapMarkdown(patientId,days); setMd(text); }
      else { const j=await generateSoap(patientId,days); setData(j); setSEdit(j?.soap?.subjective ?? null); }
    }catch(e:any){ setErr(e.message); } finally{ setLoading(false); }
  };
  const downloadMd=()=>{
    if(!md) return;
    const blob=new Blob([md],{type:"text/markdown"});
    const url=URL.createObjectURL(blob);
    const a=document.createElement("a"); a.href=url; a.download=`SOAP_${patientId}_${days}d.md`; a.click(); URL.revokeObjectURL(url);
  };
  const anomalyIds: Set<number> = new Set(Array.isArray((glucose as any)?.anomaly_ids) ? (glucose as any).anomaly_ids : []);
  const chartData = glucose?.logs? [...glucose.logs].reverse().slice(-90).map((l:any)=>({ time:String(l.measured_at).slice(5,16).replace("T"," "), value:l.value_mgdl, id:l.id, anomaly: anomalyIds.size>0 ? anomalyIds.has(l.id) : isAnomalyPoint(l.value_mgdl) })):[];
  const anomalySource = anomalyIds.size>0 ? "server" : "local";
  const dark = useIsDark();
  const gridStroke = dark ? "#334155" : "#e2e8f0";
  const lineStroke = dark ? "#60a5fa" : "#0ea5e9";
  return (
    <div className="space-y-6">
      <div><h1 className="text-xl font-bold">{title || "Hồ sơ trước tái khám (SOAP)"}</h1>{subtitle && <p className="text-sm text-slate-600 dark:text-slate-300">{subtitle}</p>}</div>
      <div className="rounded-2xl border bg-white p-5 shadow-sm dark:bg-slate-900 dark:border-slate-700">
        <div className="flex flex-wrap items-end gap-3">
          <div className="text-xs">Patient: {patientId.slice(0,8)}</div>
          <div><label className="text-xs">Số ngày</label><select value={days} onChange={(e)=>setDays(Number(e.target.value))} className="mt-1 rounded-lg border px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-700 min-h-[44px]"><option value={7}>7 ngày</option><option value={14}>14 ngày</option><option value={30}>30 ngày</option><option value={90}>90 ngày</option></select></div>
          <button data-testid="soap-json-btn" onClick={()=>gen(false)} disabled={loading} className="rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50 min-h-[44px]">{loading?"Đang tạo...":"Tạo SOAP (JSON)"}</button>
          <button onClick={()=>gen(true)} disabled={loading} className="rounded-lg border bg-white px-5 py-2.5 text-sm hover:bg-slate-50 dark:bg-slate-900 dark:border-slate-700 min-h-[44px]">Tạo Markdown</button>
        </div>
        {err && <div role="alert" className="mt-3 rounded-lg bg-red-50 border border-red-200 p-3 text-sm text-red-700 dark:bg-red-950 dark:text-red-200">{err}</div>}
      </div>
      {data && (
        <div className="rounded-2xl border bg-white p-5 shadow-sm dark:bg-slate-900 dark:border-slate-700">
          <div className="flex items-center justify-between"><h2 className="text-sm font-semibold">Kết quả SOAP — {data.user_id} · {data.period}</h2><div className="text-[11px] text-slate-500 dark:text-slate-400">{new Date(data.generated_at).toLocaleString("vi-VN")}</div></div>
          <div className="mt-3 grid gap-4 lg:grid-cols-10 max-w-6xl">
            <div className="rounded-xl border bg-white p-4 shadow-sm lg:col-span-3 dark:bg-slate-900 dark:border-slate-700">
              <div className="text-xs font-bold">Sparkline {days} ngày (30%)</div>
              <div className="mt-2 h-56">
                {chartData.length? <ResponsiveContainer width="100%" height="100%"><LineChart data={chartData}><CartesianGrid strokeDasharray="3 3" stroke={gridStroke} /><XAxis dataKey="time" tick={{fontSize:9}} interval="preserveStartEnd" stroke={gridStroke} /><YAxis domain={[40,320]} tick={{fontSize:10}} stroke={gridStroke} /><Tooltip contentStyle={dark?{backgroundColor:"#0f172a",borderColor:"#334155",color:"#e2e8f0"}:undefined} /><ReferenceLine y={126} stroke="#f59e0b" strokeDasharray="4 4" /><ReferenceLine y={200} stroke="#ef4444" strokeDasharray="4 4" /><ReferenceLine y={70} stroke="#f97316" strokeDasharray="4 4" /><Line type="monotone" dataKey="value" stroke={lineStroke} strokeWidth={1.5} dot={<AnomalyDot />} /></LineChart></ResponsiveContainer> : <div className="flex h-full items-center justify-center text-sm text-slate-500">Chưa có dữ liệu</div>}
              </div>
              <div className="mt-2 text-[11px] text-slate-500 dark:text-slate-400">Chấm đỏ = anomaly ({anomalySource === "server" ? "theo server: spike + trend cascade" : "spike >250 / <70"}). Click [Xem log #id] để xem log thô.</div>
            </div>
            <div className="grid gap-4 sm:grid-cols-2 lg:col-span-7">
              <div className="rounded-xl bg-slate-50 p-4 dark:bg-slate-800"><div className="flex items-center gap-2 text-xs font-bold">S — Subjective{hasTriageSource && <span className="rounded-full border border-violet-300 bg-violet-100 px-2 py-0.5 text-[10px] font-medium text-violet-800 dark:bg-violet-950 dark:text-violet-200">[Nguồn: AI Triage]</span>}</div><div className="mt-1 text-sm whitespace-pre-wrap">{renderWithLogLinks(data.soap.subjective)}</div><textarea aria-label="S-Subjective (chỉnh sửa)" value={sEdit ?? ""} onChange={(e)=>setSEdit(e.target.value)} rows={4} className="mt-2 w-full rounded-lg border bg-white p-2 text-sm whitespace-pre-wrap dark:bg-slate-900 dark:border-slate-700 min-h-[44px]" /></div>
              <div className="rounded-xl bg-blue-50 p-4 border border-blue-100 dark:bg-blue-950 dark:border-blue-900"><div className="text-xs font-bold text-blue-900 dark:text-blue-200">O — Objective</div><div className="mt-1 text-sm whitespace-pre-wrap">{renderWithLogLinks(data.soap.objective)}</div></div>
              <div className="rounded-xl bg-amber-50 p-4 border border-amber-100 dark:bg-amber-950 dark:border-amber-900"><div className="text-xs font-bold">A — Assessment</div><div className="mt-1 text-sm whitespace-pre-wrap">{renderWithLogLinks(data.soap.assessment)}</div></div>
              <div className="rounded-xl bg-emerald-50 p-4 border border-emerald-100 dark:bg-emerald-950 dark:border-emerald-900"><div className="text-xs font-bold">P — Plan</div>
                {data.soap.plan? <div className="mt-1 text-sm whitespace-pre-wrap">{data.soap.plan}</div>
                : <div className="mt-1 text-sm italic text-slate-400">Dành cho bác sĩ chỉ định</div>}
              </div>
            </div>
          </div>
          <div className="mt-4 rounded-lg bg-slate-900 p-3 text-xs text-slate-100 overflow-auto"><pre className="whitespace-pre-wrap break-words">{JSON.stringify(data,null,2)}</pre></div>
        </div>
      )}
      {md && <div className="rounded-2xl border bg-white p-5 shadow-sm"><div className="flex items-center justify-between"><h2 className="text-sm font-semibold">Markdown</h2><button onClick={downloadMd} className="rounded-full bg-slate-900 px-4 py-2 text-xs text-white">Tải .md</button></div><pre className="mt-3 overflow-auto rounded-lg bg-slate-50 p-4 text-xs whitespace-pre-wrap border">{md}</pre></div>}
      {!data && !md && !loading && <div className="rounded-2xl border border-dashed bg-slate-50 p-8 text-center text-sm text-slate-600">Nhập logs ở Tracker trước rồi tạo SOAP.</div>}
    </div>
  );
}
