"use client";
import { useState } from "react";
import { generateSoap, generateSoapMarkdown, getGlucose } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import Link from "next/link";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, ReferenceLine } from "recharts";

function isAnomalyPoint(v: number){ return v>250 || v<70; }

// Render [Xem log #id] as clickable link to /tracker?highlight=id
function renderWithLogLinks(text: string){
  if(!text) return text;
  const parts = String(text).split(/(\[Xem log #\d+\])/g);
  return parts.map((p,i)=>{
    const m = p.match(/\[Xem log #(\d+)\]/);
    if(m) return <Link key={i} href={`/tracker?highlight=${m[1]}`} className="font-medium text-blue-700 underline hover:text-blue-900">{p}</Link>;
    return <span key={i}>{p}</span>;
  });
}

function AnomalyDot(props:any){
  const { cx, cy, payload } = props;
  if(!payload) return <g/>;
  const anomaly = isAnomalyPoint(payload.value);
  return <circle cx={cx} cy={cy} r={anomaly?5:3} fill={anomaly?"#ef4444":"#0ea5e9"} stroke="#fff" strokeWidth={1} />;
}

export default function PrevisitPage(){
  const { user } = useAuth();
  const [days,setDays]=useState(14);
  const [data,setData]=useState<any>(null);
  const [md,setMd]=useState("");
  const [glucose,setGlucose]=useState<any>(null);
  const [loading,setLoading]=useState(false);
  const [err,setErr]=useState("");
  const effectiveId = user?.id || "";
  const gen = async (asMd=false)=>{
    if(!user){ setErr("Cần đăng nhập"); return; }
    setLoading(true); setErr(""); setData(null); setMd("");
    try{
      const g = await getGlucose(effectiveId,200,days).catch(()=>null);
      setGlucose(g);
      if(asMd){ const text=await generateSoapMarkdown(effectiveId,days); setMd(text); }
      else { const j=await generateSoap(effectiveId,days); setData(j); }
    }catch(e:any){ setErr(e.message); } finally{ setLoading(false); }
  };
  const downloadMd=()=>{
    if(!md) return;
    const blob=new Blob([md],{type:"text/markdown"});
    const url=URL.createObjectURL(blob);
    const a=document.createElement("a"); a.href=url; a.download=`SOAP_${effectiveId}_${days}d.md`; a.click(); URL.revokeObjectURL(url);
  };
  const chartData = glucose?.logs? [...glucose.logs].reverse().slice(-90).map((l:any)=>({ time:String(l.measured_at).slice(5,16).replace("T"," "), value:l.value_mgdl })):[];
  if(!user) return <div className="rounded-2xl border bg-white p-8 text-center">Cần đăng nhập <Link href="/login" className="underline">Đăng nhập</Link></div>;
  return (
    <div className="space-y-6">
      <div><h1 className="text-xl font-bold">Hướng B — Hồ sơ trước tái khám (SOAP)</h1><p className="text-sm text-slate-600">User {user.username} · B2B2C cho phòng khám.</p></div>
      <div className="rounded-2xl border bg-white p-5 shadow-sm">
        <div className="flex flex-wrap items-end gap-3">
          <div className="text-xs">User: {user.username} ({user.role})</div>
          <div><label className="text-xs">Số ngày</label><select value={days} onChange={(e)=>setDays(Number(e.target.value))} className="mt-1 rounded-lg border px-3 py-2 text-sm"><option value={7}>7 ngày</option><option value={14}>14 ngày</option><option value={30}>30 ngày</option><option value={90}>90 ngày</option></select></div>
          <button onClick={()=>gen(false)} disabled={loading} className="rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50">{loading?"Đang tạo...":"Tạo SOAP (JSON)"}</button>
          <button onClick={()=>gen(true)} disabled={loading} className="rounded-lg border bg-white px-5 py-2.5 text-sm hover:bg-slate-50">Tạo Markdown</button>
        </div>
        {err && <div className="mt-3 rounded-lg bg-red-50 border border-red-200 p-3 text-xs text-red-700">{err}</div>}
      </div>
      {data && (
        <div className="rounded-2xl border bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between"><h2 className="text-sm font-semibold">Kết quả SOAP — {data.user_id} · {data.period}</h2><div className="text-xs text-slate-500">{new Date(data.generated_at).toLocaleString("vi-VN")}</div></div>
          <div className="mt-3 grid gap-4 lg:grid-cols-10">
            <div className="rounded-xl border bg-white p-4 shadow-sm lg:col-span-3">
              <div className="text-xs font-bold">Sparkline {days} ngày (30%)</div>
              <div className="mt-2 h-56">
                {chartData.length? <ResponsiveContainer width="100%" height="100%"><LineChart data={chartData}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="time" tick={{fontSize:9}} interval="preserveStartEnd" /><YAxis domain={[40,320]} tick={{fontSize:10}} /><Tooltip /><ReferenceLine y={126} stroke="#f59e0b" strokeDasharray="4 4" /><ReferenceLine y={200} stroke="#ef4444" strokeDasharray="4 4" /><ReferenceLine y={70} stroke="#f97316" strokeDasharray="4 4" /><Line type="monotone" dataKey="value" stroke="#0ea5e9" strokeWidth={1.5} dot={<AnomalyDot />} /></LineChart></ResponsiveContainer> : <div className="flex h-full items-center justify-center text-xs text-slate-500">Chưa có dữ liệu</div>}
              </div>
              <div className="mt-2 text-[11px] text-slate-500">Chấm đỏ = anomaly (spike &gt;250 / &lt;70). Click [Xem log #id] để xem log thô.</div>
            </div>
            <div className="grid gap-4 sm:grid-cols-2 lg:col-span-7">
              <div className="rounded-xl bg-slate-50 p-4"><div className="text-xs font-bold">S — Subjective</div><div className="mt-1 text-xs whitespace-pre-wrap">{renderWithLogLinks(data.soap.subjective)}</div></div>
              <div className="rounded-xl bg-blue-50 p-4 border border-blue-100"><div className="text-xs font-bold text-blue-900">O — Objective</div><div className="mt-1 text-xs whitespace-pre-wrap">{renderWithLogLinks(data.soap.objective)}</div></div>
              <div className="rounded-xl bg-amber-50 p-4 border border-amber-100"><div className="text-xs font-bold">A — Assessment</div><div className="mt-1 text-xs whitespace-pre-wrap">{renderWithLogLinks(data.soap.assessment)}</div></div>
              <div className="rounded-xl bg-emerald-50 p-4 border border-emerald-100"><div className="text-xs font-bold">P — Plan</div>
                {data.soap.plan? <div className="mt-1 text-xs whitespace-pre-wrap">{data.soap.plan}</div>
                : <div className="mt-1 text-xs italic text-slate-400">Dành cho bác sĩ chỉ định</div>}
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
