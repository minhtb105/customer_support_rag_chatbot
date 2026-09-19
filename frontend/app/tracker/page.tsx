"use client";
import { useEffect, useState } from "react";
import { logGlucose, getGlucose, queryRag } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, ReferenceLine } from "recharts";
import Link from "next/link";
import { useSearchParams } from "next/navigation";

const contexts = [
  { value: "fasting", label: "Đói (fasting)" },
  { value: "post_meal_2h", label: "Sau ăn 2h" },
  { value: "random", label: "Ngẫu nhiên" },
  { value: "pre_meal", label: "Trước ăn" },
  { value: "bedtime", label: "Trước ngủ" },
];
function classColor(c: string){ if(c==="normal")return "bg-emerald-100 text-emerald-700 border-emerald-200"; if(c==="elevated")return "bg-amber-100 text-amber-700 border-amber-200"; if(c==="high")return "bg-red-100 text-red-700 border-red-200"; if(c==="critical")return "bg-red-600 text-white border-red-700"; if(c==="low")return "bg-orange-100 text-orange-700 border-orange-200"; return "bg-slate-100 text-slate-600";}

export default function TrackerPage(){
  const { user } = useAuth();
  const [value,setValue]=useState("110");
  const [context,setContext]=useState("fasting");
  const [notes,setNotes]=useState("");
  const [data,setData]=useState<any>(null);
  const [msg,setMsg]=useState("");
  const [trend,setTrend]=useState("");
  const [anomaly,setAnomaly]=useState<any>(null);
  const [fqg,setFqg]=useState<string[]>([]);

  const effectiveId = user?.id || "";
  const searchParams = useSearchParams();
  const highlightId = searchParams?.get("highlight");
  const refresh = async ()=>{
    if(!effectiveId) return;
    try{ const d=await getGlucose(effectiveId,50); setData(d);}catch(e:any){ setMsg(e.message); }
  };
  useEffect(()=>{ if(effectiveId) refresh(); },[effectiveId]);

  const submit = async ()=>{
    if(!user){ setMsg("Cần đăng nhập"); return; }
    setMsg(""); setTrend(""); setAnomaly(null); setFqg([]);
    try{
      const rec:any=await logGlucose({ user_id: effectiveId, value_mgdl:Number(value), context, notes: notes||undefined });
      setMsg(`${rec.classification.toUpperCase()}: ${rec.message}`);
      if(rec.anomaly && rec.anomaly.type && rec.anomaly.type!=="none") setAnomaly(rec.anomaly);
      if(Array.isArray(rec.follow_up_questions) && rec.follow_up_questions.length) setFqg(rec.follow_up_questions);
      await refresh();
    }catch(e:any){ setMsg(e.message); }
  };
  const explainTrend = async ()=>{
    if(!data?.logs?.length) return;
    setTrend("Đang phân tích xu hướng...");
    const last=data.logs.slice(0,5).map((l:any)=>`${l.measured_at.slice(0,16)} ${l.value_mgdl}mg/dL (${l.context}, ${l.classification})`).join("; ");
    const stats=data.stats;
    const q=`Tôi có các chỉ số đường huyết gần đây: ${last}. Trung bình 7 ngày ${stats.last_7_days_avg}mg/dL, tần suất ${stats.logs_per_week}/tuần, streak ${stats.streak_days} ngày. Hãy diễn giải xu hướng bằng tiếng Việt, dùng guideline WHO/ADA/BYT, chỉ escalate nếu critical hoặc 3 lần high liên tiếp.`;
    try{ const ans=await queryRag(q,5,effectiveId); setTrend(ans.status==="pending_review"? `⏳ Đang chờ duyệt (${ans.evaluation?.routed_role}): ${ans.answer.slice(0,300)}` : ans.answer); }catch(e:any){ setTrend("Lỗi RAG: "+e.message); }
  };
  const chartData=data?.logs? [...data.logs].reverse().slice(-20).map((l:any)=>({ time:l.measured_at.slice(5,16).replace("T"," "), value:l.value_mgdl, cls:l.classification })):[];

  if(!user) return <div className="rounded-2xl border bg-white p-8 text-center">Cần đăng nhập để dùng Nhật ký. <Link href="/login" className="underline">Đăng nhập</Link> hoặc <Link href="/register" className="underline">Đăng ký</Link></div>;

  return (
    <div className="space-y-6">
      <div><h1 className="text-xl font-bold">Hướng A — Nhật ký đường huyết</h1><p className="text-sm text-slate-600">User: {user.username} ({user.role}) · KPI ≥3 lần/tuần. Chỉ escalate khi critical hoặc 3×high.</p></div>
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="rounded-2xl border bg-white p-5 shadow-sm lg:col-span-1">
          <h2 className="text-sm font-semibold">Nhập chỉ số</h2>
          <div className="mt-4 space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div><label className="text-xs text-slate-600">Giá trị (mg/dL)</label><input type="number" value={value} onChange={(e)=>setValue(e.target.value)} className="mt-1 w-full rounded-lg border px-3 py-2 text-sm" /></div>
              <div><label className="text-xs text-slate-600">Bối cảnh</label><select value={context} onChange={(e)=>setContext(e.target.value)} className="mt-1 w-full rounded-lg border px-3 py-2 text-sm">{contexts.map(c=> <option key={c.value} value={c.value}>{c.label}</option>)}</select></div>
            </div>
            <div><label className="text-xs text-slate-600">Ghi chú</label><input value={notes} onChange={(e)=>setNotes(e.target.value)} placeholder="vd: sau ăn phở" className="mt-1 w-full rounded-lg border px-3 py-2 text-sm" /></div>
            <button onClick={submit} className="w-full rounded-lg bg-emerald-600 py-2.5 text-sm font-medium text-white hover:bg-emerald-700">Lưu chỉ số</button>
            {msg && <div className="rounded-lg border bg-slate-50 p-3 text-xs">{msg}</div>}
            {anomaly && (
              <div className={`rounded-lg border p-3 text-xs ${anomaly.type==="spike"?"bg-red-50 border-red-200 text-red-800":"bg-amber-50 border-amber-200 text-amber-800"}`}>
                <div className="font-bold">{anomaly.type==="spike"?"⚠️ Spike bất thường":"📈 Trend cascade"} — {anomaly.direction||""}</div>
                <div className="mt-1">{anomaly.reason}</div>
              </div>
            )}
            {fqg.length>0 && (
              <div className="rounded-lg border border-blue-100 bg-blue-50 p-3 text-xs text-blue-900">
                <div className="font-bold">Câu hỏi follow-up (FQG)</div>
                <ul className="mt-1 list-disc pl-4 space-y-1">{fqg.map((q,i)=><li key={i}>{q}</li>)}</ul>
              </div>
            )}
            <div className="rounded-lg bg-blue-50 p-3 text-xs text-blue-800 border border-blue-100">Ngưỡng WHO/ADA: Đói &lt;100, 100–125, ≥126 · Sau ăn &lt;140, 140–199, ≥200 · Hạ &lt;70 · Critical ≥300</div>
          </div>
        </div>
        <div className="space-y-6 lg:col-span-2">
          {data?.stats && (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <Kpi label="Tổng logs" value={data.stats.total_logs} />
              <Kpi label="Trung bình" value={data.stats.avg_mgdl? `${data.stats.avg_mgdl} mg/dL`:"—"} />
              <Kpi label="7 ngày" value={data.stats.last_7_days_avg? `${data.stats.last_7_days_avg} mg/dL`:"—"} />
              <Kpi label="Tần suất" value={`${data.stats.logs_per_week}/tuần`} sub={data.stats.logs_per_week<3? "Chưa đạt KPI":"Đạt KPI"} alert={data.stats.logs_per_week<3} />
            </div>
          )}
          <div className="rounded-2xl border bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between"><h3 className="text-sm font-semibold">Biểu đồ 20 lần gần nhất</h3><button onClick={explainTrend} className="rounded-full bg-slate-900 px-4 py-2 text-xs text-white hover:bg-black">Diễn giải xu hướng (RAG)</button></div>
            <div className="mt-4 h-56">
              {chartData.length? <ResponsiveContainer width="100%" height="100%"><LineChart data={chartData}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="time" tick={{fontSize:10}} /><YAxis domain={[40,320]} tick={{fontSize:10}} /><Tooltip /><ReferenceLine y={126} stroke="#f59e0b" strokeDasharray="4 4" label={{value:"126",fontSize:10}} /><ReferenceLine y={200} stroke="#ef4444" strokeDasharray="4 4" label={{value:"200",fontSize:10}} /><ReferenceLine y={70} stroke="#f97316" strokeDasharray="4 4" /><Line type="monotone" dataKey="value" stroke="#0ea5e9" strokeWidth={2} dot={{r:3}} /></LineChart></ResponsiveContainer> : <div className="flex h-full items-center justify-center text-sm text-slate-500">Chưa có dữ liệu</div>}
            </div>
            {data?.should_escalate && <div className="mt-3 rounded-lg bg-red-50 border border-red-200 p-3 text-xs text-red-800">⚠️ Khuyến nghị liên hệ bác sĩ</div>}
            {trend && <div className="mt-4 rounded-lg border bg-slate-50 p-4 text-xs leading-relaxed whitespace-pre-wrap">{trend}</div>}
          </div>
          <div className="rounded-2xl border bg-white p-5 shadow-sm">
            <h3 className="text-sm font-semibold">Lịch sử</h3>
            <div className="mt-3 max-h-72 overflow-auto divide-y rounded-lg border">
              {data?.logs?.length? data.logs.map((l:any)=><div key={l.id} id={`log-${l.id}`} className={`flex items-center justify-between px-3 py-2 text-xs ${String(l.id)===String(highlightId)?"bg-yellow-100 ring-2 ring-yellow-400":""}`}><div><div className="font-mono">#{l.id} · {l.measured_at.slice(0,16).replace("T"," ")} · {l.value_mgdl} mg/dL</div><div className="text-slate-500">{l.context} {l.notes? `· ${l.notes}`:""}</div></div><span className={`rounded-full border px-2 py-1 text-[11px] font-medium ${classColor(l.classification)}`}>{l.classification}</span></div>): <div className="p-6 text-center text-sm text-slate-500">Chưa có log</div>}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
function Kpi({label,value,sub,alert}:any){ return <div className={`rounded-xl border p-3 ${alert?"bg-amber-50 border-amber-200":"bg-white"}`}><div className="text-[11px] text-slate-500">{label}</div><div className="text-sm font-bold">{value}</div>{sub && <div className={`text-[11px] ${alert?"text-amber-700":"text-slate-500"}`}>{sub}</div>}</div>; }
