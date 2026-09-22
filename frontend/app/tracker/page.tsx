"use client";
import { useEffect, useState } from "react";
import { logGlucose, getGlucose, queryRag, saveGlucoseFollowup } from "@/lib/api";
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

const CRITICAL_TEXT = "CẢNH BÁO: Đường huyết nguy hiểm. Vui lòng liên hệ bác sĩ hoặc cấp cứu ngay lập tức.";
const GUARD_RE = /thiết kế riêng để theo dõi đường huyết|nằm ngoài phạm vi hỗ trợ/i;

const EntryAlert = (p: any) => {
  const { value, setValue, context, setContext, notes, setNotes, msg, anomaly, isCritical, submit } = p;
  return (
    <div className="rounded-2xl border bg-white p-5 shadow-sm lg:col-span-1">
      <h2 className="text-sm font-semibold">Nhập chỉ số</h2>
      {isCritical && (
        <div className="mt-4 rounded-xl border-2 border-red-600 bg-red-600 p-4 text-white shadow-xl">
          <div className="font-bold text-sm">🚨 {CRITICAL_TEXT}</div>
          {anomaly?.reason && <div className="mt-2 text-xs leading-relaxed">{anomaly.reason}</div>}
          {msg && <div className="mt-2 rounded-lg bg-white/10 p-2 text-xs">{msg}</div>}
          <div className="mt-2 text-xs text-red-100">Chat và diễn giải xu hướng đã tạm khóa để ưu tiên an toàn. Nhập một chỉ số an toàn để mở khóa.</div>
        </div>
      )}
      <div className="relative mt-4 space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div><label className="text-xs text-slate-600">Giá trị (mg/dL)</label><input type="number" value={value} onChange={(e)=>setValue(e.target.value)} className="mt-1 w-full rounded-lg border px-3 py-2 text-sm" /></div>
          <div><label className="text-xs text-slate-600">Bối cảnh</label><select value={context} onChange={(e)=>setContext(e.target.value)} className="mt-1 w-full rounded-lg border px-3 py-2 text-sm">{contexts.map(c=> <option key={c.value} value={c.value}>{c.label}</option>)}</select></div>
        </div>
        <div><label className="text-xs text-slate-600">Ghi chú</label><input value={notes} onChange={(e)=>setNotes(e.target.value)} placeholder="vd: sau ăn phở" className="mt-1 w-full rounded-lg border px-3 py-2 text-sm" /></div>
        <button onClick={submit} className="w-full rounded-lg bg-emerald-600 py-2.5 text-sm font-medium text-white hover:bg-emerald-700">Lưu chỉ số</button>
        {msg && <div className="rounded-lg border bg-slate-50 p-3 text-xs">{msg}</div>}
        {anomaly && anomaly.type !== "none" && !isCritical && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800">
            <div className="font-bold">📈 Trend cascade — {anomaly.direction||""}</div>
            <div className="mt-1">{anomaly.reason}</div>
          </div>
        )}
        <div className="rounded-lg bg-blue-50 p-3 text-xs text-blue-800 border border-blue-100">Ngưỡng WHO/ADA: Đói &lt;100, 100–125, ≥126 · Sau ăn &lt;140, 140–199, ≥200 · Hạ &lt;70 · Critical ≥300</div>
      </div>
    </div>
  );
};

const FqgChat = (p: any) => {
  const { isCritical, fqg, effectiveId, lastLogId } = p;
  const [messages, setMessages] = useState<any[]>([]);
  const [input, setInput] = useState("");
  const [open, setOpen] = useState(false);
  const [fqgMode, setFqgMode] = useState(false);
  const [sending, setSending] = useState(false);
  useEffect(()=>{
    if (fqg.length && !isCritical) {
      setFqgMode(true);
      setOpen(true);
      setMessages((prev)=>{
        const fresh = fqg.filter((q: string)=>!prev.some((m)=>m.text===q));
        return [...prev, ...fresh.map((q: string)=>({ role: "ai", text: q }))];
      });
    }
  },[fqg, isCritical]);
  if (isCritical) return null;
  const send = async ()=>{
    const text = input.trim();
    if (!text || sending) return;
    if (!effectiveId) { setMessages((prev)=>[...prev, { role: "sys", text: "Cần đăng nhập" }]); return; }
    setInput("");
    setOpen(true);
    setMessages((prev)=>[...prev, { role: "user", text }]);
    setSending(true);
    try {
      const res: any = await queryRag(text, 5, effectiveId);
      const answer: string = res?.answer || "";
      if (res?.safeguard === true || GUARD_RE.test(answer)) {
        setMessages((prev)=>[...prev, { role: "guard", text: answer || "Hệ thống này được thiết kế riêng để theo dõi đường huyết. Vấn đề này nằm ngoài phạm vi hỗ trợ và an toàn y khoa. Vui lòng tham vấn bác sĩ tại buổi khám tới." }]);
        return;
      }
      const short = answer.length > 500 ? answer.slice(0, 500) + "…" : answer;
      if (short) setMessages((prev)=>[...prev, { role: "ai", text: short }]);
      if (fqgMode) {
        try {
          await saveGlucoseFollowup({ user_id: effectiveId, text, related_log_id: lastLogId });
          setMessages((prev)=>[...prev, { role: "sys", text: "Đã lưu ngữ cảnh." }]);
        } catch (e: any) {
          setMessages((prev)=>[...prev, { role: "sys", text: "Không lưu được ngữ cảnh: " + e.message }]);
        }
        setFqgMode(false);
      }
    } catch (e: any) {
      setMessages((prev)=>[...prev, { role: "sys", text: "Lỗi: " + e.message }]);
    } finally {
      setSending(false);
    }
  };
  if (!open && !fqg.length) return null;
  return (
    <div className="rounded-2xl border border-blue-100 bg-blue-50/50 p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Hỏi thêm ngữ cảnh (FQG)</h3>
        <button onClick={()=>setOpen((o)=>!o)} className="text-xs text-blue-700 underline">{open?"Thu gọn":"Mở chat"}</button>
      </div>
      {open && (
        <>
          <div className="mt-3 max-h-64 space-y-2 overflow-auto rounded-lg border bg-white p-3">
            {messages.length ? messages.map((m,i)=>(
              <div key={i} className={`rounded-lg p-2 text-xs leading-relaxed ${m.role==="user"?"ml-8 bg-emerald-50 border border-emerald-100 text-emerald-900":m.role==="guard"?"bg-red-50 border border-red-200 text-red-800":m.role==="sys"?"bg-slate-100 text-slate-600":"mr-8 bg-blue-50 border border-blue-100 text-blue-900"}`}>
                {m.role==="guard" && <div className="font-bold">⛔ Ngoài phạm vi hỗ trợ</div>}
                <div className="whitespace-pre-wrap">{m.text}</div>
              </div>
            )) : <div className="p-2 text-xs text-slate-500">AI sẽ hỏi thêm khi phát hiện trend bất thường.</div>}
          </div>
          <div className="mt-2 flex gap-2">
            <input value={input} onChange={(e)=>setInput(e.target.value)} onKeyDown={(e)=>{ if(e.key==="Enter") send(); }} placeholder="vd: ăn 2 miếng bánh ngọt" className="w-full rounded-lg border px-3 py-2 text-sm" />
            <button onClick={send} disabled={sending} className="rounded-lg bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-50">{sending?"...":"Gửi"}</button>
          </div>
        </>
      )}
    </div>
  );
};

const TRIAGE_API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const TriageWidget = (p: any) => {
  const { effectiveId, hotline } = p;
  const hotlineNum: string | undefined = hotline;
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [msgs, setMsgs] = useState<any[]>([]);
  const [sending, setSending] = useState(false);
  const [locked, setLocked] = useState(false);
  const send = async () => {
    const text = input.trim();
    if (!text || sending) return;
    setInput(""); setOpen(true);
    setMsgs((prev) => [...prev, { role: "user", text }]);
    setSending(true);
    try {
      const res = await fetch(`${TRIAGE_API}/v1/triage`, {
        method: "POST",
        credentials: "include" as const,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: effectiveId || undefined, message: text }),
      });
      const data = await res.json();
      if (data?.emergency) {
        setLocked(true);
        setMsgs((prev) => [...prev, { role: "emg", text: data.message || "Gọi 115 ngay." }]);
        return;
      }
      const bits: string[] = [];
      if (data?.suggested_specialty) bits.push(`Chuyên khoa gợi ý: ${data.suggested_specialty}`);
      (data?.recommended_doctors || []).slice(0, 3).forEach((d: any) => {
        const s = (d.slots || []).slice(0, 3).map((x: any) => `${x.weekday} ${x.date?.slice(5)} ${x.time}`).join("; ");
        bits.push(`${d.name} (${d.specialty}) — ${s || "hết slot tuần này"}`);
      });
      if (data?.followup_question) bits.push(`❓ ${data.followup_question}`);
      if (data?.solver_used) bits.push(`⚙️ Solver: ${data.solver_used}${data?.routing_reason ? ` — ${data.routing_reason}` : ""}`);
      if (data?.simulation_summary) bits.push(`📊 GA: ${data.simulation_summary.reassigned} xếp lại / ${data.simulation_summary.unplaced} chưa xếp được, fitness=${data.simulation_summary.fitness}`);
      setMsgs((prev) => [...prev, { role: "ai", text: bits.join("\n") || "Đã ghi nhận." }]);
    } catch (e: any) {
      setMsgs((prev) => [...prev, { role: "sys", text: "Lỗi: " + e.message }]);
    } finally {
      setSending(false);
    }
  };
  return (
    <div className="rounded-2xl border bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">📅 Đặt lịch khám (AI triage)</h3>
        <button onClick={() => setOpen((o) => !o)} className="rounded-full bg-violet-600 px-4 py-2 text-xs text-white hover:bg-violet-700">{open ? "Thu gọn" : "Mở"}</button>
      </div>
      {locked && (
        <div className="mt-3 rounded-xl border-2 border-red-600 bg-red-600 p-4 text-sm font-bold text-white shadow-xl">
          🚨 CẢNH BÁO CẤP CỨU: Gọi 115 ngay lập tức. Chat đặt lịch đã khóa.
          <div className="mt-3 flex flex-wrap gap-2">
            <a href="tel:115" className="rounded-full bg-white px-5 py-2.5 text-sm font-bold text-red-700 hover:bg-red-50">📞 Gọi 115</a>
            {hotlineNum && <a href={`tel:${hotlineNum}`} className="rounded-full border border-white px-5 py-2.5 text-sm text-white hover:bg-white/10">🏥 Hotline viện: {hotlineNum}</a>}
          </div>
        </div>
      )}
      {open && !locked && (
        <>
          <div className="mt-3 max-h-64 space-y-2 overflow-auto rounded-lg border bg-slate-50 p-3">
            {msgs.length ? msgs.map((m, i) => (
              <div key={i} className={`rounded-lg p-2 text-xs leading-relaxed whitespace-pre-wrap ${m.role === "user" ? "ml-8 border border-emerald-100 bg-emerald-50 text-emerald-900" : m.role === "emg" ? "border border-red-300 bg-red-50 text-red-800 font-bold" : "mr-8 border border-violet-100 bg-violet-50 text-violet-900"}`}>{m.text}</div>
            )) : <div className="p-2 text-xs text-slate-500">Mô tả triệu chứng + thời gian muốn khám, vd: “tê chân, mắt mờ, tuần sau trừ T4”.</div>}
          </div>
          <div className="mt-2 flex gap-2">
            <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") send(); }} placeholder="vd: tê chân, mắt mờ, tuần sau trừ T4" className="w-full rounded-lg border px-3 py-2 text-sm" />
            <button onClick={send} disabled={sending} className="rounded-lg bg-violet-600 px-4 py-2 text-sm text-white hover:bg-violet-700 disabled:opacity-50">{sending ? "..." : "Gửi"}</button>
          </div>
        </>
      )}
    </div>
  );
};

const MyTracker = (p: any) => {
  const { data, highlightId, trend, explainTrend, isCritical, anomalySet, hasServerIds } = p;
  const chartData = data?.logs ? [...data.logs].reverse().map((l: any)=>({ id: l.id, time: l.measured_at.slice(5,16).replace("T"," "), value: l.value_mgdl })) : [];
  const isAnom = (l: any) => hasServerIds ? anomalySet.has(Number(l.id)) : (l.value_mgdl > 250 || l.value_mgdl < 70);
  const dot = (props: any) => {
    const hot = hasServerIds ? anomalySet.has(Number(props?.payload?.id)) : (props?.payload?.value > 250 || props?.payload?.value < 70);
    return <circle cx={props.cx} cy={props.cy} r={hot?5:3} fill={hot?"#ef4444":"#0ea5e9"} stroke="#fff" strokeWidth={1} />;
  };
  return (
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
        <div className="flex items-center justify-between"><h3 className="text-sm font-semibold">Biểu đồ 14 ngày</h3>{!isCritical && <button onClick={explainTrend} className="rounded-full bg-slate-900 px-4 py-2 text-xs text-white hover:bg-black">Diễn giải xu hướng (RAG)</button>}</div>
        <div className="mt-4 h-56">
          {chartData.length? <ResponsiveContainer width="100%" height="100%"><LineChart data={chartData}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="time" tick={{fontSize:10}} /><YAxis domain={[40,320]} tick={{fontSize:10}} /><Tooltip /><ReferenceLine y={126} stroke="#f59e0b" strokeDasharray="4 4" label={{value:"126",fontSize:10}} /><ReferenceLine y={200} stroke="#ef4444" strokeDasharray="4 4" label={{value:"200",fontSize:10}} /><ReferenceLine y={70} stroke="#f97316" strokeDasharray="4 4" /><Line type="monotone" dataKey="value" stroke="#0ea5e9" strokeWidth={2} dot={dot} /></LineChart></ResponsiveContainer> : <div className="flex h-full items-center justify-center text-sm text-slate-500">Chưa có dữ liệu</div>}
        </div>
        {data?.should_escalate && <div className="mt-3 rounded-lg bg-red-50 border border-red-200 p-3 text-xs text-red-800">⚠️ Khuyến nghị liên hệ bác sĩ</div>}
        {trend && <div className="mt-4 rounded-lg border bg-slate-50 p-4 text-xs leading-relaxed whitespace-pre-wrap">{trend}</div>}
      </div>
      <div className="rounded-2xl border bg-white p-5 shadow-sm">
        <h3 className="text-sm font-semibold">Lịch sử</h3>
        <div className="mt-3 max-h-72 overflow-auto divide-y rounded-lg border">
          {data?.logs?.length? data.logs.map((l:any)=>{
            const hl = String(l.id)===String(highlightId);
            const an = isAnom(l);
            return <div key={l.id} id={`log-${l.id}`} className={`flex items-center justify-between px-3 py-2 text-xs ${hl?"bg-yellow-100 ring-2 ring-yellow-400":""} ${an?"border-l-4 border-l-orange-500 bg-orange-50":""}`}><div><div className="font-mono">#{l.id} · {l.measured_at.slice(0,16).replace("T"," ")} · {l.value_mgdl} mg/dL</div><div className="text-slate-500">{l.context} {l.notes? `· ${l.notes}`:""}</div></div><span className={`rounded-full border px-2 py-1 text-[11px] font-medium ${classColor(l.classification)}`}>{l.classification}</span></div>;
          }): <div className="p-6 text-center text-sm text-slate-500">Chưa có log</div>}
        </div>
      </div>
    </div>
  );
};

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
  const [lastLogId,setLastLogId]=useState<number|undefined>(undefined);

  const effectiveId = user?.id || "";
  const searchParams = useSearchParams();
  const highlightId = searchParams?.get("highlight");
  const isCritical = anomaly?.type === "spike";
  const anomalySet = new Set(((Array.isArray(data?.anomaly_ids) ? data.anomaly_ids : []) as any[]).map((x)=>Number(x)));
  const hasServerIds = Array.isArray(data?.anomaly_ids);
  const refresh = async ()=>{
    if(!effectiveId) return;
    try{ const d=await getGlucose(effectiveId,50,14); setData(d);}catch(e:any){ setMsg(e.message); }
  };
  useEffect(()=>{ if(effectiveId) refresh(); },[effectiveId]);

  const submit = async ()=>{
    if(!user){ setMsg("Cần đăng nhập"); return; }
    setMsg(""); setTrend(""); setAnomaly(null); setFqg([]);
    try{
      const rec:any=await logGlucose({ user_id: effectiveId, value_mgdl:Number(value), context, notes: notes||undefined });
      setMsg(`${rec.classification.toUpperCase()}: ${rec.message}`);
      if(rec.anomaly && rec.anomaly.type && rec.anomaly.type!=="none") setAnomaly(rec.anomaly);
      if(typeof rec.id === "number") setLastLogId(rec.id);
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

  if(!user) return <div className="rounded-2xl border bg-white p-8 text-center">Cần đăng nhập để dùng Nhật ký. <Link href="/login" className="underline">Đăng nhập</Link> hoặc <Link href="/register" className="underline">Đăng ký</Link></div>;

  return (
    <div className="space-y-6">
      <div><h1 className="text-xl font-bold">Hướng A — Nhật ký đường huyết</h1><p className="text-sm text-slate-600">User: {user.username} ({user.role}) · KPI ≥3 lần/tuần. Chỉ escalate khi critical hoặc 3×high.</p></div>
      <div className="grid gap-6 lg:grid-cols-3">
        <EntryAlert value={value} setValue={setValue} context={context} setContext={setContext} notes={notes} setNotes={setNotes} msg={msg} anomaly={anomaly} isCritical={isCritical} submit={submit} />
        <MyTracker data={data} highlightId={highlightId} trend={trend} explainTrend={explainTrend} isCritical={isCritical} anomalySet={anomalySet} hasServerIds={hasServerIds} />
      </div>
      <FqgChat isCritical={isCritical} fqg={fqg} effectiveId={effectiveId} lastLogId={lastLogId} />
      <TriageWidget effectiveId={effectiveId} />
    </div>
  );
}
function Kpi({label,value,sub,alert}:any){ return <div className={`rounded-xl border p-3 ${alert?"bg-amber-50 border-amber-200":"bg-white"}`}><div className="text-[11px] text-slate-500">{label}</div><div className="text-sm font-bold">{value}</div>{sub && <div className={`text-[11px] ${alert?"text-amber-700":"text-slate-500"}`}>{sub}</div>}</div>; }
