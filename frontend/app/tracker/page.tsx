"use client";
import { useEffect, useRef, useState } from "react";
import { logGlucose, getGlucose, queryRag, saveGlucoseFollowup, getDoctorPatients, fmtDT } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import StatusBadge from "@/components/StatusBadge";
import GlucoseChart from "@/components/GlucoseChart";
import { playServerAudio, loadPersona, vnToday } from "@/lib/voice";
import NutritionLookup from "@/components/NutritionLookup";
import { Siren, TrendingUp, ShieldAlert, CalendarPlus, Phone, Building2, TriangleAlert, Mic, Square, AudioLines } from "lucide-react";

const contexts = [
  { value: "fasting", label: "Lúc đói" },
  { value: "post_meal_2h", label: "Sau ăn 2h" },
  { value: "random", label: "Ngẫu nhiên" },
  { value: "pre_meal", label: "Trước ăn" },
  { value: "bedtime", label: "Trước ngủ" },
];
function classColor(c: string){ if(c==="normal")return "bg-emerald-100 text-emerald-700 border-emerald-200"; if(c==="elevated")return "bg-amber-100 text-amber-700 border-amber-200"; if(c==="high")return "bg-red-100 text-red-700 border-red-200"; if(c==="critical")return "bg-red-600 text-white border-red-700"; if(c==="low")return "bg-orange-100 text-orange-700 border-orange-200"; return "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300";}

const CRITICAL_TEXT = "CẢNH BÁO: Đường huyết nguy hiểm. Vui lòng liên hệ bác sĩ hoặc cấp cứu ngay lập tức.";
const GUARD_RE = /thiết kế riêng để theo dõi đường huyết|nằm ngoài phạm vi hỗ trợ/i;

const EntryAlert = (p: any) => {
  const { value, setValue, context, setContext, notes, setNotes, msg, anomaly, isCritical, submit } = p;
  return (
    <div id="manual-entry" className="rounded-2xl border bg-white p-5 shadow-sm lg:col-span-1 dark:bg-slate-900 dark:border-slate-700">
      <h2 className="text-sm font-semibold">Nhập chỉ số</h2>
      {isCritical && (
        <div data-testid="critical-banner" role="alert" className="animate-critical-pulse mt-4 rounded-xl border-2 border-red-600 bg-red-600 p-4 text-white shadow-xl">
          <div className="font-bold text-sm flex items-center gap-1.5"><Siren size={15} aria-hidden /> {CRITICAL_TEXT}</div>
          {anomaly?.reason && <div className="mt-2 text-xs leading-relaxed">{anomaly.reason}</div>}
          {msg && <div className="mt-2 rounded-lg bg-white/10 p-2 text-xs">{msg}</div>}
          <div className="mt-2 text-xs text-red-100">Chat và diễn giải xu hướng đã tạm khóa để ưu tiên an toàn. Nhập một chỉ số an toàn để mở khóa.</div>
        </div>
      )}
      <div className="relative mt-4 space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div><label htmlFor="glucose-value" className="text-xs text-slate-600 dark:text-slate-300">Giá trị (mg/dL)</label><input id="glucose-value" type="number" inputMode="decimal" value={value} onChange={(e)=>setValue(e.target.value)} placeholder="vd: 110" className="mt-1 w-full rounded-lg border px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-700" /></div>
          <div><label htmlFor="glucose-context" className="text-xs text-slate-600 dark:text-slate-300">Bối cảnh</label><select id="glucose-context" value={context} onChange={(e)=>setContext(e.target.value)} className="mt-1 w-full rounded-lg border px-3 py-2 text-sm min-h-[44px] dark:bg-slate-900 dark:border-slate-700">{contexts.map(c=> <option key={c.value} value={c.value}>{c.label}</option>)}</select></div>
        </div>
        <div><label htmlFor="glucose-notes" className="text-xs text-slate-600 dark:text-slate-300">Ghi chú</label><input id="glucose-notes" value={notes} onChange={(e)=>setNotes(e.target.value)} placeholder="vd: sau ăn phở" className="mt-1 w-full rounded-lg border px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-700 min-h-[44px]" /></div>
        <button data-testid="tracker-save" onClick={submit} className="w-full rounded-lg bg-emerald-600 py-2.5 text-sm font-medium text-white hover:bg-emerald-700 min-h-[44px] focus-visible:outline-2">Lưu chỉ số</button>
        {msg && <div role="status" className="rounded-lg border bg-slate-50 p-3 text-xs dark:bg-slate-800 dark:border-slate-700 dark:text-slate-200">{msg}</div>}
        {anomaly && anomaly.type !== "none" && !isCritical && (
          <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800 dark:bg-amber-950 dark:border-amber-800 dark:text-amber-100">
            <div className="font-bold flex items-center gap-1.5"><TrendingUp size={14} aria-hidden /> Cảnh báo chuỗi tăng — {anomaly.direction||""}</div>
            <div className="mt-1">{anomaly.reason}</div>
          </div>
        )}
        <div className="rounded-lg bg-blue-50 p-3 text-xs text-blue-800 border border-blue-100 dark:bg-blue-950 dark:text-blue-100 dark:border-blue-900">Ngưỡng WHO/ADA: Đói &lt;100, 100–125, ≥126 · Sau ăn &lt;140, 140–199, ≥200 · Hạ &lt;70 · Critical ≥300</div>
      </div>
    </div>
  );
};

// Voice-mode hands-free loop tuning (constants pinned by debate D2 / plan B2).
const GAP_MS = 600;
const TURN_WINDOW_MS = 7000;
const MAX_VOICE_TURNS = 20;
const MAX_EMPTY_STREAK = 2;
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

const FqgChat = (p: any) => {
  const { isCritical, fqg, effectiveId, lastLogId, onAnomaly, persona } = p;
  const [messages, setMessages] = useState<any[]>([]);
  const [input, setInput] = useState("");
  const [open, setOpen] = useState(false);
  const [fqgMode, setFqgMode] = useState(false);
  const [sending, setSending] = useState(false);
  const [recording, setRecording] = useState(false);
  const [voicePending, setVoicePending] = useState<any>(null);
  const mediaRef = useRef<any>(null);
  const chunksRef = useRef<Blob[]>([]);
  // === Voice-mode loop state (P1) — delimited block, liftable to a hook later ===
  const [voiceLoop, setVoiceLoop] = useState(false);
  const [countdown, setCountdown] = useState<number | null>(null);
  const loopRef = useRef(false);
  const cancelledRef = useRef(false);
  const streamRef = useRef<any>(null);
  const turnTimerRef = useRef<any>(null);
  const turnCountRef = useRef(0);
  const emptyStreakRef = useRef(0);
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
  // D4 lifecycle: critical flip and unmount must abort the loop (release mic/timers).
  useEffect(()=>{ if (isCritical) stopLoop(); },[isCritical]);
  useEffect(()=>()=>{ stopLoop(); },[]);
  if (isCritical) return null;
  const send = async (textOverride?: string)=>{
    const text = (textOverride ?? input).trim();
    if (!text || sending) return;
    if (!effectiveId) { setMessages((prev)=>[...prev, { role: "sys", text: "Cần đăng nhập" }]); return; }
    if (textOverride === undefined) setInput("");
    setOpen(true);
    setMessages((prev)=>[...prev, { role: "user", text }]);
    setSending(true);
    try {
      const res: any = await queryRag(text, 5, effectiveId);
      const answer: string = res?.answer || "";
      if (res?.safeguard === true || GUARD_RE.test(answer)) {
        setMessages((prev)=>[...prev, { role: "guard", text: answer || "Hệ thống này được thiết kế riêng để theo dõi đường huyết. Vấn đề này nằm ngoài phạm vi hỗ trợ và an toàn y khoa. Vui lòng tham vấn bác sĩ tại buổi khám tới." }]);
        return "";
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
      return short;
    } catch (e: any) {
      setMessages((prev)=>[...prev, { role: "sys", text: "Lỗi: " + e.message }]);
      return "";
    } finally {
      setSending(false);
    }
  };
  const transcribeBlob = async (blob: Blob)=>{
    const fd = new FormData();
    fd.append("audio", blob, "voice.webm");
    const res = await fetch(`${TRIAGE_API}/v1/voice/transcribe`, { method: "POST", credentials: "include" as const, body: fd });
    if (!res.ok) throw new Error(`transcribe ${res.status}`);
    return res.json();
  };
  // D3 outcome contract: the loop branches on this; single-turn mic ignores the return.
  const handleVoiceBlob = async (blob: Blob): Promise<"emergency"|"confirm"|"answered"|"empty"|"error"> => {
    if (!effectiveId) { setMessages((prev)=>[...prev, { role: "sys", text: "Cần đăng nhập" }]); return "error"; }
    setOpen(true);
    let data: any;
    try {
      data = await transcribeBlob(blob);
    } catch (e: any) {
      setMessages((prev)=>[...prev, { role: "sys", text: "Không nghe rõ. Bác gõ giúp con nhé." }]);
      return "error";
    }
    if (cancelledRef.current) return "error";
    const transcript = (data?.text || "").trim();
    if (data?.emergency === true) {
      const emgText = data?.message || "Gọi 115 ngay.";
      setMessages((prev)=>[...prev, { role: "emg", text: emgText }]);
      await playServerAudio(emgText, { persona });
      return "emergency";
    }
    if (data?.parse && !data.parse.ambiguous) {
      if (transcript) setMessages((prev)=>[...prev, { role: "user", text: transcript }]);
      setVoicePending({ parse: data.parse, transcript });
      const tts = await playServerAudio(data.parse.confirm_text || "", { persona });
      if (cancelledRef.current) return "error";
      return tts === "error" ? "error" : "confirm";
    }
    if (!transcript) {
      setMessages((prev)=>[...prev, { role: "sys", text: "Con nghe chưa rõ. Bác bấm mic nói lại giúp con nhé." }]);
      return "empty";
    }
    const answer = await send(transcript);
    if (cancelledRef.current) return "error";
    if (answer) {
      const tts = await playServerAudio(answer, { persona });
      if (cancelledRef.current) return "error";
      if (tts === "error") return "error"; // fail-open: answer text already shown; loop stops
    }
    return "answered";
  };
  const toggleMic = async ()=>{
    if (recording) {
      try { mediaRef.current?.stop(); } catch { setRecording(false); }
      return;
    }
    setVoicePending(null);
    cancelledRef.current = false; // new single-turn recording: clear any stale loop cancel
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e: any)=>{ if (e.data?.size) chunksRef.current.push(e.data); };
      mr.onstop = async ()=>{
        setRecording(false);
        try { stream.getTracks().forEach((t: any)=>t.stop()); } catch { /* noop */ }
        await handleVoiceBlob(new Blob(chunksRef.current, { type: "audio/webm" }));
      };
      mediaRef.current = mr;
      mr.start();
      setRecording(true);
    } catch {
      setOpen(true);
      setMessages((prev)=>[...prev, { role: "sys", text: "Không mở được mic. Bác gõ giúp con nhé." }]);
    }
  };
  const confirmVoiceSave = async ()=>{
    if (!voicePending?.parse || voicePending.parse.ambiguous) return;
    if (!effectiveId) { setMessages((prev)=>[...prev, { role: "sys", text: "Cần đăng nhập" }]); return; }
    try {
      const rec: any = await logGlucose({ user_id: effectiveId, value_mgdl: voicePending.parse.value, context: voicePending.parse.context });
      setMessages((prev)=>[...prev, { role: "sys", text: `Đã lưu ${rec.value_mgdl} mg/dL (${rec.context}).` }]);
      try { await (onAnomaly?.(rec.anomaly, rec.follow_up_questions, rec)); } catch { /* fail-open */ }
      setVoicePending(null);
      // Loop paused on the confirm card: resume listening once the user saved.
      if (loopRef.current && !cancelledRef.current) {
        await sleep(GAP_MS);
        if (loopRef.current && !cancelledRef.current) startListen();
      }
    } catch (e: any) {
      setMessages((prev)=>[...prev, { role: "sys", text: "Lỗi lưu: " + e.message }]);
    }
  };
  // === Voice-mode loop controller (P1) — single delimited block, liftable to a hook later ===
  // Idempotent teardown: stop timer, recorder, and release the mic stream.
  // Hoisted function (not const): the [isCritical]/unmount effects below run even on
  // renders that early-return before this block, so `stopLoop` must exist already.
  function stopLoop(){
    loopRef.current = false;
    cancelledRef.current = true;
    if (turnTimerRef.current) { clearTimeout(turnTimerRef.current); turnTimerRef.current = null; }
    try { mediaRef.current?.stop(); } catch { /* noop */ }
    try { streamRef.current?.getTracks?.().forEach((t: any)=>t.stop()); } catch { /* noop */ }
    streamRef.current = null;
    setVoiceLoop(false);
    setCountdown(null);
  }
  // One listen turn on the kept stream: fixed window auto-stop + tap/Stop early finish.
  const startListen = ()=>{
    if (!loopRef.current || cancelledRef.current) return;
    const stream = streamRef.current;
    if (!stream) { stopLoop(); return; }
    let mr: any;
    try { mr = new MediaRecorder(stream); } catch { stopLoop(); return; }
    const chunks: Blob[] = [];
    mr.ondataavailable = (e: any)=>{ if (e.data?.size) chunks.push(e.data); };
    mr.onstop = async ()=>{
      if (turnTimerRef.current) { clearTimeout(turnTimerRef.current); turnTimerRef.current = null; }
      setCountdown(null);
      if (cancelledRef.current || !loopRef.current) return;
      const outcome = await handleVoiceBlob(new Blob(chunks, { type: "audio/webm" }));
      if (cancelledRef.current || !loopRef.current) return;
      if (outcome === "emergency" || outcome === "error") { stopLoop(); return; }
      if (outcome === "confirm") return; // PAUSE: resume from the confirm card's Đúng/Nói lại
      if (outcome === "empty") {
        emptyStreakRef.current += 1;
        if (emptyStreakRef.current >= MAX_EMPTY_STREAK) {
          setMessages((prev)=>[...prev, { role: "sys", text: "Không nghe rõ 2 lần liên tiếp. Con tắt voice mode." }]);
          stopLoop();
          return;
        }
        await sleep(GAP_MS);
        if (cancelledRef.current || !loopRef.current) return;
        startListen();
        return;
      }
      // answered
      emptyStreakRef.current = 0;
      turnCountRef.current += 1;
      if (turnCountRef.current >= MAX_VOICE_TURNS) {
        setMessages((prev)=>[...prev, { role: "sys", text: "Kết thúc voice mode (giới hạn 20 lượt)." }]);
        stopLoop();
        return;
      }
      await sleep(GAP_MS);
      if (cancelledRef.current || !loopRef.current) return;
      startListen();
    };
    mediaRef.current = mr;
    try { mr.start(); } catch { stopLoop(); return; }
    setCountdown(Math.ceil(TURN_WINDOW_MS / 1000));
    const startAt = Date.now();
    const tick = ()=>{
      if (!loopRef.current || cancelledRef.current) return;
      const remain = Math.ceil((TURN_WINDOW_MS - (Date.now() - startAt)) / 1000);
      if (remain <= 0) { try { mr.stop(); } catch { /* noop */ } return; }
      setCountdown(remain);
      turnTimerRef.current = setTimeout(tick, Math.min(1000, TURN_WINDOW_MS - (Date.now() - startAt)));
    };
    turnTimerRef.current = setTimeout(tick, 1000);
  };
  const startLoop = async ()=>{
    if (loopRef.current) return;
    setVoicePending(null);
    loopRef.current = true;
    cancelledRef.current = false;
    turnCountRef.current = 0;
    emptyStreakRef.current = 0;
    setOpen(true);
    setVoiceLoop(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (cancelledRef.current || !loopRef.current) { try { stream.getTracks().forEach((t: any)=>t.stop()); } catch { /* noop */ } return; }
      streamRef.current = stream;
    } catch {
      stopLoop();
      setMessages((prev)=>[...prev, { role: "sys", text: "Không mở được mic. Bác gõ giúp con nhé." }]);
      return;
    }
    startListen();
  };
  const toggleVoiceMode = ()=>{ if (loopRef.current) stopLoop(); else startLoop(); };
  if (!open && !fqg.length) return null;
  return (
    <div className="rounded-2xl border border-blue-100 bg-blue-50/50 p-4 shadow-sm dark:bg-blue-950/40 dark:border-blue-900">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <img src={recording || voiceLoop ? "/cocon-listen.png" : "/cocon.png"} alt="Cò Con" className="h-10 w-auto rounded-lg" />
          <h3 className="text-sm font-semibold">Hỏi thêm ngữ cảnh (FQG)</h3>
        </div>
        <button onClick={()=>setOpen((o)=>!o)} aria-expanded={open} className="text-xs text-blue-700 underline dark:text-blue-300 min-h-[44px] px-2">{open?"Thu gọn":"Mở chat"}</button>
      </div>
      {open && (
        <>
          <div className="mt-3 max-h-64 space-y-2 overflow-auto rounded-lg border bg-white p-3 dark:bg-slate-900 dark:border-slate-700">
            {messages.length ? messages.map((m,i)=>(
              m.role==="emg" ? (
                <div key={i} role="alert" className="rounded-lg border-2 border-red-600 bg-red-50 p-2 text-sm font-bold text-red-800 dark:bg-red-950 dark:border-red-500 dark:text-red-100">
                  <div className="whitespace-pre-wrap">{m.text}</div>
                  <a href="tel:115" aria-label="Gọi cấp cứu 115" className="mt-2 inline-flex min-h-[44px] items-center rounded-full bg-red-600 px-5 py-2 text-sm font-bold text-white hover:bg-red-700">Gọi 115</a>
                </div>
              ) : m.role==="ai" ? (
                <div key={i} className="flex items-start gap-1.5">
                  <img src="/cocon.png" alt="" aria-hidden className="mt-0.5 h-7 w-auto shrink-0 rounded-md" />
                  <div className="mr-8 flex-1 rounded-lg p-2 text-sm leading-relaxed bg-blue-50 border border-blue-100 text-blue-900 dark:bg-blue-950 dark:border-blue-800 dark:text-blue-100">
                    <div className="whitespace-pre-wrap">{m.text}</div>
                  </div>
                </div>
              ) : (
              <div key={i} className={`rounded-lg p-2 text-sm leading-relaxed ${m.role==="user"?"ml-8 bg-emerald-50 border border-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:border-emerald-800 dark:text-emerald-100":m.role==="guard"?"bg-red-50 border border-red-200 text-red-800 dark:bg-red-950 dark:border-red-800 dark:text-red-100":"bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300"}`}>
                {m.role==="guard" && <div className="font-bold flex items-center gap-1.5"><ShieldAlert size={14} aria-hidden /> Ngoài phạm vi hỗ trợ</div>}
                <div className="whitespace-pre-wrap">{m.text}</div>
              </div>
              )
            )) : <div className="p-2 text-sm text-slate-500 dark:text-slate-400">AI sẽ hỏi thêm khi phát hiện trend bất thường.</div>}
          </div>
          {voicePending?.parse && !voicePending.parse.ambiguous && (
            <div className="mt-3 rounded-xl border border-blue-200 bg-blue-50 p-4 dark:bg-blue-950 dark:border-blue-800">
              <div className="text-sm font-semibold text-blue-900 dark:text-blue-100">{voicePending.parse.confirm_text}</div>
              <div className="mt-3 flex gap-2">
                <button onClick={confirmVoiceSave} aria-label="Xác nhận đúng, lưu chỉ số" className="rounded-lg bg-emerald-600 px-6 py-3 text-sm font-bold text-white hover:bg-emerald-700 min-h-[44px]">Đúng, lưu</button>
                <button onClick={()=>{ setVoicePending(null); if (loopRef.current && !cancelledRef.current) startListen(); else toggleMic(); }} aria-label="Nói lại chỉ số" className="rounded-lg border border-blue-300 px-6 py-3 text-sm text-blue-900 dark:text-blue-100 dark:border-blue-700 min-h-[44px]">Nói lại</button>
              </div>
            </div>
          )}
          <div className="mt-2 flex gap-2">
            <input data-testid="fqg-input" value={input} onChange={(e)=>setInput(e.target.value)} onKeyDown={(e)=>{ if(e.key==="Enter") send(); }} placeholder="vd: ăn 2 miếng bánh ngọt" className="w-full rounded-lg border px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-700 min-h-[44px]" />
            <button data-testid="mic-btn" onClick={toggleMic} disabled={voiceLoop} aria-label={recording ? "Dừng ghi âm" : "Bắt đầu ghi âm bằng giọng nói"} title={recording ? "Dừng ghi âm" : "Nói thay vì gõ"} className={`rounded-lg px-4 py-2 text-sm min-h-[44px] min-w-[44px] flex items-center justify-center disabled:opacity-50 ${recording ? "bg-red-600 text-white animate-pulse" : "bg-slate-200 text-slate-700 hover:bg-slate-300 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"}`}>{recording ? <Square size={16} aria-hidden /> : <Mic size={16} aria-hidden />}</button>
            <button data-testid="voice-mode-btn" onClick={toggleVoiceMode} disabled={recording} aria-label={voiceLoop ? "Dừng voice mode" : "Bật chế độ nói liên tục"} title={voiceLoop ? "Dừng voice mode" : "Nói liên tục (hands-free)"} className={`rounded-lg px-4 py-2 text-sm min-h-[44px] min-w-[44px] flex items-center justify-center disabled:opacity-50 ${voiceLoop ? "bg-red-600 text-white" : "bg-blue-100 text-blue-700 hover:bg-blue-200 dark:bg-blue-900 dark:text-blue-100 dark:hover:bg-blue-800"}`}>{voiceLoop ? <Square size={16} aria-hidden /> : <AudioLines size={16} aria-hidden />}</button>
            <button data-testid="fqg-send" onClick={()=>send()} disabled={sending} className="rounded-lg bg-blue-600 px-4 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-50 min-h-[44px]">{sending?"...":"Gửi"}</button>
          </div>
          {voiceLoop && countdown !== null && (
            <div role="status" className="mt-2 text-xs text-red-700 dark:text-red-300">Đang nghe... (tự dừng sau {countdown}s)</div>
          )}
        </>
      )}
    </div>
  );
};

const TRIAGE_API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Round-2: map backend codes to patient-facing Vietnamese (display-only).
const SPECIALTY_VI: Record<string,string> = {
  Endocrinology_General: "Nội tiết chung",
  Endocrinology_Complication: "Biến chứng đái tháo đường",
  Nutrition_Lifestyle: "Dinh dưỡng & Lối sống",
};
const SOLVER_VI: Record<string,string> = {
  greedy: "nhanh",
  dfs: "xét kỹ ràng buộc",
  ga: "tối ưu nhiều bệnh nhân",
};
const TIER_VI: Record<string,string> = {
  T0: "T0 (tự động)",
  T1: "T1 (lễ tân)",
  T2: "T2 (dược sĩ/điều dưỡng)",
  T3: "T3 (bác sĩ)",
};

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
      if (data?.suggested_specialty) bits.push(`Chuyên khoa gợi ý: ${SPECIALTY_VI[data.suggested_specialty] || data.suggested_specialty}`);
      (data?.recommended_doctors || []).slice(0, 3).forEach((d: any) => {
        const s = (d.slots || []).slice(0, 3).map((x: any) => `${x.weekday} ${x.date?.slice(5)} ${x.time}`).join("; ");
        bits.push(`${d.name} (${d.specialty}) — ${s || "hết slot tuần này"}`);
      });
      if (data?.followup_question) bits.push(`Hỏi thêm: ${data.followup_question}`);
      if (data?.solver_used) bits.push(`Bộ giải: ${data.solver_used}${SOLVER_VI[data.solver_used] ? ` (${SOLVER_VI[data.solver_used]})` : ""}${data?.routing_reason ? ` — ${data.routing_reason}` : ""}`);
      if (data?.handoff_tier) bits.push(`Mức bàn giao: ${TIER_VI[data.handoff_tier] || data.handoff_tier}${data?.what_happens_next ? ` — ${data.what_happens_next}` : ""}`);
      if (data?.panic) bits.push(`Lab nguy hiểm: ${data?.message || "Gọi 115 ngay."}`);
      if (data?.simulation_summary) bits.push(`Tối ưu GA: ${data.simulation_summary.reassigned} xếp lại / ${data.simulation_summary.unplaced} chưa xếp được, điểm=${data.simulation_summary.fitness}`);
      setMsgs((prev) => [...prev, { role: "ai", text: bits.join("\n") || "Đã ghi nhận." }]);
    } catch (e: any) {
      setMsgs((prev) => [...prev, { role: "sys", text: "Lỗi: " + e.message }]);
    } finally {
      setSending(false);
    }
  };
  return (
    <div className="rounded-2xl border bg-white p-5 shadow-sm dark:bg-slate-900 dark:border-slate-700">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold flex items-center gap-1.5"><CalendarPlus size={15} aria-hidden /> Đặt lịch khám (AI triage)</h3>
        <button onClick={() => setOpen((o) => !o)} className="rounded-full bg-violet-600 px-4 py-2 text-xs text-white hover:bg-violet-700 min-h-[44px]">{open ? "Thu gọn" : "Mở"}</button>
      </div>
      {locked && (
        <div className="mt-3 rounded-xl border-2 border-red-600 bg-red-600 p-4 text-sm font-bold text-white shadow-xl">
          <span className="flex items-center gap-1.5"><Siren size={15} aria-hidden /> CẢNH BÁO CẤP CỨU: Gọi 115 ngay lập tức. Chat đặt lịch đã khóa.</span>
          <div className="mt-3 flex flex-wrap gap-2">
            <a href="tel:115" className="rounded-full bg-white px-5 py-2.5 text-sm font-bold text-red-700 hover:bg-red-50 inline-flex items-center gap-1.5 min-h-[44px]"><Phone size={15} aria-hidden /> Gọi 115</a>
            {hotlineNum && <a href={`tel:${hotlineNum}`} className="rounded-full border border-white px-5 py-2.5 text-sm text-white hover:bg-white/10 inline-flex items-center gap-1.5 min-h-[44px]"><Building2 size={15} aria-hidden /> Hotline viện: {hotlineNum}</a>}
          </div>
        </div>
      )}
      {open && !locked && (
        <>
          <div className="mt-3 max-h-64 space-y-2 overflow-auto rounded-lg border bg-slate-50 p-3 dark:bg-slate-800 dark:border-slate-700">
            {msgs.length ? msgs.map((m, i) => (
              <div key={i} className={`rounded-lg p-2 text-xs leading-relaxed whitespace-pre-wrap ${m.role === "user" ? "ml-8 border border-emerald-100 bg-emerald-50 text-emerald-900 dark:bg-emerald-950 dark:border-emerald-800 dark:text-emerald-100" : m.role === "emg" ? "border border-red-300 bg-red-50 text-red-800 font-bold dark:bg-red-950 dark:border-red-800 dark:text-red-100" : "mr-8 border border-violet-100 bg-violet-50 text-violet-900 dark:bg-violet-950 dark:border-violet-800 dark:text-violet-100"}`}>{m.text}</div>
            )) : <div className="p-2 text-xs text-slate-500 dark:text-slate-400">Mô tả triệu chứng + thời gian muốn khám, vd: “tê chân, mắt mờ, tuần sau trừ T4”.</div>}
          </div>
          <div className="mt-2 flex gap-2">
            <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") send(); }} placeholder="vd: tê chân, mắt mờ, tuần sau trừ T4" className="w-full rounded-lg border px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-700 min-h-[44px]" />
            <button onClick={send} disabled={sending} className="rounded-lg bg-violet-600 px-4 py-2 text-sm text-white hover:bg-violet-700 disabled:opacity-50 min-h-[44px]">{sending ? "..." : "Gửi"}</button>
          </div>
        </>
      )}
    </div>
  );
};

// Daily missing-metrics nag: once per day after login (D5/D6). Lives at page
// level, not in FqgChat (which is null when collapsed/critical). Persona-aware
// greeting TTS; anti-nag key persists in localStorage.
const DailyCheckBanner = (p: any) => {
  const { userId, persona } = p;
  const [daily, setDaily] = useState<any>(null);
  useEffect(() => {
    if (!userId || !persona) return;
    const key = `voice_daily_${userId}_${vnToday()}`;
    if (localStorage.getItem(key)) return;
    (async () => {
      try {
        const res = await fetch(`${TRIAGE_API}/v1/voice/daily-check?user_id=${encodeURIComponent(userId)}`, { credentials: "include" });
        if (!res.ok) return;
        const data = await res.json();
        localStorage.setItem(key, "1");
        if (data?.missing_slots?.length) {
          setDaily(data);
          playServerAudio(data.greeting || "", { persona });
        }
      } catch { /* fail-open: no nag on error */ }
    })();
  }, [userId, persona]);
  if (!daily?.missing_slots?.length) return null;
  return (
    <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800 dark:bg-amber-950 dark:border-amber-800 dark:text-amber-100">
      <div className="flex items-start justify-between gap-2">
        <span>{daily.greeting}</span>
        <a href="#manual-entry" className="shrink-0 px-2 py-2 text-xs underline min-h-[44px]">Tự điền</a>
      </div>
    </div>
  );
};

const MyTracker = (p: any) => {
  const { data, highlightId, trend, explainTrend, isCritical, anomalySet, hasServerIds, readOnly, loaded } = p;
  const points = data?.logs ? [...data.logs].reverse().map((l: any)=>({ time: l.measured_at.slice(5,16).replace("T"," "), value: l.value_mgdl, anomaly: hasServerIds ? anomalySet.has(Number(l.id)) : (l.value_mgdl > 250 || l.value_mgdl < 70) })) : [];
  const isAnom = (l: any) => hasServerIds ? anomalySet.has(Number(l.id)) : (l.value_mgdl > 250 || l.value_mgdl < 70);
  const vals = points.map((x: any)=>Number(x.value)).filter((v: number)=>Number.isFinite(v));
  const hotCount = points.filter((x: any)=>x.anomaly).length;
  const trendSummary = vals.length ? `Tóm tắt xu hướng 14 ngày: ${vals.length} chỉ số, cao nhất ${Math.max(...vals)} mg/dL, thấp nhất ${Math.min(...vals)} mg/dL, ${hotCount} điểm bất thường (hình tam giác trên biểu đồ).` : "Chưa có dữ liệu đường huyết 14 ngày.";
  if (!loaded && !data) {
    return (
      <div className="space-y-6 lg:col-span-2" aria-busy="true">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">{[0,1,2,3].map((i)=><div key={i} className="h-16 animate-pulse rounded-xl bg-slate-200 dark:bg-slate-700" />)}</div>
        <div className="rounded-2xl border bg-white p-5 shadow-sm dark:bg-slate-900 dark:border-slate-700"><div className="h-4 w-1/3 animate-pulse rounded bg-slate-200 dark:bg-slate-700" /><div className="mt-4 h-56 animate-pulse rounded bg-slate-200 dark:bg-slate-700" /></div>
        <div className="rounded-2xl border bg-white p-5 shadow-sm dark:bg-slate-900 dark:border-slate-700"><div className="h-4 w-1/4 animate-pulse rounded bg-slate-200 dark:bg-slate-700" /><div className="mt-3 space-y-2">{[0,1,2].map((i)=><div key={i} className="h-10 animate-pulse rounded bg-slate-200 dark:bg-slate-700" />)}</div></div>
      </div>
    );
  }
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
      <div className="rounded-2xl border bg-white p-5 shadow-sm dark:bg-slate-900 dark:border-slate-700">
        <div className="flex items-center justify-between"><h3 className="text-sm font-semibold">Biểu đồ 14 ngày</h3>{!isCritical && !readOnly && <button data-testid="trend-explain" onClick={explainTrend} className="rounded-full bg-slate-900 px-4 py-2 text-xs text-white hover:bg-black dark:bg-slate-100 dark:text-slate-900 min-h-[44px]">Diễn giải xu hướng (RAG)</button>}</div>
        <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-slate-500 dark:text-slate-400"><span className="inline-flex items-center gap-1"><span aria-hidden className="inline-block h-2 w-2 rounded-full bg-slate-400" /> Bình thường (chấm tròn)</span><span className="inline-flex items-center gap-1 text-red-600 dark:text-red-400"><span aria-hidden className="text-red-600">▲</span> Bất thường spike/trend (hình tam giác)</span><span>Ngưỡng 70 / 126 / 200</span></div>
        <div className="mt-4 h-56">
          <GlucoseChart points={points} emptyHint="Chưa có dữ liệu — nhập chỉ số đầu tiên ở khung bên trái." />
        </div>
        <div className="sr-only"><p>{trendSummary}</p><table><tbody>{points.map((x: any, i: number)=><tr key={i}><td>{x.time}</td><td>{x.value} mg/dL{x.anomaly ? " (bất thường)" : ""}</td></tr>)}</tbody></table></div>
        {data?.should_escalate && <div className="mt-3 rounded-lg bg-red-50 border border-red-200 p-3 text-xs text-red-800 dark:bg-red-950 dark:border-red-800 dark:text-red-100 flex items-center gap-1.5"><TriangleAlert size={14} aria-hidden /> Khuyến nghị liên hệ bác sĩ</div>}
        {trend && <div className="mt-4 rounded-lg border bg-slate-50 p-4 text-xs leading-relaxed whitespace-pre-wrap dark:bg-slate-800 dark:border-slate-700 dark:text-slate-200">{trend}</div>}
      </div>
      <div className="rounded-2xl border bg-white p-5 shadow-sm dark:bg-slate-900 dark:border-slate-700">
        <h3 className="text-sm font-semibold">Lịch sử</h3>
        <div className="mt-3 max-h-72 overflow-auto divide-y rounded-lg border dark:border-slate-700">
          {data?.logs?.length? data.logs.map((l:any)=>{
            const hl = String(l.id)===String(highlightId);
            const an = isAnom(l);
            return <div key={l.id} id={`log-${l.id}`} className={`flex items-center justify-between gap-2 px-3 py-2 text-sm ${hl?"bg-yellow-100 ring-2 ring-yellow-400 dark:bg-yellow-950":""} ${an?"border-l-4 border-l-orange-500 bg-orange-50 dark:bg-orange-950/40":""}`}><div className="min-w-0 flex-1"><div className="font-mono text-[11px]">#{l.id} · {fmtDT(l.measured_at)} · {l.value_mgdl} mg/dL</div><div className="text-slate-500 dark:text-slate-400 truncate">{l.context} {l.notes? `· ${l.notes}`:""}</div></div><span className="shrink-0"><StatusBadge level={l.classification} text={l.classification} /></span></div>;
          }): <div className="p-6 text-center text-sm text-slate-500 dark:text-slate-400">Chưa có log — dữ liệu sẽ hiện sau lần lưu đầu tiên.</div>}
        </div>
      </div>
    </div>
  );
};

export default function TrackerPage(){
  const { user, isExpert, isAdmin } = useAuth();
  const [value,setValue]=useState("");
  const [context,setContext]=useState("fasting");
  const [notes,setNotes]=useState("");
  const [data,setData]=useState<any>(null);
  const [msg,setMsg]=useState("");
  const [trend,setTrend]=useState("");
  const [anomaly,setAnomaly]=useState<any>(null);
  const [fqg,setFqg]=useState<string[]>([]);
  const [lastLogId,setLastLogId]=useState<number|undefined>(undefined);
  const [loaded,setLoaded]=useState(false);
  const [persona,setPersona]=useState("");

  const effectiveId = user?.id || "";
  const searchParams = useSearchParams();
  const highlightId = searchParams?.get("highlight");
  // Staff read-only patient view: ?patient=<user_id> only takes effect for expert/admin.
  // Patients ignore the param entirely (fail-closed); the API 403s cross-user reads anyway.
  const staffView = isExpert || isAdmin;
  const viewPatient = staffView ? (searchParams?.get("patient") || "") : "";
  const readOnly = !!viewPatient;
  const fetchId = viewPatient || effectiveId;
  // M3: tên bệnh nhân thay UUID trơ trụi (best-effort, fallback slice).
  const [viewPatientName, setViewPatientName] = useState("");
  useEffect(() => {
    if (!viewPatient) { setViewPatientName(""); return; }
    getDoctorPatients().then((d: any) => {
      const hit = (d.patients || []).find((p: any) => p.user_id === viewPatient);
      if (hit?.username) setViewPatientName(hit.username);
    }).catch(() => {});
  }, [viewPatient]);
  const isCritical = anomaly?.type === "spike";
  const anomalySet = new Set(((Array.isArray(data?.anomaly_ids) ? data.anomaly_ids : []) as any[]).map((x)=>Number(x)));
  const hasServerIds = Array.isArray(data?.anomaly_ids);
  const refresh = async ()=>{
    if(!fetchId) return;
    try{ const d=await getGlucose(fetchId,50,14); setData(d);}catch(e:any){ setMsg(e.message); }finally{ setLoaded(true); }
  };
  useEffect(()=>{ if(fetchId) refresh(); },[fetchId]);
  // Load selected voice persona once (feeds daily-check greeting + FqgChat TTS).
  useEffect(()=>{ if(effectiveId) loadPersona(effectiveId).then(setPersona); },[effectiveId]);

  const submit = async ()=>{
    if(!user){ setMsg("Cần đăng nhập"); return; }
    const v = Number(value);
    if(!value.trim() || !Number.isFinite(v)){ setMsg("Nhập chỉ số đường huyết trước khi lưu (vd: 110)."); return; }
    setMsg(""); setTrend(""); setAnomaly(null); setFqg([]);
    try{
      const rec:any=await logGlucose({ user_id: effectiveId, value_mgdl:v, context, notes: notes||undefined });
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
  const handleFqgAnomaly = async (anomaly: any, followUp: string[], rec?: any)=>{
    if (anomaly && anomaly.type && anomaly.type !== "none") setAnomaly(anomaly);
    if (rec && typeof rec.id === "number") setLastLogId(rec.id);
    if (rec?.classification && rec?.message) setMsg(`${String(rec.classification).toUpperCase()}: ${rec.message}`);
    if (Array.isArray(followUp) && followUp.length) setFqg(followUp);
    await refresh();
  };

  if(!user) return <div className="rounded-2xl border bg-white p-8 text-center dark:bg-slate-900 dark:border-slate-700">Cần đăng nhập để dùng Nhật ký. <Link href="/login" className="underline">Đăng nhập</Link> hoặc <Link href="/register" className="underline">Đăng ký</Link></div>;

  return (
    <div className="space-y-6">
      {readOnly && (
        <div className="rounded-xl border-2 border-blue-600 bg-blue-50 p-4 text-sm text-blue-900 dark:bg-blue-950 dark:border-blue-700 dark:text-blue-100">
          <span className="font-bold">Chế độ chỉ đọc</span> — bệnh nhân {viewPatientName || viewPatient.slice(0,8)}. Dữ liệu bên dưới là của bệnh nhân, không phải của bạn; mọi thao tác nhập/gửi đã ẩn.{" "}
          <Link href="/expert/patients" className="underline">← Danh sách bệnh nhân</Link>
        </div>
      )}
      <div><h1 className="text-xl font-bold">Hướng A — Nhật ký đường huyết</h1><p className="text-sm text-slate-600 dark:text-slate-300">{readOnly ? `Bệnh nhân ${viewPatientName || viewPatient.slice(0,8)} (chỉ đọc) · Chỉ escalate khi critical hoặc 3×high.` : `User: ${user.username} (${user.role}) · KPI ≥3 lần/tuần. Chỉ escalate khi critical hoặc 3×high.`}</p></div>
      <div className="grid gap-6 lg:grid-cols-3">
        {!readOnly && <EntryAlert value={value} setValue={setValue} context={context} setContext={setContext} notes={notes} setNotes={setNotes} msg={msg} anomaly={anomaly} isCritical={isCritical} submit={submit} />}
        <MyTracker data={data} highlightId={highlightId} trend={trend} explainTrend={explainTrend} isCritical={isCritical} anomalySet={anomalySet} hasServerIds={hasServerIds} readOnly={readOnly} loaded={loaded} />
      </div>
      {!readOnly && (
      <>
      <DailyCheckBanner userId={effectiveId} persona={persona} />
      <FqgChat isCritical={isCritical} fqg={fqg} effectiveId={effectiveId} lastLogId={lastLogId} onAnomaly={handleFqgAnomaly} persona={persona} />
      <NutritionLookup />
      <TriageWidget effectiveId={effectiveId} />
      </>
      )}
    </div>
  );
}
function Kpi({label,value,sub,alert}:any){ return <div className={`rounded-xl border p-3 ${alert?"bg-amber-50 border-amber-200 dark:bg-amber-950 dark:border-amber-800":"bg-white dark:bg-slate-900 dark:border-slate-700"}`}><div className="text-[11px] text-slate-500 dark:text-slate-400">{label}</div><div className="text-sm font-bold">{value}</div>{sub && <div className={`text-[11px] ${alert?"text-amber-700 dark:text-amber-300":"text-slate-500 dark:text-slate-400"}`}>{sub}</div>}</div>; }
