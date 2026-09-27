"use client";
import { useEffect, useRef, useState } from "react";
import { logGlucose, queryRag } from "@/lib/api";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Daily anti-nag key per user + VN date (Asia/Ho_Chi_Minh, not UTC).
function vnToday(): string {
  return new Intl.DateTimeFormat("en-CA", { timeZone: "Asia/Ho_Chi_Minh", year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date());
}

function speakBrowser(text: string) {
  try {
    if (!("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text.slice(0, 500));
    u.lang = "vi-VN";
    window.speechSynthesis.speak(u);
  } catch { /* fail-open text-only */ }
}

// Server TTS first (Piper-or-better via /v1/voice/speak), browser last resort.
// opts.persona: backend resolves selected_persona -> region server-side (D1).
export async function playServerAudio(text: string, opts?: { persona?: string; region?: string }) {
  try {
    const res = await fetch(`${API}/v1/voice/speak`, {
      method: "POST", credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text.slice(0, 500),
        ...(opts?.persona ? { selected_persona: opts.persona } : {}),
        ...(opts?.region ? { voice_region: opts.region } : {}),
      }),
    });
    if (!res.ok) throw new Error(`speak ${res.status}`);
    const data = await res.json();
    if (data?.audio_available && data?.audio_base64) {
      const audio = new Audio(`data:${data.mime || "audio/wav"};base64,${data.audio_base64}`);
      await audio.play();
      return;
    }
    throw new Error("no audio");
  } catch {
    speakBrowser(text);
  }
}

export default function VoiceWidget({ userId }: { userId: string }) {
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [parse, setParse] = useState<any>(null);
  const [emergency, setEmergency] = useState("");
  const [msg, setMsg] = useState("");
  const [daily, setDaily] = useState<any>(null);
  const [persona, setPersona] = useState("lan");
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  // Persona voice for every playback (chosen in onboarding/Settings).
  useEffect(() => {
    if (!userId) return;
    (async () => {
      try {
        const res = await fetch(`${API}/v1/voice/prefs?user_id=${encodeURIComponent(userId)}`, { credentials: "include" });
        if (!res.ok) return;
        const data = await res.json();
        if (data?.selected_persona) setPersona(data.selected_persona);
      } catch { /* fail-open: default lan */ }
    })();
  }, [userId]);

  // Daily missing-metrics check: once per day after login (D5).
  useEffect(() => {
    if (!userId) return;
    const key = `voice_daily_${userId}_${vnToday()}`;
    if (localStorage.getItem(key)) return;
    (async () => {
      try {
        const res = await fetch(`${API}/v1/voice/daily-check?user_id=${encodeURIComponent(userId)}`, { credentials: "include" });
        if (!res.ok) return;
        const data = await res.json();
        localStorage.setItem(key, "1");
        if (data?.missing_slots?.length) {
          setDaily(data);
          playServerAudio(data.greeting || "", { persona });
        }
      } catch { /* fail-open: no nag on error */ }
    })();
  }, [userId]);

  const toggleRec = async () => {
    if (recording) {
      mediaRef.current?.stop();
      return;
    }
    setMsg(""); setTranscript(""); setParse(null); setEmergency("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => { if (e.data.size) chunksRef.current.push(e.data); };
      mr.onstop = async () => {
        setRecording(false);
        stream.getTracks().forEach((t) => t.stop());
        await sendAudio(new Blob(chunksRef.current, { type: "audio/webm" }));
      };
      mediaRef.current = mr;
      mr.start();
      setRecording(true);
    } catch {
      setMsg("Không mở được mic. Bác dùng nút Tự điền bên dưới nhé.");
    }
  };

  const sendAudio = async (blob: Blob) => {
    setMsg("Đang nghe...");
    try {
      const fd = new FormData();
      fd.append("audio", blob, "voice.webm");
      const res = await fetch(`${API}/v1/voice/transcribe`, { method: "POST", credentials: "include", body: fd });
      if (!res.ok) throw new Error(`transcribe ${res.status}`);
      const data = await res.json();
      setTranscript(data.text || "");
      if (data?.emergency) {
        setEmergency(data.message || "Gọi 115 ngay.");
        playServerAudio(data.message || "Gọi 115 ngay.", { persona });
        return;
      }
      setParse(data.parse || null);
      if (data.parse && !data.parse.ambiguous) playServerAudio(data.parse.confirm_text || "", { persona });
      else if (data.parse) setMsg("Con nghe chưa rõ. Bác bấm mic nói lại giúp con nhé.");
    } catch (e: any) {
      setMsg("Lỗi voice: " + e.message);
    }
  };

  const confirmSave = async () => {
    if (!parse || parse.ambiguous) return;
    try {
      const rec: any = await logGlucose({ user_id: userId, value_mgdl: parse.value, context: parse.context });
      setMsg(`${rec.classification?.toUpperCase()}: ${rec.message} (đã lưu #${rec.id})`);
      setParse(null); setTranscript("");
    } catch (e: any) {
      setMsg("Lỗi lưu: " + e.message);
    }
  };

  const askVoice = async (q: string) => {
    try {
      const res: any = await queryRag(q, 5, userId);
      const short = (res?.answer || "").slice(0, 500);
      if (short) playServerAudio(short, { persona });
    } catch { /* text already shown */ }
  };

  return (
    <div className="rounded-2xl border bg-white p-5 shadow-sm dark:bg-slate-900 dark:border-slate-700">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">🎙️ Voicebot (nói thay vì gõ)</h3>
        <a href="#manual-entry" className="text-xs text-blue-700 underline dark:text-blue-300 min-h-[44px] px-2 py-2">Tự điền</a>
      </div>
      {daily?.missing_slots?.length > 0 && (
        <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800 dark:bg-amber-950 dark:border-amber-800 dark:text-amber-100">
          {daily.greeting}
        </div>
      )}
      <div className="mt-3 flex items-center gap-3">
        <button onClick={toggleRec} className={`rounded-full px-6 py-3 text-sm font-bold text-white min-h-[44px] ${recording ? "bg-red-600 animate-pulse" : "bg-violet-600 hover:bg-violet-700"}`}>
          {recording ? "⏹ Dừng" : "🎤 Bấm để nói"}
        </button>
        {transcript && <div className="text-sm text-slate-600 dark:text-slate-300">“{transcript}”</div>}
      </div>
      {emergency && (
        <div role="alert" className="mt-3 rounded-xl border-2 border-red-600 bg-red-600 p-4 text-sm font-bold text-white">
          🚨 {emergency} <a href="tel:115" className="ml-2 rounded-full bg-white px-5 py-2 text-red-700">📞 Gọi 115</a>
        </div>
      )}
      {parse && !parse.ambiguous && !emergency && (
        <div className="mt-3 rounded-xl border border-blue-200 bg-blue-50 p-4 dark:bg-blue-950 dark:border-blue-800">
          <div className="text-sm font-semibold">{parse.confirm_text}</div>
          <div className="mt-3 flex gap-2">
            <button onClick={confirmSave} className="rounded-lg bg-emerald-600 px-6 py-3 text-sm font-bold text-white hover:bg-emerald-700 min-h-[44px]">Đúng, lưu</button>
            <button onClick={toggleRec} className="rounded-lg border px-6 py-3 text-sm min-h-[44px]">Nói lại</button>
          </div>
        </div>
      )}
      {msg && <div className="mt-2 text-xs text-slate-600 dark:text-slate-300">{msg}</div>}
      {transcript && !emergency && (!parse || parse.ambiguous) && (
        <button onClick={() => askVoice(transcript)} className="mt-2 rounded-lg bg-slate-900 px-4 py-2 text-xs text-white dark:bg-slate-100 dark:text-slate-900 min-h-[44px]">Hỏi AI câu này</button>
      )}
    </div>
  );
}
