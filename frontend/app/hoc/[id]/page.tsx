"use client";
import { use, useEffect, useState } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import { getLesson, submitLessonQuiz } from "@/lib/api";
import { playServerAudio } from "@/components/VoiceWidget";

const LETTERS = ["A", "B", "C"];
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Chunked playback (F3): split on sentence boundaries, pack to <=480 chars
// so no single /speak call (truncate [:500]) ever cuts mid-sentence.
export function splitScript(text: string): string[] {
  const t = (text || "").trim();
  if (!t) return [];
  const parts = t.match(/[^.!?…]+[.!?…]+/g);
  const sents = (parts && parts.join("").length >= t.length - 2 ? parts : [t])
    .map((s) => s.trim()).filter(Boolean);
  const chunks: string[] = [];
  let cur = "";
  const pushLong = (s: string) => {
    const words = s.split(/\s+/);
    let c = "";
    for (const w of words) {
      const cand = (c + " " + w).trim();
      if (cand.length <= 480) c = cand;
      else { if (c) chunks.push(c); c = w; }
    }
    return c;
  };
  for (const s of sents) {
    const cand = (cur + " " + s).trim();
    if (cand.length <= 480) { cur = cand; continue; }
    if (cur) chunks.push(cur);
    cur = s.length > 480 ? pushLong(s) : s;
  }
  if (cur) chunks.push(cur);
  return chunks;
}

export default function LessonPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { user, loading } = useAuth();
  const [lesson, setLesson] = useState<any>(null);
  const [picked, setPicked] = useState<number | null>(null);
  const [result, setResult] = useState<any>(null);
  const [err, setErr] = useState("");
  const [playing, setPlaying] = useState(false);
  const [persona, setPersona] = useState("lan");
  const [chunkLabel, setChunkLabel] = useState("");
  const [voiceMsg, setVoiceMsg] = useState("");
  const [voiceEmergency, setVoiceEmergency] = useState("");
  const [recording, setRecording] = useState(false);
  const stopRef = useState({ stop: false })[0];

  useEffect(() => {
    if (loading || !user) return;
    (async () => {
      try {
        setLesson(await getLesson(id, user.id));
        const res = await fetch(
          `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/v1/voice/prefs?user_id=${encodeURIComponent(user.id)}`,
          { credentials: "include" }
        );
        if (res.ok) {
          const p = await res.json();
          if (p?.selected_persona) setPersona(p.selected_persona);
        }
      } catch (e: any) {
        setErr(e.message);
      }
    })();
  }, [loading, user, id]);

  const play = async () => {
    if (!lesson?.script_vi) return;
    const chunks = splitScript(lesson.script_vi);
    setPlaying(true);
    stopRef.stop = false;
    try {
      for (let i = 0; i < chunks.length; i++) {
        if (stopRef.stop) break;
        setChunkLabel(`Đang đọc đoạn ${i + 1}/${chunks.length}…`);
        await playServerAudio(chunks[i], { persona });
      }
    } finally {
      setPlaying(false);
      setChunkLabel("");
    }
  };

  const stop = () => { stopRef.stop = true; };

  // Quiz voice 2nd channel (F2): mic -> /transcribe (red_flag preserved) ->
  // submitLessonQuiz(answer_text) via map_quiz_answer, fail-open to buttons.
  const answerVoice = async () => {
    if (!user) return;
    setVoiceMsg("");
    setVoiceEmergency("");
    let stream: MediaStream | null = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setVoiceMsg("Không mở được mic. Bác bấm nút A, B hoặc C giúp con nhé.");
      return;
    }
    setRecording(true);
    try {
      const rec = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      const blob: Blob = await new Promise((resolve, reject) => {
        rec.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
        rec.onstop = () => resolve(new Blob(chunks, { type: "audio/webm" }));
        rec.onerror = () => reject(new Error("record failed"));
        rec.start();
        setTimeout(() => { try { rec.stop(); } catch {} }, 8000);
      });
      const fd = new FormData();
      fd.append("audio", blob, "quiz.webm");
      const res = await fetch(`${API}/v1/voice/transcribe`, { method: "POST", credentials: "include", body: fd });
      if (!res.ok) throw new Error(`transcribe ${res.status}`);
      const data = await res.json();
      if (data?.emergency) {
        setVoiceEmergency(data.message || "Gọi 115 ngay.");
        return;
      }
      const text = (data?.text || "").trim();
      if (!text) {
        setVoiceMsg("Con nghe chưa rõ. Bác bấm nút A, B hoặc C giúp con nhé.");
        return;
      }
      const r = await submitLessonQuiz(id, user.id, text);
      if (r?.mapped === null) {
        setVoiceMsg(`Bác nói: “${text}”. ` + (r.message || "Bác bấm nút A, B hoặc C giúp con nhé."));
        return;
      }
      setResult(r);
      setVoiceMsg(`Đã nhận câu trả lời bằng giọng nói: “${text}”`);
    } catch (e: any) {
      setVoiceMsg("Lỗi voice: " + (e?.message || e));
    } finally {
      setRecording(false);
      stream?.getTracks().forEach((t) => t.stop());
    }
  };

  const answer = async (i: number) => {
    if (!user) return;
    setPicked(i);
    try {
      setResult(await submitLessonQuiz(id, user.id, i));
    } catch (e: any) {
      setErr(e.message);
    }
  };

  if (loading) return <div className="mx-auto max-w-2xl p-6 text-sm">Đang tải…</div>;
  if (!user) return <div className="mx-auto max-w-2xl p-6 text-sm">Bác đăng nhập để học nhé.</div>;
  if (err && !lesson) return <div className="mx-auto max-w-2xl p-6 text-sm text-red-600">{err}</div>;
  if (!lesson) return <div className="mx-auto max-w-2xl p-6 text-sm">Đang tải bài…</div>;

  return (
    <div className="mx-auto max-w-2xl p-6">
      <Link href="/hoc" className="text-sm text-blue-700 underline dark:text-blue-300 min-h-[44px] inline-block py-2">← Về danh sách bài</Link>
      <h1 className="mt-2 text-lg font-bold">{lesson.title}</h1>
      <button onClick={play} disabled={playing} className="mt-3 rounded-full bg-violet-600 px-6 py-3 text-sm font-bold text-white hover:bg-violet-700 min-h-[44px]">
        {playing ? `🔊 ${chunkLabel || "Đang đọc…"}` : "🔊 Nghe bài (khoảng 1 phút)"}
      </button>
      {playing && (
        <div className="mt-2 flex items-center gap-2">
          <div className="text-xs text-slate-600 dark:text-slate-300">{chunkLabel}</div>
          <button onClick={stop} className="rounded-full border px-4 py-2 text-xs min-h-[44px]">⏹ Dừng</button>
        </div>
      )}
      <p className="mt-3 text-sm leading-relaxed">{lesson.script_vi}</p>
      <div className="mt-5 rounded-2xl border bg-white p-4 shadow-sm dark:bg-slate-900 dark:border-slate-700">
        <div className="text-sm font-bold">❓ {lesson.quiz?.q}</div>
        <div className="mt-3 space-y-2">
          {lesson.quiz?.options?.map((op: string, i: number) => (
            <button
              key={i}
              onClick={() => answer(i)}
              className={`block w-full rounded-xl border px-4 py-3 text-left text-sm min-h-[44px] ${
                picked === i
                  ? result?.correct
                    ? "border-emerald-600 bg-emerald-50 dark:bg-emerald-950"
                    : "border-red-600 bg-red-50 dark:bg-red-950"
                  : "hover:bg-slate-50 dark:hover:bg-slate-800"
              }`}
            >
              <b>{LETTERS[i]}.</b> {op}
            </button>
          ))}
        </div>
        {result && (
          <div className="mt-3 text-sm">
            {result.correct ? "🎉 Đúng rồi, giỏi lắm bác!" : "💪 Chưa đúng, không sao cả."}{" "}
            {result.explanation}
          </div>
        )}
        {err && <div className="mt-2 text-sm text-red-600">{err}</div>}
        <div className="mt-3 border-t pt-3">
          <button
            onClick={answerVoice}
            disabled={recording}
            className="rounded-full bg-slate-900 px-5 py-3 text-xs font-bold text-white dark:bg-slate-100 dark:text-slate-900 min-h-[44px]"
          >
            {recording ? "🎤 Đang nghe… (tối đa 8s)" : "🎤 Trả lời bằng giọng nói"}
          </button>
          <div className="mt-1 text-xs text-slate-500">Nói “đáp án A/B/C” — nút A/B/C vẫn là cách chính.</div>
          {voiceEmergency && (
            <div role="alert" className="mt-2 rounded-xl border-2 border-red-600 bg-red-600 p-3 text-xs font-bold text-white">
              🚨 {voiceEmergency} <a href="tel:115" className="ml-2 rounded-full bg-white px-4 py-2 text-red-700">📞 Gọi 115</a>
            </div>
          )}
          {voiceMsg && <div className="mt-2 text-xs text-slate-600 dark:text-slate-300">{voiceMsg}</div>}
        </div>
      </div>
    </div>
  );
}
