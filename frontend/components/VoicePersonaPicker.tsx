"use client";
import { useEffect, useState } from "react";
import { PERSONAS, DEFAULT_PERSONA_ID } from "@/lib/personas";
import { playServerAudio } from "./VoiceWidget";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const REGION_LABEL: Record<string, string> = { Bac: "Giọng Bắc", Trung: "Giọng Trung", Nam: "Giọng Nam" };

// Reused by /onboarding/voice (post-register) and Settings voice section.
export default function VoicePersonaPicker({
  userId,
  showSkip = true,
  onDone,
}: {
  userId: string;
  showSkip?: boolean;
  onDone?: (personaId: string) => void;
}) {
  const [selected, setSelected] = useState(DEFAULT_PERSONA_ID);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState("");

  useEffect(() => {
    if (!userId) return;
    (async () => {
      try {
        const res = await fetch(`${API}/v1/voice/prefs?user_id=${encodeURIComponent(userId)}`, { credentials: "include" });
        if (!res.ok) return;
        const data = await res.json();
        if (data?.selected_persona) setSelected(data.selected_persona);
      } catch { /* fail-open: default lan */ }
    })();
  }, [userId]);

  const save = async (id: string) => {
    setBusy(id); setMsg("");
    try {
      await fetch(`${API}/v1/voice/prefs?user_id=${encodeURIComponent(userId)}`, {
        method: "POST", credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selected_persona: id }),
      });
      setSelected(id);
      onDone?.(id);
    } catch (e: any) {
      setMsg("Lỗi lưu: " + e.message);
    } finally {
      setBusy("");
    }
  };

  return (
    <div>
      <div className="grid gap-3 sm:grid-cols-3">
        {PERSONAS.map((p) => (
          <div
            key={p.id}
            data-testid={`persona-card-${p.id}`}
            className={`rounded-2xl border p-4 text-center shadow-sm dark:border-slate-700 dark:bg-slate-900 ${selected === p.id ? "border-blue-600 ring-2 ring-blue-200 dark:ring-blue-800" : "bg-white"}`}
          >
            <div className="mx-auto h-16 w-16" dangerouslySetInnerHTML={{ __html: p.avatarSvg }} />
            <div className="mt-2 text-sm font-bold">{p.name}</div>
            <div className="text-[11px] text-slate-500 dark:text-slate-400">{REGION_LABEL[p.region] || p.region}</div>
            <div className="mt-2 min-h-[3rem] text-xs italic text-slate-600 dark:text-slate-300">“{p.greeting}”</div>
            <div className="mt-3 flex justify-center gap-2">
              <button
                onClick={() => playServerAudio(p.greeting, { persona: p.id })}
                className="rounded-lg border px-4 py-2 text-xs min-h-[44px] hover:bg-slate-50 dark:hover:bg-slate-800"
              >
                🔊 Nghe thử
              </button>
              <button
                onClick={() => save(p.id)}
                disabled={busy !== ""}
                className="rounded-lg bg-blue-600 px-4 py-2 text-xs font-bold text-white hover:bg-blue-700 min-h-[44px] disabled:opacity-50"
              >
                {busy === p.id ? "Đang lưu…" : selected === p.id ? "✓ Đã chọn" : "Chọn"}
              </button>
            </div>
          </div>
        ))}
      </div>
      {showSkip && (
        <div className="mt-4 text-center">
          <button onClick={() => save("lan")} disabled={busy !== ""} className="text-xs text-slate-500 underline min-h-[44px] px-3 py-2">
            Bỏ qua (dùng giọng Bắc mặc định)
          </button>
        </div>
      )}
      {msg && <div className="mt-2 text-xs text-red-600">{msg}</div>}
    </div>
  );
}
