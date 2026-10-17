// Shared voice playback helpers (tracker, persona picker, lessons).
// Server TTS first (Piper-or-better via /v1/voice/speak), browser last resort.
// playServerAudio now reports outcome ("ended" | "fallback" | "error") so callers
// can await real end-of-speech (fixes chunked playback overlap in hoc/[id]).

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Daily anti-nag key per user + VN date (Asia/Ho_Chi_Minh, not UTC).
export function vnToday(): string {
  return new Intl.DateTimeFormat("en-CA", { timeZone: "Asia/Ho_Chi_Minh", year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date());
}

// Resolve with `value` if the event never fires (headless / stalled engine).
function withCap<T>(p: Promise<T>, ms: number, value: T): Promise<T> {
  return Promise.race([p, new Promise<T>((resolve) => setTimeout(() => resolve(value), ms))]);
}

// Persona voice for every playback (chosen in onboarding/Settings). Fail-open "lan".
export async function loadPersona(userId: string): Promise<string> {
  if (!userId) return "lan";
  try {
    const res = await fetch(`${API}/v1/voice/prefs?user_id=${encodeURIComponent(userId)}`, { credentials: "include" });
    if (!res.ok) return "lan";
    const data = await res.json();
    return data?.selected_persona || "lan";
  } catch {
    return "lan";
  }
}

// Browser TTS last resort. "fallback" when it spoke (or errored), "error" if
// speechSynthesis is unavailable/throws.
function speakBrowser(text: string): Promise<"fallback" | "error"> {
  try {
    if (!("speechSynthesis" in window)) return Promise.resolve("error");
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text.slice(0, 500));
    u.lang = "vi-VN";
    const done = new Promise<"fallback">((resolve) => {
      u.onend = () => resolve("fallback");
      u.onerror = () => resolve("fallback");
    });
    window.speechSynthesis.speak(u);
    return withCap(done, Math.max(3000, text.length * 100), "fallback");
  } catch {
    return Promise.resolve("error");
  }
}

// Server TTS first, browser last resort. Resolves "ended" when the server audio
// finishes (onended/onerror), else forwards speakBrowser's "fallback"/"error".
// opts.persona: backend resolves selected_persona -> region server-side.
export async function playServerAudio(text: string, opts?: { persona?: string; region?: string }): Promise<"ended" | "fallback" | "error"> {
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
      const done = new Promise<"ended">((resolve) => {
        audio.onended = () => resolve("ended");
        audio.onerror = () => resolve("ended");
      });
      await audio.play(); // rejection -> outer catch -> browser fallback
      return await withCap(done, Math.max(3000, text.length * 100), "ended");
    }
    throw new Error("no audio");
  } catch {
    return speakBrowser(text);
  }
}
