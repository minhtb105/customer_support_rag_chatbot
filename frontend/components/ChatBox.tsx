"use client";
import { useEffect, useState } from "react";

// Shared collapsible chat shell for tracker (FqgChat + TriageWidget).
// Wrappers own domain logic via onSend(text) -> reply messages; shell owns
// open/input/messages/sending state. Visual classes stay per-wrapper via accent.
export interface ChatMsg { role: string; text: string; }

export default function ChatBox({ title, placeholder, emptyHint, seeds = [], onSend, accent = "blue", testPrefix }: {
  title: string;
  placeholder: string;
  emptyHint: string;
  seeds?: ChatMsg[];
  onSend: (text: string) => Promise<ChatMsg[]>;
  accent?: "blue" | "violet";
  testPrefix?: string;
}) {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [open, setOpen] = useState(false);
  const [sending, setSending] = useState(false);

  useEffect(() => {
    if (seeds.length) {
      setOpen(true);
      setMessages((prev) => {
        const fresh = seeds.filter((s) => !prev.some((m) => m.text === s.text));
        return fresh.length ? [...prev, ...fresh] : prev;
      });
    }
  }, [seeds]);

  const send = async () => {
    const text = input.trim();
    if (!text || sending) return;
    setInput("");
    setOpen(true);
    setMessages((prev) => [...prev, { role: "user", text }]);
    setSending(true);
    try {
      const replies = await onSend(text);
      if (replies.length) setMessages((prev) => [...prev, ...replies]);
    } catch (e: any) {
      setMessages((prev) => [...prev, { role: "sys", text: "Lỗi: " + e.message }]);
    } finally {
      setSending(false);
    }
  };

  const box = accent === "violet"
    ? "rounded-2xl border bg-white p-5 shadow-sm dark:bg-slate-900 dark:border-slate-700"
    : "rounded-2xl border border-blue-100 bg-blue-50/50 p-4 shadow-sm dark:bg-blue-950/40 dark:border-blue-900";
  const btn = accent === "violet" ? "bg-violet-600 hover:bg-violet-700" : "bg-blue-600 hover:bg-blue-700";
  const msgCls = (r: string) => r === "user"
    ? "ml-8 bg-emerald-50 border border-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:border-emerald-800 dark:text-emerald-100"
    : r === "guard" || r === "emg"
      ? "bg-red-50 border border-red-200 text-red-800 dark:bg-red-950 dark:border-red-800 dark:text-red-100"
      : r === "sys"
        ? "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300"
        : accent === "violet"
          ? "mr-8 border border-violet-100 bg-violet-50 text-violet-900"
          : "mr-8 bg-blue-50 border border-blue-100 text-blue-900 dark:bg-blue-950 dark:border-blue-800 dark:text-blue-100";

  return (
    <div className={box}>
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">{title}</h3>
        <button onClick={() => setOpen((o) => !o)} className="text-xs text-blue-700 underline dark:text-blue-300 min-h-[44px] px-2">
          {open ? "Thu gọn" : accent === "violet" ? "Mở" : "Mở chat"}
        </button>
      </div>
      {open && (
        <>
          <div className="mt-3 max-h-64 space-y-2 overflow-auto rounded-lg border bg-white p-3 dark:bg-slate-900 dark:border-slate-700">
            {messages.length ? messages.map((m, i) => (
              <div key={i} className={`rounded-lg p-2 text-sm leading-relaxed whitespace-pre-wrap ${msgCls(m.role)}`}>
                {m.role === "guard" && <div className="font-bold">⛔ Ngoài phạm vi hỗ trợ</div>}
                <div className="whitespace-pre-wrap">{m.text}</div>
              </div>
            )) : <div className="p-2 text-sm text-slate-500 dark:text-slate-400">{emptyHint}</div>}
          </div>
          <div className="mt-2 flex gap-2">
            <input
              data-testid={testPrefix ? `${testPrefix}-input` : undefined}
              value={input} onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") send(); }}
              placeholder={placeholder}
              className="w-full rounded-lg border px-3 py-2 text-sm dark:bg-slate-900 dark:border-slate-700 min-h-[44px]"
            />
            <button data-testid={testPrefix ? `${testPrefix}-send` : undefined} onClick={send} disabled={sending} className={`rounded-lg px-4 py-2 text-sm text-white disabled:opacity-50 min-h-[44px] ${btn}`}>
              {sending ? "..." : "Gửi"}
            </button>
          </div>
        </>
      )}
    </div>
  );
}
