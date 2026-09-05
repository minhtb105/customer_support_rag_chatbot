"use client";

import { useState } from "react";
import { queryRag, API_BASE } from "@/lib/api";

export default function ApiPlaygroundPage() {
  const [query, setQuery] = useState("Dấu hiệu đái tháo đường type 2 là gì? Khi nào cần đi khám?");
  const [topK, setTopK] = useState(5);
  const [userId, setUserId] = useState("demo_user");
  const [loading, setLoading] = useState(false);
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState("");

  const examples = [
    "Dấu hiệu đái tháo đường type 2 là gì?",
    "Ngưỡng chẩn đoán đái tháo đường theo WHO là bao nhiêu?",
    "Chế độ ăn cho người tiền đái tháo đường theo WHO?",
    "Khi nào cần đo HbA1c và tần suất tái khám?",
    "Biến chứng của đái tháo đường không kiểm soát?",
  ];

  const run = async () => {
    setLoading(true); setErr(""); setRes(null);
    try {
      const data = await queryRag(query, topK, userId);
      setRes(data);
    } catch (e: any) { setErr(e.message); }
    finally { setLoading(false); }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold">Hướng C — WHO-RAG Infrastructure API</h1>
        <p className="text-sm text-slate-600">Lớp truy vấn y tế đáng tin cậy cho app bên thứ 3 (eDoctor/Medihome). Trả về audit trail: citations, prompt_version, faithfulness.</p>
        <div className="mt-2 flex flex-wrap gap-2 text-xs">
          <code className="rounded-full border bg-white px-3 py-1">POST {API_BASE}/v1/query</code>
          <code className="rounded-full border bg-white px-3 py-1">GET {API_BASE}/docs</code>
          <code className="rounded-full border bg-white px-3 py-1">GET {API_BASE}/v1/health</code>
        </div>
      </div>

      <div className="rounded-2xl border bg-white p-5 shadow-sm">
        <div className="grid gap-4 sm:grid-cols-3">
          <div className="sm:col-span-2">
            <label className="text-xs font-medium">Câu hỏi</label>
            <textarea value={query} onChange={(e) => setQuery(e.target.value)} rows={3} className="mt-1 w-full rounded-lg border px-3 py-2 text-sm" placeholder="Nhập câu hỏi y tế..." />
            <div className="mt-2 flex flex-wrap gap-1.5">
              {examples.map((ex) => (
                <button key={ex} onClick={() => setQuery(ex)} className="rounded-full border bg-slate-50 px-3 py-1 text-xs hover:bg-slate-100 text-left">{ex}</button>
              ))}
            </div>
          </div>
          <div className="space-y-3">
            <div>
              <label className="text-xs">User ID</label>
              <input value={userId} onChange={(e) => setUserId(e.target.value)} className="mt-1 w-full rounded-lg border px-3 py-2 text-sm" />
            </div>
            <div>
              <label className="text-xs">Top K: {topK}</label>
              <input type="range" min={3} max={10} value={topK} onChange={(e) => setTopK(Number(e.target.value))} className="w-full" />
            </div>
            <button onClick={run} disabled={loading || !query.trim()} className="w-full rounded-lg bg-violet-600 py-2.5 text-sm font-medium text-white hover:bg-violet-700 disabled:opacity-50">
              {loading ? "Đang truy vấn..." : "Gửi truy vấn →"}
            </button>
          </div>
        </div>
        {err && <div className="mt-4 rounded-lg bg-red-50 border border-red-200 p-3 text-xs text-red-700">{err}</div>}
      </div>

      {res && (
        <div className="space-y-4">
          <div className="rounded-2xl border bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold">Answer</h2>
              <span className={`rounded-full px-2 py-1 text-xs border ${res.cache_hit ? "bg-emerald-50 border-emerald-200 text-emerald-700" : "bg-slate-50 border-slate-200"}`}>{res.cache_hit ? "cache hit" : "generated"}</span>
            </div>
            <div className="mt-3 rounded-lg bg-slate-50 p-4 text-sm leading-relaxed whitespace-pre-wrap border">{res.answer}</div>
            {res.cited_sources?.length ? <div className="mt-2 text-xs text-slate-600">Cited: [{res.cited_sources.join(", ")}]</div> : null}
            <div className="mt-3 text-xs text-slate-500">Latency: {res.audit?.latency_ms ?? "—"} ms · Prompt version: <code>{res.audit?.prompt_version ?? res.langsmith?.prompt_version ?? "—"}</code> · Reranker: {res.audit?.reranker_model ?? "—"}</div>
          </div>

          <div className="rounded-2xl border bg-white p-5 shadow-sm">
            <h3 className="text-sm font-semibold">Audit Trail — Citations ({res.audit?.citations?.length ?? 0})</h3>
            <div className="mt-3 space-y-3 max-h-96 overflow-auto pr-1">
              {(res.audit?.citations || res.contexts || []).map((c: any, i: number) => (
                <div key={i} className="rounded-lg border bg-slate-50 p-3">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-mono font-bold">[Source {c.source_id ?? i}] {c.dataset ?? ""}</span>
                    {c.score != null && <span className="rounded-full bg-white border px-2 py-0.5">score {Number(c.score).toFixed(3)}</span>}
                  </div>
                  {c.section_path && <div className="text-[11px] text-slate-500">{Array.isArray(c.section_path) ? c.section_path.join(" › ") : c.section_path}</div>}
                  {c.page_numbers?.length ? <div className="text-[11px] text-slate-500">pages {c.page_numbers.join(", ")}</div> : null}
                  <div className="mt-2 text-xs leading-relaxed text-slate-700 line-clamp-6">{c.content_snippet ?? (c.content ?? "").slice(0, 400)}</div>
                </div>
              ))}
            </div>
          </div>

          <details className="rounded-2xl border bg-white p-5 shadow-sm">
            <summary className="cursor-pointer text-sm font-semibold">Raw JSON (dành cho tích hợp B2B)</summary>
            <pre className="mt-3 overflow-auto rounded-lg bg-slate-900 p-4 text-xs text-slate-100">{JSON.stringify(res, null, 2)}</pre>
          </details>

          <div className="rounded-xl bg-violet-50 border border-violet-200 p-4 text-xs">
            <div className="font-semibold text-violet-900">Cách tích hợp (B2B2C)</div>
            <pre className="mt-2 overflow-auto rounded-lg bg-white p-3 border text-xs">{`curl -X POST ${API_BASE}/v1/query \\
  -H "Content-Type: application/json" \\
  -d '{"query":"${query.slice(0, 40)}...","top_k":${topK},"user_id":"${userId}"}'`}</pre>
            <div className="mt-2 text-violet-800">Response luôn có <code>cited_sources</code> + <code>citations[].content_snippet</code> + <code>audit.prompt_version</code> để audit & đo faithfulness.</div>
          </div>
        </div>
      )}

      {!res && !err && !loading && (
        <div className="rounded-2xl border border-dashed bg-slate-50 p-8 text-center text-sm text-slate-600">
          Nhập câu hỏi và bấm “Gửi truy vấn”. Thử câu hỏi mẫu phía trên. Cần FastAPI chạy tại <code>{API_BASE}</code> và đã index PDFs.
        </div>
      )}
    </div>
  );
}
