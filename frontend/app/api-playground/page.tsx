"use client";
import { useState } from "react";
import { queryRag, API_BASE } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import Link from "next/link";

export default function ApiPlaygroundPage() {
  const { user } = useAuth();
  const [query, setQuery] = useState("Dấu hiệu đái tháo đường type 2 là gì? Khi nào cần đi khám?");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [res, setRes] = useState<any>(null);
  const [err, setErr] = useState("");

  const examples = ["Dấu hiệu đái tháo đường type 2 là gì?","Ngưỡng chẩn đoán đái tháo đường theo WHO là bao nhiêu?","Chế độ ăn cho người tiền đái tháo đường theo WHO?","Khi nào cần đo HbA1c và tần suất tái khám?","Biến chứng của đái tháo đường không kiểm soát?"];

  const run = async () => {
    setLoading(true); setErr(""); setRes(null);
    try {
      const data = await queryRag(query, topK, user?.id || "demo");
      setRes(data);
    } catch (e: any) { setErr(e.message); }
    finally { setLoading(false); }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold">Hướng C — WHO-RAG Infrastructure API + HILT</h1>
        <p className="text-sm text-slate-600">Bắt buộc login. Trả về audit + 4 metrics faithfulness (HILT) → nếu low confidence → pending_review → expert duyệt.</p>
        <div className="mt-2 flex flex-wrap gap-2 text-xs">
          <code className="rounded-full border bg-white px-3 py-1">POST {API_BASE}/v1/query (auth)</code>
          <code className="rounded-full border bg-white px-3 py-1">GET {API_BASE}/docs</code>
        </div>
        {!user && <div className="mt-2 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg p-2">Cần đăng nhập để truy vấn. <Link href="/login" className="underline">Đăng nhập</Link></div>}
      </div>

      <div className="rounded-2xl border bg-white p-5 shadow-sm">
        <div className="grid gap-4 sm:grid-cols-3">
          <div className="sm:col-span-2">
            <label className="text-xs font-medium">Câu hỏi</label>
            <textarea value={query} onChange={(e) => setQuery(e.target.value)} rows={3} className="mt-1 w-full rounded-lg border px-3 py-2 text-sm" />
            <div className="mt-2 flex flex-wrap gap-1.5">
              {examples.map((ex) => <button key={ex} onClick={() => setQuery(ex)} className="rounded-full border bg-slate-50 px-3 py-1 text-xs hover:bg-slate-100 text-left">{ex}</button>)}
            </div>
          </div>
          <div className="space-y-3">
            <div className="text-xs">User: {user?.username || "chưa login"} ({user?.role || "—"})</div>
            <div><label className="text-xs">Top K: {topK}</label><input type="range" min={3} max={10} value={topK} onChange={(e) => setTopK(Number(e.target.value))} className="w-full" /></div>
            <button onClick={run} disabled={loading || !query.trim() || !user} className="w-full rounded-lg bg-violet-600 py-2.5 text-sm font-medium text-white hover:bg-violet-700 disabled:opacity-50">{loading ? "Đang truy vấn..." : "Gửi truy vấn →"}</button>
          </div>
        </div>
        {err && <div className="mt-4 rounded-lg bg-red-50 border border-red-200 p-3 text-xs text-red-700">{err}</div>}
      </div>

      {res && (
        <div className="space-y-4">
          {res.status==="pending_review" && (
            <div className="rounded-2xl border border-amber-200 bg-amber-50 p-4">
              <div className="text-sm font-bold text-amber-900">⏳ Đang chờ chuyên gia duyệt</div>
              <div className="text-xs text-amber-800">AI chưa tự tin (failed: {res.evaluation?.failed_metrics?.join(", ")} → routed tới {res.evaluation?.routed_role}). Review ID: {res.review_id}</div>
              <div className="text-xs mt-1">Bạn sẽ nhận notification khi expert duyệt. Xem tại <Link href="/my/reviews" className="underline">Lịch sử</Link></div>
            </div>
          )}
          <div className="rounded-2xl border bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold">Answer {res.is_low_confidence ? "(low confidence)" : ""}</h2>
              <span className={`rounded-full px-2 py-1 text-xs border ${res.cache_hit ? "bg-emerald-50 border-emerald-200 text-emerald-700" : "bg-slate-50 border-slate-200"}`}>{res.cache_hit ? "cache hit" : "generated"} · {res.status}</span>
            </div>
            <div className="mt-3 rounded-lg bg-slate-50 p-4 text-sm leading-relaxed whitespace-pre-wrap border">{res.answer}</div>
            {res.evaluation && (
              <div className="mt-3 rounded-lg border bg-white p-3 text-xs">
                <div className="font-semibold">HILT Evaluation (4 metrics, ngưỡng 3.5)</div>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-2">
                  <Metric k="faithfulness" v={res.evaluation.metrics?.faithfulness} t={res.evaluation.thresholds?.faithfulness} />
                  <Metric k="precision" v={res.evaluation.metrics?.context_precision} t={res.evaluation.thresholds?.context_precision} />
                  <Metric k="recall" v={res.evaluation.metrics?.context_recall} t={res.evaluation.thresholds?.context_recall} />
                  <Metric k="answer_rel" v={res.evaluation.metrics?.answer_relevance} t={res.evaluation.thresholds?.answer_relevance} />
                </div>
                <div className="mt-2 text-slate-600">Failed: {res.evaluation.failed_metrics?.join(", ") || "none"} · Routed: {res.evaluation.routed_role} · Confidence: {res.evaluation.confidence}</div>
              </div>
            )}
            <div className="mt-3 text-xs text-slate-500">Latency: {res.audit?.latency_ms ?? "—"} ms · Prompt: <code>{res.audit?.prompt_version ?? "—"}</code> · Review: {res.review_id || "—"}</div>
          </div>

          <div className="rounded-2xl border bg-white p-5 shadow-sm">
            <h3 className="text-sm font-semibold">Audit Trail — Citations ({res.audit?.citations?.length ?? 0})</h3>
            <div className="mt-3 space-y-3 max-h-96 overflow-auto pr-1">
              {(res.audit?.citations || res.contexts || []).map((c: any, i: number) => (
                <div key={i} className="rounded-lg border bg-slate-50 p-3">
                  <div className="flex items-center justify-between text-xs"><span className="font-mono font-bold">[Source {c.source_id ?? i}] {c.dataset ?? ""}</span>{c.score != null && <span className="rounded-full bg-white border px-2 py-0.5">score {Number(c.score).toFixed(3)}</span>}</div>
                  <div className="mt-2 text-xs leading-relaxed text-slate-700">{c.content_snippet ?? (c.content ?? "").slice(0, 400)}</div>
                </div>
              ))}
            </div>
          </div>
          <details className="rounded-2xl border bg-white p-5 shadow-sm"><summary className="cursor-pointer text-sm font-semibold">Raw JSON</summary><pre className="mt-3 overflow-auto rounded-lg bg-slate-900 p-4 text-xs text-slate-100">{JSON.stringify(res, null, 2)}</pre></details>
        </div>
      )}
      {!res && !err && !loading && <div className="rounded-2xl border border-dashed bg-slate-50 p-8 text-center text-sm text-slate-600">Nhập câu hỏi và bấm Gửi truy vấn. Cần login.</div>}
    </div>
  );
}
function Metric({k,v,t}:{k:string, v:number, t:number}){
  const ok = v>=t;
  return <div className={`rounded-lg border p-2 ${ok?"bg-emerald-50 border-emerald-200":"bg-red-50 border-red-200"}`}><div className="font-bold text-xs">{k}</div><div className="text-sm">{v?.toFixed(1)} / 5 <span className="text-[11px]">ngưỡng {t}</span></div><div className="text-[11px]">{ok?"✅ đạt":"❌ thiếu"}</div></div>;
}
