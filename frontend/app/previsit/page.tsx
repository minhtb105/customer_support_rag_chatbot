"use client";

import { useState } from "react";
import { generateSoap, generateSoapMarkdown } from "@/lib/api";

export default function PrevisitPage() {
  const [userId, setUserId] = useState("demo_user");
  const [days, setDays] = useState(14);
  const [data, setData] = useState<any>(null);
  const [md, setMd] = useState("");
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");

  const gen = async (asMd = false) => {
    setLoading(true); setErr(""); setData(null); setMd("");
    try {
      if (asMd) {
        const text = await generateSoapMarkdown(userId, days);
        setMd(text);
      } else {
        const j = await generateSoap(userId, days);
        setData(j);
      }
    } catch (e: any) { setErr(e.message); }
    finally { setLoading(false); }
  };

  const downloadMd = () => {
    if (!md) return;
    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `SOAP_${userId}_${days}d.md`; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold">Hướng B — Hồ sơ trước tái khám (SOAP)</h1>
        <p className="text-sm text-slate-600">Tóm tắt dữ liệu tự theo dõi thành bản chuẩn hóa SOAP/ADA, bán B2B2C cho phòng khám. Tận dụng BHYT telehealth 1/7/2025.</p>
      </div>

      <div className="rounded-2xl border bg-white p-5 shadow-sm">
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <label htmlFor="previsit-userid" className="text-xs text-slate-600">User ID</label>
            <input id="previsit-userid" data-testid="previsit-userid" value={userId} onChange={(e) => setUserId(e.target.value)} className="mt-1 rounded-lg border px-3 py-2 text-sm" />
          </div>
          <div>
            <label htmlFor="previsit-days" className="text-xs text-slate-600">Số ngày</label>
            <select id="previsit-days" data-testid="previsit-days" value={days} onChange={(e) => setDays(Number(e.target.value))} className="mt-1 rounded-lg border px-3 py-2 text-sm">
              <option value={7}>7 ngày</option>
              <option value={14}>14 ngày</option>
              <option value={30}>30 ngày</option>
            </select>
          </div>
          <button onClick={() => gen(false)} disabled={loading} className="rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50">
            {loading ? "Đang tạo..." : "Tạo SOAP (JSON)"}
          </button>
          <button onClick={() => gen(true)} disabled={loading} className="rounded-lg border bg-white px-5 py-2.5 text-sm hover:bg-slate-50">Tạo Markdown</button>
        </div>
        {err && <div className="mt-3 rounded-lg bg-red-50 border border-red-200 p-3 text-xs text-red-700">{err} — kiểm tra FastAPI đã chạy và có logs cho user này.</div>}
      </div>

      {data && (
        <div className="rounded-2xl border bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Kết quả SOAP — {data.user_id} · {data.period}</h2>
            <div className="text-xs text-slate-500">{new Date(data.generated_at).toLocaleString("vi-VN")}</div>
          </div>
          <div className="mt-3 grid gap-4 sm:grid-cols-2">
            <div className="rounded-xl bg-slate-50 p-4">
              <div className="text-xs font-bold text-slate-700">S — Subjective</div>
              <div className="mt-1 text-xs leading-relaxed text-slate-700 whitespace-pre-wrap">{data.soap.subjective}</div>
            </div>
            <div className="rounded-xl bg-blue-50 p-4 border border-blue-100">
              <div className="text-xs font-bold text-blue-900">O — Objective</div>
              <div className="mt-1 text-xs leading-relaxed whitespace-pre-wrap">{data.soap.objective}</div>
            </div>
            <div className="rounded-xl bg-amber-50 p-4 border border-amber-100">
              <div className="text-xs font-bold">A — Assessment</div>
              <div className="mt-1 text-xs leading-relaxed whitespace-pre-wrap">{data.soap.assessment}</div>
            </div>
            <div className="rounded-xl bg-emerald-50 p-4 border border-emerald-100">
              <div className="text-xs font-bold text-emerald-900">P — Plan</div>
              <div className="mt-1 text-xs leading-relaxed whitespace-pre-wrap">{data.soap.plan}</div>
            </div>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4 text-xs">
            <div className="rounded-lg border bg-white p-3"><div className="text-slate-500">Tổng logs</div><div className="font-bold">{data.stats.total_logs}</div></div>
            <div className="rounded-lg border bg-white p-3"><div className="text-slate-500">Avg</div><div className="font-bold">{data.stats.avg_mgdl ?? "—"} mg/dL</div></div>
            <div className="rounded-lg border bg-white p-3"><div className="text-slate-500">7d avg</div><div className="font-bold">{data.stats.last_7_days_avg ?? "—"} mg/dL</div></div>
            <div className="rounded-lg border bg-white p-3"><div className="text-slate-500">Tần suất</div><div className="font-bold">{data.stats.logs_per_week}/tuần</div></div>
          </div>
          <div className="mt-4 rounded-lg bg-slate-900 p-3 text-xs text-slate-100 overflow-auto">
            <div className="font-mono text-[11px] text-slate-400">JSON (gửi B2B2C)</div>
            <pre className="mt-1 whitespace-pre-wrap break-words text-xs">{JSON.stringify(data, null, 2)}</pre>
          </div>
        </div>
      )}

      {md && (
        <div className="rounded-2xl border bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Markdown — sẵn sàng gửi bác sĩ</h2>
            <button onClick={downloadMd} className="rounded-full bg-slate-900 px-4 py-2 text-xs text-white">Tải .md</button>
          </div>
          <pre className="mt-3 overflow-auto rounded-lg bg-slate-50 p-4 text-xs leading-relaxed whitespace-pre-wrap border">{md}</pre>
          <div className="mt-2 text-[11px] text-slate-500">Lưu ý: Tài liệu do AI hỗ trợ, không thay thế chẩn đoán y khoa.</div>
        </div>
      )}

      {!data && !md && !loading && (
        <div className="rounded-2xl border border-dashed bg-slate-50 p-8 text-center text-sm text-slate-600">
          Nhập User ID đã có logs (thử <code>demo_user</code> sau khi nhập ở tab Tracker) rồi bấm Tạo SOAP.
        </div>
      )}
    </div>
  );
}
