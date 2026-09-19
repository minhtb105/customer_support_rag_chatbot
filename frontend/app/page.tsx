"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getHealth, getGuidelinesStatus, API_BASE } from "@/lib/api";

export default function OverviewPage() {
  const [health, setHealth] = useState<any>(null);
  const [guidelines, setGuidelines] = useState<any>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: "offline" }));
    getGuidelinesStatus().then(setGuidelines).catch(() => {});
  }, []);

  return (
    <div className="space-y-6">
      {/* Hero */}
      <div className="rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-600 p-6 text-white sm:p-8">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full bg-white/20 px-3 py-1 text-xs backdrop-blur">
              <span className="h-2 w-2 rounded-full bg-emerald-300" /> BHYT chi trả telehealth từ 1/7/2025
            </div>
            <h1 className="mt-3 text-2xl font-bold leading-tight sm:text-3xl">
              Trợ lý Đái tháo đường
              <span className="block text-lg font-normal text-blue-100 sm:text-xl">WHO-RAG • Tuân thủ • Tái khám • Hạ tầng API</span>
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-blue-100">
              Giải quyết khoảng trống giữa 2 lần tái khám: tự theo dõi chỉ 20,2% dù uống thuốc 76,9–83,5%. AI diễn giải xu hướng, chỉ đẩy lên bác sĩ khi vượt ngưỡng — giảm tải cho 14 bác sĩ/10k dân.
            </p>
          </div>
          <div className="flex flex-col gap-2 sm:items-end">
            <div className="rounded-xl bg-white p-3 text-slate-900 shadow">
              <div className="text-xs text-slate-500">API Status</div>
              <div className="text-sm font-mono">{health ? `${health.embedding_provider} · ${health.embedding_model}` : "…"}</div>
              <div className="text-xs text-slate-500">{API_BASE}</div>
            </div>
          </div>
        </div>
        <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat k="7,3%" v="~7 triệu người mắc" sub="Tiền ĐTĐ 17,8%" />
          <Stat k=">60%" v="Chưa chẩn đoán" sub=">55% đã biến chứng" />
          <Stat k="20,2%" v="Tự theo dõi tại nhà" sub="vs 76,9% uống thuốc" />
          <Stat k="14/10k" v="Bác sĩ" sub="Mục tiêu 15; Pháp 34" />
        </div>
      </div>

      {/* 5 criteria */}
      <section className="rounded-2xl border bg-white p-5 shadow-sm">
        <h2 className="text-sm font-semibold">Khung 5 tiêu chí “Pain Point chín muồi”</h2>
        <div className="mt-3 grid gap-3 sm:grid-cols-5">
          <Criteria n="1" t="Quy mô & nghiêm trọng" d="7,3% ~7M, biến chứng vĩnh viễn" ok />
          <Criteria n="2" t="Tần suất" d="Mù thông tin hàng tháng (HbA1c 3 tháng)" ok />
          <Criteria n="3" t="Chi trả thật" d="FreeStyle 2–4tr, Jio $20M — nhưng gắn hardware" warn />
          <Criteria n="4" t="Lực đẩy" d="BHYT 1/7/2025 + thiếu BS + AI đã chấp nhận" ok />
          <Criteria n="5" t="Khoảng trống" d="Thiếu WHO-RAG audit trail" ok />
        </div>
      </section>

      {/* Market evidence table */}
      <section className="rounded-2xl border bg-white p-5 shadow-sm">
        <h2 className="text-sm font-semibold">Bằng chứng định lượng — khoảng trống tái khám</h2>
        <div className="mt-3 overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b text-left text-xs text-slate-500">
              <tr>
                <th className="py-2">Nghiên cứu</th>
                <th>Uống thuốc</th>
                <th>Tự theo dõi tại nhà</th>
                <th>Tái khám</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              <tr>
                <td className="py-2 font-medium">BV Thanh Nhàn 2025 (n=104)</td>
                <td>76,9%</td>
                <td className="font-bold text-red-600">20,2%</td>
                <td>46,2%</td>
              </tr>
              <tr>
                <td className="py-2">BV Thống Nhất 2023 (n=255)</td>
                <td>83,5%</td>
                <td colSpan={2} className="text-slate-600">63,1% (gộp)</td>
              </tr>
              <tr>
                <td className="py-2">PK ĐH YTCC 2022 (n=240)</td>
                <td>—</td>
                <td>—</td>
                <td>53,75%</td>
              </tr>
            </tbody>
          </table>
          <div className="mt-2 text-xs text-slate-500">Nguồn: Tạp chí Y học Việt Nam 2022/2025; vista.gov.vn 2023. Chuẩn HbA1c mỗi 3 tháng.</div>
        </div>
      </section>

      {/* 3 tracks */}
      <section className="grid gap-4 sm:grid-cols-3">
        <TrackCard
          badge="Hướng A"
          title="Trợ lý tuân thủ"
          desc="Nhắm 20,2%. Nhập đường huyết → RAG WHO/BYT diễn giải trend, chỉ escalate khi critical/3×high."
          kpi="KPI: ≥3 lần/tuần sau 4 tuần"
          href="/tracker"
          cta="Mở nhật ký →"
          color="emerald"
        />
        <TrackCard
          badge="Hướng B"
          title="Hồ sơ trước tái khám"
          desc="Nhắm 46–53%. Tóm tắt SOAP/ADA chuẩn hóa, bán B2B2C cho phòng khám (BHYT telehealth)."
          kpi="B2B2C • PDF/MD"
          href="/previsit"
          cta="Tạo SOAP →"
          color="blue"
        />
        <TrackCard
          badge="Hướng C"
          title="WHO-RAG API"
          desc="Lớp hạ tầng truy vấn đáng tin (audit + faithfulness) cho eDoctor/Medihome tích hợp."
          kpi="POST /v1/query"
          href="/api-playground"
          cta="Thử API →"
          color="violet"
        />
      </section>

      {/* Guidelines status */}
      <section className="rounded-2xl border bg-white p-5 shadow-sm">
        <h2 className="text-sm font-semibold">Trạng thái Corpus Guideline</h2>
        {guidelines ? (
          <div className="mt-3 flex flex-wrap gap-2">
            {Object.entries(guidelines.by_source || {}).map(([k, v]) => (
              <span key={k} className="rounded-full border bg-slate-50 px-3 py-1 text-xs">
                {k}: <b>{String(v)}</b> PDFs
              </span>
            ))}
            <span className="rounded-full bg-slate-900 px-3 py-1 text-xs text-white">Tổng {guidelines.total_pdfs} PDFs</span>
          </div>
        ) : (
          <div className="mt-3 text-xs text-slate-500">Đang tải… (cần FastAPI chạy tại {API_BASE})</div>
        )}
        <div className="mt-3 text-xs text-slate-500">
          Pipeline: <code>scripts/crawl_guidelines.py</code> — WHO IRIS (DSpace 7 API), ADA, GOLD/GINA, BYT (QĐ 3319). Indexer đệ quy <code>data/raw/pdfs/**/*.pdf</code> → Chroma (
          {health?.embedding_model || "text-embedding-3-small"}).
        </div>
      </section>

      <div className="flex flex-wrap gap-2 text-xs">
        <a href="https://iris.who.int/handle/10665/325182" target="_blank" className="rounded-full border bg-white px-3 py-1 hover:bg-slate-50">WHO Classification 2019</a>
        <a href="https://iris.who.int/handle/10665/331710" target="_blank" className="rounded-full border bg-white px-3 py-1 hover:bg-slate-50">WHO HEARTS-D 2020</a>
        <a href="https://iris.who.int/handle/10665/334186" target="_blank" className="rounded-full border bg-white px-3 py-1 hover:bg-slate-50">WHO PEN 2020</a>
        <Link href="/api-playground" className="rounded-full bg-slate-900 px-3 py-1 text-white">Thử hỏi “dấu hiệu đái tháo đường type 2?” →</Link>
      </div>
    </div>
  );
}

function Stat({ k, v, sub }: { k: string; v: string; sub?: string }) {
  return (
    <div className="rounded-xl bg-white/10 p-3 backdrop-blur border border-white/20">
      <div className="text-lg font-bold">{k}</div>
      <div className="text-xs text-blue-50">{v}</div>
      {sub && <div className="text-[11px] text-blue-200">{sub}</div>}
    </div>
  );
}
function Criteria({ n, t, d, ok, warn }: { n: string; t: string; d: string; ok?: boolean; warn?: boolean }) {
  return (
    <div className={`rounded-xl border p-3 ${ok ? "bg-emerald-50 border-emerald-200" : warn ? "bg-amber-50 border-amber-200" : "bg-white"}`}>
      <div className="text-xs font-bold">{n}. {t}</div>
      <div className="mt-1 text-xs text-slate-600">{d}</div>
      <div className="mt-2 text-[11px]">{ok ? "● Mạnh" : warn ? "● Lưu ý" : ""}</div>
    </div>
  );
}
function TrackCard({ badge, title, desc, kpi, href, cta, color }: any) {
  const colorMap: any = {
    emerald: "from-emerald-500 to-teal-600",
    blue: "from-blue-500 to-indigo-600",
    violet: "from-violet-500 to-purple-600",
  };
  return (
    <div className="rounded-2xl border bg-white p-5 shadow-sm flex flex-col">
      <div className={`inline-flex w-fit rounded-full bg-gradient-to-r ${colorMap[color]} px-3 py-1 text-xs font-semibold text-white`}>{badge}</div>
      <h3 className="mt-3 text-sm font-semibold">{title}</h3>
      <p className="mt-1 text-xs text-slate-600 line-clamp-3">{desc}</p>
      <div className="mt-2 text-[11px] font-mono text-slate-500">{kpi}</div>
      <Link href={href} className="mt-4 inline-flex w-fit rounded-full bg-slate-900 px-4 py-2 text-xs font-medium text-white hover:bg-black">
        {cta}
      </Link>
    </div>
  );
}
