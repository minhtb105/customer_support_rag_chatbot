"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { getHealth, getGuidelinesStatus, API_BASE } from "@/lib/api";
import { Play, ArrowRight, Check, ShieldCheck } from "lucide-react";

export default function OverviewPage() {
  const [health, setHealth] = useState<any>(null);
  const [guidelines, setGuidelines] = useState<any>(null);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: "offline" }));
    getGuidelinesStatus().then(setGuidelines).catch(() => {});
  }, []);

  return (
    <div className="space-y-6">
      {/* Sticky CTA bar (dan toi #thu). TODO(user): doi text/primary-href khi co pricing that */}
      <div className="sticky top-[57px] z-30 flex flex-wrap items-center justify-between gap-2 rounded-2xl border border-blue-200 bg-white/90 px-4 py-2 shadow-sm backdrop-blur dark:bg-slate-900/90 dark:border-slate-700">
        <div className="text-xs text-slate-600 dark:text-slate-300">Bản demo nội bộ — miễn phí dùng thử luồng bệnh nhân & bác sĩ</div>
        <div className="flex gap-2">
          <a href="#demo" className="inline-flex min-h-[44px] items-center gap-1 rounded-full border px-4 py-2 text-xs font-medium hover:bg-slate-50 dark:hover:bg-slate-800">
            <Play size={14} /> Xem video demo
          </a>
          <Link href="/register" className="inline-flex min-h-[44px] items-center gap-1 rounded-full bg-slate-900 px-4 py-2 text-xs font-medium text-white hover:bg-black dark:bg-slate-100 dark:text-slate-900">
            Dùng thử miễn phí <ArrowRight size={14} />
          </Link>
        </div>
      </div>

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
      <section className="rounded-2xl border bg-white p-5 shadow-sm dark:bg-slate-900 dark:border-slate-700">
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
      <section className="rounded-2xl border bg-white p-5 shadow-sm dark:bg-slate-900 dark:border-slate-700">
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
      <section className="rounded-2xl border bg-white p-5 shadow-sm dark:bg-slate-900 dark:border-slate-700">
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

      {/* Video demo (file phuc vu tu /public/demo.webm, canonical giu o docs/demo/) */}
      <section id="demo" className="rounded-2xl border bg-white p-5 shadow-sm dark:bg-slate-900 dark:border-slate-700">
        <h2 className="text-sm font-semibold">Xem demo 4 phút: bệnh nhân → bác sĩ → audit</h2>
        <p className="mt-1 text-xs text-slate-500">Nhập 280 → banner đỏ • trend 3 ngày → chat hỏi ngữ cảnh • safeguard kháng sinh • bác sĩ xem sparkline 90 ngày + SOAP</p>
        <video controls preload="none" src="/demo.webm" className="mt-3 aspect-video w-full rounded-xl bg-black">
          Trình duyệt không hỗ trợ video. Tải tại <a href="/demo.webm" className="underline">demo.webm</a>.
        </video>
        <div className="mt-2 text-xs text-slate-500">Kịch bản chi tiết: <code>docs/demo/script.md</code> • File gốc: <code>docs/demo/demo.webm</code></div>
      </section>

      {/* Social proof strip — TODO(user): thay 3 o placeholder bang doi tac that khi co */}
      <section className="rounded-2xl border bg-white p-5 shadow-sm dark:bg-slate-900 dark:border-slate-700">
        <h2 className="text-sm font-semibold">Đang thử nghiệm nội bộ cùng phòng khám đối tác</h2>
        <div className="mt-3 grid gap-3 sm:grid-cols-3">
          <Proof logo="PK" name="Phòng khám đối tác (ẩn danh)" note="Bản nháp — chờ xác nhận" />
          <Proof logo="BV" name="Bệnh viện thử nghiệm (ẩn danh)" note="Bản nháp — chờ xác nhận" />
          <Proof logo="API" name="Đối tác tích hợp WHO-RAG API" note="Bản nháp — chờ xác nhận" />
        </div>
      </section>

      {/* Pricing — TODO(user): dien gia that; hien tai la placeholder "Lien he", khong gia gia */}
      <section id="thu" className="rounded-2xl border bg-white p-5 shadow-sm dark:bg-slate-900 dark:border-slate-700">
        <h2 className="text-sm font-semibold">Gói sử dụng <span className="ml-1 rounded-full bg-amber-100 px-2 py-0.5 text-[11px] font-normal text-amber-800">Giá dự kiến — liên hệ xác nhận</span></h2>
        <div className="mt-3 grid gap-3 sm:grid-cols-3">
          <PriceCard
            name="Cá nhân"
            price="Liên hệ"
            desc="Người bệnh tự theo dõi đường huyết tại nhà."
            feats={["Nhật ký đường huyết + cảnh báo", "Chat hỏi ngữ cảnh (FQG)", "Nhắc đo qua giọng nói"]}
            href="/register"
            cta="Dùng thử miễn phí"
          />
          <PriceCard
            name="Phòng khám"
            price="Liên hệ"
            desc="Bác sĩ xem hàng đợi ưu tiên + hồ sơ trước tái khám."
            feats={["Smart queue critical → safe", "Sparkline 90 ngày + SOAP", "Audit [Xem log] từng nhận định"]}
            href="/register"
            cta="Đăng ký demo"
            hot
          />
          <PriceCard
            name="API Doanh nghiệp"
            price="Liên hệ"
            desc="Tích hợp WHO-RAG đáng tin cho eDoctor/Medihome."
            feats={["POST /v1/query có citation", "Faithfulness + tracing", "Tài liệu kỹ thuật đầy đủ"]}
            href="/api-playground"
            cta="Thử API"
          />
        </div>
      </section>

      {/* Testimonials — TODO(user): thay bang danh gia that co ten/vai tro khi co; hien an danh + nhan ban nhap */}
      <section className="rounded-2xl border bg-white p-5 shadow-sm dark:bg-slate-900 dark:border-slate-700">
        <h2 className="text-sm font-semibold">Người dùng thử nghiệm nói gì <span className="ml-1 rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-normal text-slate-500">Bản nháp — chờ nội dung thật</span></h2>
        <div className="mt-3 grid gap-3 sm:grid-cols-3">
          <Quote who="Người bệnh thử nghiệm (ẩn danh)" text="Nhập chỉ số mỗi tối, khi đường huyết vọt lên là app hỏi ngay ăn gì — thấy yên tâm hơn giữa 2 lần tái khám." />
          <Quote who="Bác sĩ thử nghiệm (ẩn danh)" text="Sparkline 90 ngày kèm chú thích [Xem log] giúp tôi nắm bệnh nhân trong 1 phút thay vì lật cả xấp giấy." />
          <Quote who="Điều dưỡng thử nghiệm (ẩn danh)" text="Cụ già nói không cần gõ — tính năng giọng nói đọc bài học 60 giây rất hợp." />
        </div>
      </section>

      {/* Final CTA + disclaimer demo-only */}
      <section className="rounded-2xl bg-gradient-to-br from-blue-600 to-indigo-600 p-6 text-white sm:p-8">
        <h2 className="text-xl font-bold">Bắt đầu theo dõi đường huyết ngay hôm nay</h2>
        <p className="mt-1 text-sm text-blue-100">Đăng ký 1 phút — nhập chỉ số đầu tiên, xem AI diễn giải trend, mời bác sĩ cùng theo dõi.</p>
        <div className="mt-4 flex flex-wrap gap-2">
          <Link href="/register" className="inline-flex min-h-[44px] items-center gap-1 rounded-full bg-white px-5 py-2 text-sm font-medium text-slate-900 hover:bg-blue-50">
            Dùng thử miễn phí <ArrowRight size={16} />
          </Link>
          <Link href="/tracker" className="inline-flex min-h-[44px] items-center rounded-full border border-white/40 px-5 py-2 text-sm hover:bg-white/10">
            Mở nhật ký
          </Link>
        </div>
        <p className="mt-4 flex items-start gap-1 text-[11px] text-blue-200">
          <ShieldCheck size={14} className="mt-0.5 shrink-0" />
          <span>Bản demo nội bộ — nội dung mang tính giáo dục, không thay thế chỉ định bác sĩ. Đánh giá đạt mục tiêu theo HbA1c &lt; 7% (BYT QĐ5481/2020 + ADA 2024); kế hoạch điều trị luôn để trống cho bác sĩ quyết định.</span>
        </p>
      </section>
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
    <div className="rounded-2xl border bg-white p-5 shadow-sm flex flex-col dark:bg-slate-900 dark:border-slate-700">
      <div className={`inline-flex w-fit rounded-full bg-gradient-to-r ${colorMap[color]} px-3 py-1 text-xs font-semibold text-white`}>{badge}</div>
      <h3 className="mt-3 text-sm font-semibold">{title}</h3>
      <p className="mt-1 text-sm text-slate-600 dark:text-slate-300 line-clamp-3">{desc}</p>
      <div className="mt-2 text-[11px] font-mono text-slate-500 dark:text-slate-400">{kpi}</div>
      <Link href={href} className="mt-4 inline-flex w-fit rounded-full bg-slate-900 px-4 py-2 text-xs font-medium text-white hover:bg-black dark:bg-slate-100 dark:text-slate-900 min-h-[44px] items-center">
        {cta}
      </Link>
    </div>
  );
}
function Proof({ logo, name, note }: { logo: string; name: string; note: string }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border bg-slate-50 p-3 dark:bg-slate-800 dark:border-slate-700">
      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-slate-900 text-xs font-bold text-white dark:bg-slate-100 dark:text-slate-900">{logo}</div>
      <div>
        <div className="text-xs font-medium">{name}</div>
        <div className="text-[11px] text-slate-500">{note}</div>
      </div>
    </div>
  );
}
function PriceCard({ name, price, desc, feats, href, cta, hot }: { name: string; price: string; desc: string; feats: string[]; href: string; cta: string; hot?: boolean }) {
  return (
    <div className={`flex flex-col rounded-xl border p-4 ${hot ? "border-blue-500 bg-blue-50/50 dark:bg-slate-800 dark:border-blue-400" : "bg-white dark:bg-slate-900 dark:border-slate-700"}`}>
      {hot && <div className="mb-2 w-fit rounded-full bg-blue-600 px-2 py-0.5 text-[11px] font-semibold text-white">Phổ biến nhất</div>}
      <div className="text-sm font-semibold">{name}</div>
      <div className="mt-1 text-lg font-bold">{price}</div>
      <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">{desc}</p>
      <ul className="mt-2 space-y-1">
        {feats.map((f) => (
          <li key={f} className="flex items-start gap-1 text-xs text-slate-600 dark:text-slate-300">
            <Check size={14} className="mt-0.5 shrink-0 text-emerald-600" /> {f}
          </li>
        ))}
      </ul>
      <Link href={href} className="mt-4 inline-flex w-fit rounded-full bg-slate-900 px-4 py-2 text-xs font-medium text-white hover:bg-black dark:bg-slate-100 dark:text-slate-900 min-h-[44px] items-center">
        {cta}
      </Link>
    </div>
  );
}
function Quote({ who, text }: { who: string; text: string }) {
  return (
    <figure className="rounded-xl border bg-slate-50 p-4 dark:bg-slate-800 dark:border-slate-700">
      <blockquote className="text-xs leading-relaxed text-slate-700 dark:text-slate-200">“{text}”</blockquote>
      <figcaption className="mt-2 text-[11px] font-medium text-slate-500">— {who}</figcaption>
    </figure>
  );
}
