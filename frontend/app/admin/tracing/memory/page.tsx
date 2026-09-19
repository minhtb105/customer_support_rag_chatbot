"use client";
import { useEffect, useState } from "react";
import { adminListUserFacts, adminListUsers, API_BASE } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useRouter } from "next/navigation";
import Link from "next/link";

const FACT_TYPES = ["all", "medication", "symptom", "condition", "allergy", "lifestyle", "general"];

export default function MemoryFactsPage() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [users, setUsers] = useState<any[]>([]);
  const [userId, setUserId] = useState("");
  const [factType, setFactType] = useState("all");
  const [q, setQ] = useState("");
  const [facts, setFacts] = useState<any[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  const [msg, setMsg] = useState("");
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  useEffect(() => { if (!loading && (!user || user.role !== "admin")) router.push("/"); }, [user, loading]);

  useEffect(() => {
    if (!user) return;
    adminListUsers().then((d: any) => {
      const list = d.users || d || [];
      if (Array.isArray(list)) setUsers(list);
    }).catch(() => {});
  }, [user]);

  const load = async (p = page) => {
    if (!userId.trim()) { setMsg("Nhập hoặc chọn user_id trước"); return; }
    try {
      const data = await adminListUserFacts(userId.trim(), { page: p, limit: 10, fact_type: factType, q: q || undefined });
      setFacts(data.facts || []); setTotal(data.total || 0); setTotalPages(data.total_pages || 0); setPage(data.page || p); setMsg("");
    } catch (e: any) { setMsg(e.message); }
  };

  const apply = () => { setPage(1); load(1); };
  const go = (p: number) => { setPage(p); load(p); };
  const toggle = (id: string) => setExpanded((s) => ({ ...s, [id]: !s[id] }));

  if (loading) return <div>loading</div>;
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold">Admin — Memory</h1>
        <p className="text-xs text-slate-600">Xem long-term facts theo user. 10 facts/trang. Chỉ admin.</p>
        <div className="text-xs text-slate-500">{API_BASE} · Tổng {total} facts · <Link href="/admin/tracing" className="text-blue-700 hover:underline font-medium">← Traces</Link> · <Link href="/admin/tracing/chunks" className="text-blue-700 hover:underline font-medium">Chunks →</Link></div>
      </div>
      <div className="rounded-2xl border bg-white p-4 shadow-sm flex flex-wrap gap-2 items-end">
        <div>
          <label className="text-xs">User</label>
          <select value={userId} onChange={(e) => setUserId(e.target.value)} className="ml-2 rounded border px-2 py-1 text-xs max-w-[220px]">
            <option value="">— chọn —</option>
            {users.map((u: any) => (
              <option key={u.id || u.user_id || u.username} value={u.user_id || u.username || u.id}>{u.username || u.user_id || u.id}</option>
            ))}
          </select>
        </div>
        <div><label className="text-xs">hoặc nhập tay</label><input value={userId} onChange={(e) => setUserId(e.target.value)} placeholder="user_id" className="ml-2 rounded border px-2 py-1 text-xs" /></div>
        <div><label className="text-xs">Loại fact</label>
          <select value={factType} onChange={(e) => setFactType(e.target.value)} className="ml-2 rounded border px-2 py-1 text-xs">
            {FACT_TYPES.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
        </div>
        <div><label className="text-xs">Tìm kiếm</label><input value={q} onChange={(e) => setQ(e.target.value)} placeholder="từ khóa trong fact" className="ml-2 rounded border px-2 py-1 text-xs" /></div>
        <button onClick={apply} className="rounded-full bg-slate-900 text-white px-4 py-1.5 text-xs">Lọc</button>
        <button onClick={() => { setQ(""); setFactType("all"); setPage(1); }} className="rounded-full border px-4 py-1.5 text-xs">Xóa</button>
      </div>
      {msg && <div className="rounded-lg bg-red-50 border border-red-200 p-2 text-xs text-red-700">{msg}</div>}
      <div className="rounded-2xl border bg-white shadow-sm overflow-hidden">
        <table className="w-full text-xs">
          <thead className="bg-slate-50 border-b text-left"><tr><th className="p-2">#</th><th>Fact type</th><th>Entities</th><th>Nội dung</th><th>Conf.</th><th>Weight</th><th>Source</th></tr></thead>
          <tbody>
            {facts.map((f: any, i: number) => (
              <tr key={f.fact_id} className="border-b hover:bg-slate-50 align-top">
                <td className="p-2">{(page - 1) * 10 + i + 1}</td>
                <td className="p-2"><span className="rounded-full bg-violet-50 border border-violet-200 px-2 py-0.5">{f.fact_type || "—"}</span></td>
                <td className="p-2 max-w-[140px] truncate">{(f.entities || []).join(", ") || "—"}</td>
                <td className="p-2 max-w-[380px]"><div className="line-clamp-3 whitespace-pre-wrap">{expanded[f.fact_id] ? f.text_full : f.text_snippet}</div>
                  <button onClick={() => toggle(f.fact_id)} className="text-blue-700 hover:underline">{expanded[f.fact_id] ? "Thu gọn" : "Expand"}</button></td>
                <td className="p-2">{f.confidence ?? "—"}</td>
                <td className="p-2">{f.temporal_weight != null ? Number(f.temporal_weight).toFixed(3) : "—"}</td>
                <td className="p-2 text-[11px] text-slate-500">{f.source || "—"}</td>
              </tr>
            ))}
            {!facts.length && <tr><td colSpan={7} className="p-6 text-center text-slate-500">Không có fact — chọn user rồi bấm Lọc</td></tr>}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between text-xs">
        <div>Trang {page}/{totalPages || 1} · {total} facts</div>
        <div className="flex gap-1">
          <button disabled={page <= 1} onClick={() => go(page - 1)} className="rounded-full border px-3 py-1 disabled:opacity-50">Trước</button>
          {Array.from({ length: Math.min(totalPages, 5) }, (_, i) => i + 1).map((p) => <button key={p} onClick={() => go(p)} className={`rounded-full px-3 py-1 ${p === page ? "bg-slate-900 text-white" : "border"}`}>{p}</button>)}
          <button disabled={page >= totalPages} onClick={() => go(page + 1)} className="rounded-full border px-3 py-1 disabled:opacity-50">Sau</button>
        </div>
      </div>
    </div>
  );
}
