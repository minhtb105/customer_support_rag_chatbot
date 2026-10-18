"use client";
import { useEffect, useState } from "react";
import { adminListUserFacts, adminListUsers } from "@/lib/api";
import { AdminTable, AdminEmpty, AdminPagination, AdminFilterBar, AdminSkeletonRows } from "@/components/AdminTable";
import { useAuth } from "@/lib/auth";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

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
  const [busy, setBusy] = useState(false);
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
    setBusy(true);
    try {
      const data = await adminListUserFacts(userId.trim(), { page: p, limit: 10, fact_type: factType, q: q || undefined });
      setFacts(data.facts || []); setTotal(data.total || 0); setTotalPages(data.total_pages || 0); setPage(data.page || p); setMsg("");
    } catch (e: any) { setMsg(e.message); } finally { setBusy(false); }
  };

  const apply = () => { setPage(1); load(1); };
  const go = (p: number) => { setPage(p); load(p); };
  const toggle = (id: string) => setExpanded((s) => ({ ...s, [id]: !s[id] }));

  if (loading) return <div>loading</div>;
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold">Admin — Memory</h1>
        <p className="text-xs text-slate-600 dark:text-slate-300">Xem long-term facts theo user. 10 facts/trang. Chỉ admin.</p>
        <div className="text-xs text-slate-500 dark:text-slate-400">Tổng {total} facts · <Link href="/admin/tracing" className="text-blue-700 hover:underline font-medium dark:text-blue-300">← Traces</Link> · <Link href="/admin/tracing/chunks" className="text-blue-700 hover:underline font-medium dark:text-blue-300">Chunks →</Link></div>
      </div>
      <AdminFilterBar>
        <div>
          <label className="text-xs">User</label>
          <select value={userId} onChange={(e) => setUserId(e.target.value)} className="ml-2 rounded border px-2 py-1 text-xs max-w-[220px]">
            <option value="">— chọn —</option>
            {users.map((u: any) => (
              <option key={u.id || u.user_id || u.username} value={u.user_id || u.username || u.id}>{u.username || u.user_id || u.id}</option>
            ))}
          </select>
        </div>
        <div><label className="text-xs">hoặc nhập tay</label><Input value={userId} onChange={(e) => setUserId(e.target.value)} placeholder="nhập ID người dùng" className="ml-2 w-auto text-xs" /></div>
        <div><label className="text-xs">Loại fact</label>
          <select value={factType} onChange={(e) => setFactType(e.target.value)} className="ml-2 rounded border px-2 py-1 text-xs">
            {FACT_TYPES.map((f) => <option key={f} value={f}>{f === "all" ? "tất cả" : f}</option>)}
          </select>
        </div>
        <div><label className="text-xs">Tìm kiếm</label><Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="từ khóa trong fact" className="ml-2 w-auto text-xs" /></div>
        <Button variant="dark" size="sm" onClick={apply}>Lọc</Button>
        <Button variant="outline" size="sm" onClick={() => { setQ(""); setFactType("all"); setPage(1); }}>Xóa</Button>
      </AdminFilterBar>
      {msg && <div className="rounded-lg bg-red-50 border border-red-200 p-2 text-xs text-red-700">{msg}</div>}
      <AdminTable columns={["#", "Fact type", "Entities", "Nội dung", "Conf.", "Weight", "Source"]} emptyText="Không có fact — chọn user rồi bấm Lọc">
            {busy ? <AdminSkeletonRows cols={7} /> : (<>
            {facts.map((f: any, i: number) => (
              <tr key={f.fact_id} className="border-b hover:bg-slate-50 dark:hover:bg-slate-800 align-top">
                <td className="p-2">{(page - 1) * 10 + i + 1}</td>
                <td className="p-2"><span className="rounded-full bg-violet-50 border border-violet-200 px-2 py-0.5">{f.fact_type || "—"}</span></td>
                <td className="p-2 max-w-[140px] truncate">{(f.entities || []).join(", ") || "—"}</td>
                <td className="p-2 max-w-[380px]"><div className="line-clamp-3 whitespace-pre-wrap">{expanded[f.fact_id] ? f.text_full : f.text_snippet}</div>
                  <button onClick={() => toggle(f.fact_id)} aria-expanded={!!expanded[f.fact_id]} className="text-blue-700 hover:underline min-h-[44px] px-1 dark:text-blue-300">{expanded[f.fact_id] ? "Thu gọn" : "Mở rộng"}</button></td>
                <td className="p-2">{f.confidence ?? "—"}</td>
                <td className="p-2">{f.temporal_weight != null ? Number(f.temporal_weight).toFixed(3) : "—"}</td>
                <td className="p-2 text-[11px] text-slate-500 dark:text-slate-400">{f.source || "—"}</td>
              </tr>
            ))}
            {!facts.length && <AdminEmpty colSpan={7} text="Không có fact — chọn user rồi bấm Lọc" />}
            </>)}
      </AdminTable>
      <AdminPagination page={page} totalPages={totalPages} total={total} label="facts" go={go} />
    </div>
  );
}
