"use client";
import { useEffect, useState } from "react";
import { adminListTraces, API_BASE, fmtDT } from "@/lib/api";
import { AdminTable, AdminEmpty, AdminPagination, AdminFilterBar, AdminSkeletonRows } from "@/components/AdminTable";
import { useAuth } from "@/lib/auth";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { TriangleAlert } from "lucide-react";

export default function TracingListPage(){
  const { user, loading } = useAuth();
  const router = useRouter();
  const [traces,setTraces]=useState<any[]>([]);
  const [total,setTotal]=useState(0);
  const [page,setPage]=useState(1);
  const [q,setQ]=useState("");
  const [tone,setTone]=useState("");
  const [totalPages,setTotalPages]=useState(1);
  const [msg,setMsg]=useState("");
  const [busy,setBusy]=useState(true);

  useEffect(()=>{ if(!loading && (!user || user.role!=="admin")) router.push("/"); },[user,loading]);

  const load=async(p=page)=>{
    try{
      const data=await adminListTraces({ page:p, limit:10, q: q||undefined, tone: tone||undefined });
      setTraces(data.traces); setTotal(data.total); setTotalPages(data.total_pages); setMsg("");
    }catch(e:any){ setMsg(e.message); }finally{ setBusy(false); }
  };
  useEffect(()=>{ if(user) load(1); },[user]);
  const go=(p:number)=>{ setPage(p); load(p); };

  if(loading) return <div>loading</div>;
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold">Admin — Tracing</h1>
        <p className="text-xs text-slate-600 dark:text-slate-300">Truy vết user query → chunks retrieved → final answer + RAGAS. Lưu 30 ngày, 10 trace/trang. Chỉ admin.</p>
        <div className="text-xs text-slate-500 dark:text-slate-400">{API_BASE} · Tổng {total} traces · <Link href="/admin/tracing/chunks" className="text-blue-700 hover:underline font-medium dark:text-blue-300">Chunks →</Link> · <Link href="/admin/tracing/memory" className="text-blue-700 hover:underline font-medium dark:text-blue-300">Memory →</Link></div>
      </div>
      <AdminFilterBar>
        <div><label className="text-xs">Tìm kiếm</label><input value={q} onChange={(e)=>setQ(e.target.value)} placeholder="câu hỏi/câu trả lời" className="ml-2 rounded border px-2 py-1 text-xs" /></div>
        <div><label className="text-xs">Tone</label><select value={tone} onChange={(e)=>setTone(e.target.value)} className="ml-2 rounded border px-2 py-1 text-xs"><option value="">tất cả</option><option value="diabetes">diabetes</option><option value="hypertension">hypertension</option><option value="respiratory">respiratory</option><option value="mental">mental</option><option value="balanced">balanced</option></select></div>
        <button onClick={()=>load(1)} className="rounded-full bg-slate-900 text-white px-4 py-1.5 text-xs min-h-[44px]">Lọc</button>
        <button onClick={()=>{ setQ(""); setTone(""); load(1); }} className="rounded-full border px-4 py-1.5 text-xs min-h-[44px]">Xóa</button>
      </AdminFilterBar>
      {msg && <div className="rounded-lg bg-red-50 border border-red-200 p-2 text-xs text-red-700">{msg}</div>}
      <AdminTable columns={["Query", "Answer", "Tone", "RAGAS", "Latency", "Thời gian"]} emptyText="Không có trace (30 ngày)">
          {busy ? <AdminSkeletonRows cols={6} /> : (<>
          {traces.map((t:any)=>(
              <tr key={t.id} className="border-b hover:bg-slate-50 dark:hover:bg-slate-800">
                <td className="p-2 max-w-[260px]"><Link href={`/admin/tracing/${t.id}`} className="font-medium text-blue-700 hover:underline line-clamp-2 dark:text-blue-300">{t.query}</Link><div className="text-[11px] text-slate-500 dark:text-slate-400">{t.username || t.user_id.slice(0,8)} · {t.prompt_version}</div></td>
                <td className="p-2 max-w-[260px] line-clamp-2">{t.answer?.slice(0,120) || "—"}</td>
                <td className="p-2"><span className="rounded-full bg-slate-100 border px-2 py-0.5 dark:bg-slate-800 dark:border-slate-700">{t.tone}</span></td>
                <td className="p-2">{t.has_ragas ? <span className="rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 px-2 py-0.5 dark:bg-emerald-950 dark:text-emerald-200 dark:border-emerald-800">có</span> : <span className="rounded-full bg-amber-50 text-amber-700 border px-2 py-0.5 dark:bg-amber-950 dark:text-amber-200 dark:border-amber-800">chưa</span>}{t.is_low_confidence? <span className="inline-flex items-center gap-1"><TriangleAlert size={13} aria-hidden /> low</span>:""}</td>
                <td className="p-2">{t.total_latency_ms ? `${Math.round(t.total_latency_ms)}ms` : "—"}</td>
                <td className="p-2 text-[11px] text-slate-500 dark:text-slate-400">{fmtDT(t.created_at)}</td>
              </tr>
            ))}
            {!traces.length && <AdminEmpty colSpan={6} text="Không có trace (30 ngày)" />}
          </>)}
      </AdminTable>
      <AdminPagination page={page} totalPages={totalPages} total={total} label="traces (30 ngày)" go={go} />
    </div>
  );
}
