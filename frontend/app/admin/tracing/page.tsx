"use client";
import { useEffect, useState } from "react";
import { adminListTraces, API_BASE } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useRouter } from "next/navigation";
import Link from "next/link";

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

  useEffect(()=>{ if(!loading && (!user || user.role!=="admin")) router.push("/"); },[user,loading]);

  const load=async(p=page)=>{
    try{
      const data=await adminListTraces({ page:p, limit:10, q: q||undefined, tone: tone||undefined });
      setTraces(data.traces); setTotal(data.total); setTotalPages(data.total_pages); setMsg("");
    }catch(e:any){ setMsg(e.message); }
  };
  useEffect(()=>{ if(user) load(1); },[user]);
  const go=(p:number)=>{ setPage(p); load(p); };

  if(loading) return <div>loading</div>;
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold">Admin — Tracing</h1>
        <p className="text-xs text-slate-600">Truy vết user query → chunks retrieved → final answer + RAGAS. Lưu 30 ngày, 10 trace/trang. Chỉ admin.</p>
        <div className="text-xs text-slate-500">{API_BASE} · Tổng {total} traces · <Link href="/admin/tracing/chunks" className="text-blue-700 hover:underline font-medium">Chunks →</Link> · <Link href="/admin/tracing/memory" className="text-blue-700 hover:underline font-medium">Memory →</Link></div>
      </div>
      <div className="rounded-2xl border bg-white dark:bg-slate-900 dark:border-slate-700 p-4 shadow-sm flex flex-wrap gap-2 items-end">
        <div><label className="text-xs">Tìm kiếm</label><input value={q} onChange={(e)=>setQ(e.target.value)} placeholder="query/answer" className="ml-2 rounded border px-2 py-1 text-xs" /></div>
        <div><label className="text-xs">Tone</label><select value={tone} onChange={(e)=>setTone(e.target.value)} className="ml-2 rounded border px-2 py-1 text-xs"><option value="">all</option><option value="diabetes">diabetes</option><option value="hypertension">hypertension</option><option value="respiratory">respiratory</option><option value="mental">mental</option><option value="balanced">balanced</option></select></div>
        <button onClick={()=>load(1)} className="rounded-full bg-slate-900 text-white px-4 py-1.5 text-xs">Lọc</button>
        <button onClick={()=>{ setQ(""); setTone(""); load(1); }} className="rounded-full border px-4 py-1.5 text-xs">Xóa</button>
      </div>
      {msg && <div className="rounded-lg bg-red-50 border border-red-200 p-2 text-xs text-red-700">{msg}</div>}
      <div className="rounded-2xl border bg-white dark:bg-slate-900 dark:border-slate-700 shadow-sm overflow-hidden">
        <table className="w-full text-xs">
          <thead className="bg-slate-50 border-b text-left"><tr><th className="p-2">Query</th><th>Answer</th><th>Tone</th><th>RAGAS</th><th>Latency</th><th>Thời gian</th></tr></thead>
          <tbody>
            {traces.map((t:any)=>(
              <tr key={t.id} className="border-b hover:bg-slate-50">
                <td className="p-2 max-w-[260px]"><Link href={`/admin/tracing/${t.id}`} className="font-medium text-blue-700 hover:underline line-clamp-2">{t.query}</Link><div className="text-[11px] text-slate-500">{t.username || t.user_id.slice(0,8)} · {t.prompt_version}</div></td>
                <td className="p-2 max-w-[260px] line-clamp-2">{t.answer?.slice(0,120) || "—"}</td>
                <td className="p-2"><span className="rounded-full bg-slate-100 border px-2 py-0.5">{t.tone}</span></td>
                <td className="p-2">{t.has_ragas ? <span className="rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 px-2 py-0.5">có</span> : <span className="rounded-full bg-amber-50 text-amber-700 border px-2 py-0.5">chưa</span>}{t.is_low_confidence? " ⚠️ low":""}</td>
                <td className="p-2">{t.total_latency_ms ? `${Math.round(t.total_latency_ms)}ms` : "—"}</td>
                <td className="p-2 text-[11px] text-slate-500">{t.created_at?.slice(0,16).replace("T"," ")}</td>
              </tr>
            ))}
            {!traces.length && <tr><td colSpan={6} className="p-6 text-center text-slate-500">Không có trace (30 ngày)</td></tr>}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between text-xs">
        <div>Trang {page}/{totalPages} · {total} traces (30 ngày)</div>
        <div className="flex gap-1">
          <button disabled={page<=1} onClick={()=>go(page-1)} className="rounded-full border px-3 py-1 disabled:opacity-50">Trước</button>
          {Array.from({length: Math.min(totalPages,5)}, (_,i)=>i+1).map(p=> <button key={p} onClick={()=>go(p)} className={`rounded-full px-3 py-1 ${p===page?"bg-slate-900 text-white":"border"}`}>{p}</button>)}
          <button disabled={page>=totalPages} onClick={()=>go(page+1)} className="rounded-full border px-3 py-1 disabled:opacity-50">Sau</button>
        </div>
      </div>
    </div>
  );
}
