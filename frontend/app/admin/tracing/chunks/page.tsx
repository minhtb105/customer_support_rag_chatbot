"use client";
import { useEffect, useState } from "react";
import { adminListCollections, adminListCollectionChunks, API_BASE } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function ChunksPage(){
  const { user, loading } = useAuth();
  const router = useRouter();
  const [collections,setCollections]=useState<any[]>([]);
  const [strategy,setStrategy]=useState("structure");
  const [datasets,setDatasets]=useState<string[]>([]);
  const [dataset,setDataset]=useState("all");
  const [q,setQ]=useState("");
  const [chunks,setChunks]=useState<any[]>([]);
  const [total,setTotal]=useState(0);
  const [page,setPage]=useState(1);
  const [totalPages,setTotalPages]=useState(0);
  const [msg,setMsg]=useState("");
  const [expanded,setExpanded]=useState<Record<string,boolean>>({});

  useEffect(()=>{ if(!loading && (!user || user.role!=="admin")) router.push("/"); },[user,loading]);

  const loadCollections=async()=>{
    try{
      const data=await adminListCollections();
      setCollections(data.collections||[]);
      const cur=(data.collections||[]).find((c:any)=>c.strategy===strategy);
      setDatasets(cur?.datasets||[]);
    }catch(e:any){ setMsg(e.message); }
  };

  const loadChunks=async(p=page, strat=strategy, ds=dataset, qq=q)=>{
    try{
      const data=await adminListCollectionChunks(strat, { page:p, limit:10, dataset: ds==="all"?"":ds, q: qq||undefined });
      setChunks(data.chunks); setTotal(data.total); setTotalPages(data.total_pages); setMsg("");
    }catch(e:any){ setMsg(e.message); }
  };

  useEffect(()=>{ if(user){ loadCollections(); } },[user]);
  useEffect(()=>{
    const cur=collections.find((c:any)=>c.strategy===strategy);
    setDatasets(cur?.datasets||[]);
    setDataset("all"); setPage(1);
    if(user) loadChunks(1, strategy, "all", q);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[strategy]);
  useEffect(()=>{ if(user) loadChunks(1); },[user]);

  const go=(p:number)=>{ setPage(p); loadChunks(p); };
  const apply=()=>{ setPage(1); loadChunks(1); };
  const toggle=(id:string)=>setExpanded(s=>({ ...s, [id]: !s[id] }));

  if(loading) return <div>loading</div>;
  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-bold">Admin — Chunks</h1>
        <p className="text-xs text-slate-600">Duyệt toàn bộ corpus vector DB theo từng collection. 10 chunks/trang. Chỉ admin.</p>
        <div className="text-xs text-slate-500">{API_BASE} · Tổng {total} chunks · <Link href="/admin/tracing" className="text-blue-700 hover:underline font-medium">← Traces</Link> · <Link href="/admin/tracing/memory" className="text-blue-700 hover:underline font-medium">Memory →</Link></div>
      </div>
      <div className="rounded-2xl border bg-white p-4 shadow-sm flex flex-wrap gap-2 items-end">
        <div><label className="text-xs">Collection</label>
          <select value={strategy} onChange={(e)=>setStrategy(e.target.value)} className="ml-2 rounded border px-2 py-1 text-xs">
            {collections.map((c:any)=><option key={c.strategy} value={c.strategy}>{c.strategy} ({c.count})</option>)}
            {!collections.length && <option value="structure">structure</option>}
          </select>
        </div>
        <div><label className="text-xs">Dataset</label>
          <select value={dataset} onChange={(e)=>setDataset(e.target.value)} className="ml-2 rounded border px-2 py-1 text-xs">
            <option value="all">all</option>
            {datasets.map((d)=><option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        <div><label className="text-xs">Tìm kiếm</label><input value={q} onChange={(e)=>setQ(e.target.value)} placeholder="từ khóa trong chunk" className="ml-2 rounded border px-2 py-1 text-xs" /></div>
        <button onClick={apply} className="rounded-full bg-slate-900 text-white px-4 py-1.5 text-xs">Lọc</button>
        <button onClick={()=>{ setQ(""); setDataset("all"); setPage(1); loadChunks(1, strategy, "all", ""); }} className="rounded-full border px-4 py-1.5 text-xs">Xóa</button>
      </div>
      {msg && <div className="rounded-lg bg-red-50 border border-red-200 p-2 text-xs text-red-700">{msg}</div>}
      <div className="rounded-2xl border bg-white shadow-sm overflow-hidden">
        <table className="w-full text-xs">
          <thead className="bg-slate-50 border-b text-left"><tr><th className="p-2">#</th><th>Source</th><th>File/dataset</th><th>Trang</th><th>Chunk idx</th><th>Nội dung</th><th>Cập nhật</th></tr></thead>
          <tbody>
            {chunks.map((c:any, i:number)=>(
              <tr key={c.id} className="border-b hover:bg-slate-50 align-top">
                <td className="p-2">{(page-1)*10+i+1}</td>
                <td className="p-2 max-w-[140px] truncate">{c.source_id}</td>
                <td className="p-2"><div className="max-w-[160px] truncate">{c.file_name}</div><span className="rounded-full bg-slate-100 border px-2 py-0.5">{c.dataset||"—"}</span></td>
                <td className="p-2">{(c.page_numbers||[]).join(",")||"—"}</td>
                <td className="p-2">{c.chunk_index ?? "—"}</td>
                <td className="p-2 max-w-[380px]"><div className="line-clamp-3 whitespace-pre-wrap">{expanded[c.id] ? c.content_full : c.content_snippet}</div>
                  <button onClick={()=>toggle(c.id)} className="text-blue-700 hover:underline">{expanded[c.id] ? "Thu gọn" : "Expand"}</button></td>
                <td className="p-2 text-[11px] text-slate-500">{c.updated_at ? String(c.updated_at).slice(0,16).replace("T"," ") : "—"}</td>
              </tr>
            ))}
            {!chunks.length && <tr><td colSpan={7} className="p-6 text-center text-slate-500">Không có chunk</td></tr>}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between text-xs">
        <div>Trang {page}/{totalPages||1} · {total} chunks</div>
        <div className="flex gap-1">
          <button disabled={page<=1} onClick={()=>go(page-1)} className="rounded-full border px-3 py-1 disabled:opacity-50">Trước</button>
          {Array.from({length: Math.min(totalPages,5)}, (_,i)=>i+1).map(p=> <button key={p} onClick={()=>go(p)} className={`rounded-full px-3 py-1 ${p===page?"bg-slate-900 text-white":"border"}`}>{p}</button>)}
          <button disabled={page>=totalPages} onClick={()=>go(page+1)} className="rounded-full border px-3 py-1 disabled:opacity-50">Sau</button>
        </div>
      </div>
    </div>
  );
}
