"use client";
import { useEffect, useState } from "react";
import { adminListCollections, adminListCollectionChunks, fmtDT } from "@/lib/api";
import { AdminTable, AdminEmpty, AdminPagination, AdminFilterBar, AdminSkeletonRows } from "@/components/AdminTable";
import { useAuth } from "@/lib/auth";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

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
  const [busy,setBusy]=useState(true);
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
    }catch(e:any){ setMsg(e.message); }finally{ setBusy(false); }
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
        <p className="text-xs text-slate-600 dark:text-slate-300">Duyệt toàn bộ corpus vector DB theo từng collection. 10 chunks/trang. Chỉ admin.</p>
        <div className="text-xs text-slate-500 dark:text-slate-400">Tổng {total} chunks · <Link href="/admin/tracing" className="text-blue-700 hover:underline font-medium dark:text-blue-300">← Traces</Link> · <Link href="/admin/tracing/memory" className="text-blue-700 hover:underline font-medium dark:text-blue-300">Memory →</Link></div>
      </div>
      <AdminFilterBar>
        <div><label className="text-xs">Collection</label>
          <select value={strategy} onChange={(e)=>setStrategy(e.target.value)} className="ml-2 rounded border px-2 py-1 text-xs">
            {collections.map((c:any)=><option key={c.strategy} value={c.strategy}>{c.strategy} ({c.count})</option>)}
            {!collections.length && <option value="structure">structure</option>}
          </select>
        </div>
        <div><label className="text-xs">Dataset</label>
          <select value={dataset} onChange={(e)=>setDataset(e.target.value)} className="ml-2 rounded border px-2 py-1 text-xs">
            <option value="all">tất cả</option>
            {datasets.map((d)=><option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        <div><label className="text-xs">Tìm kiếm</label><Input value={q} onChange={(e)=>setQ(e.target.value)} placeholder="từ khóa trong chunk" className="ml-2 w-auto text-xs" /></div>
        <Button variant="dark" size="sm" onClick={apply}>Lọc</Button>
        <Button variant="outline" size="sm" onClick={()=>{ setQ(""); setDataset("all"); setPage(1); loadChunks(1, strategy, "all", ""); }}>Xóa</Button>
      </AdminFilterBar>
      {msg && <div className="rounded-lg bg-red-50 border border-red-200 p-2 text-xs text-red-700">{msg}</div>}
      <AdminTable columns={["#", "Source", "File/dataset", "Trang", "Chunk idx", "Nội dung", "Cập nhật"]} emptyText="Không có chunk">
            {busy ? <AdminSkeletonRows cols={7} /> : (<>
            {chunks.map((c:any, i:number)=>(
              <tr key={c.id} className="border-b hover:bg-slate-50 dark:hover:bg-slate-800 align-top">
                <td className="p-2">{(page-1)*10+i+1}</td>
                <td className="p-2 max-w-[140px] truncate">{c.source_id}</td>
                <td className="p-2"><div className="max-w-[160px] truncate">{c.file_name}</div><span className="rounded-full bg-slate-100 border px-2 py-0.5 dark:bg-slate-800 dark:border-slate-700">{c.dataset||"—"}</span></td>
                <td className="p-2">{(c.page_numbers||[]).join(",")||"—"}</td>
                <td className="p-2">{c.chunk_index ?? "—"}</td>
                <td className="p-2 max-w-[380px]"><div className="line-clamp-3 whitespace-pre-wrap">{expanded[c.id] ? c.content_full : c.content_snippet}</div>
                  <button onClick={()=>toggle(c.id)} aria-expanded={!!expanded[c.id]} className="text-blue-700 hover:underline min-h-[44px] px-1 dark:text-blue-300">{expanded[c.id] ? "Thu gọn" : "Mở rộng"}</button></td>
                <td className="p-2 text-[11px] text-slate-500 dark:text-slate-400">{fmtDT(c.updated_at)}</td>
              </tr>
            ))}
            {!chunks.length && <AdminEmpty colSpan={7} text="Không có chunk" />}
            </>)}
      </AdminTable>
      <AdminPagination page={page} totalPages={totalPages} total={total} label="chunks" go={go} />
      <div className="text-[11px] text-slate-500 dark:text-slate-400">Nguồn Chroma collections: xem trước cắt 500 ký tự, bản đầy đủ 4000 ký tự.</div>
    </div>
  );
}
