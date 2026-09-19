"use client";
import { useEffect, useState } from "react";
import { adminListPrompts, adminGetPromptTone, adminCreatePromptDraft, adminApprovePrompt, adminRejectPrompt, adminDryRun } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useRouter } from "next/navigation";

const tones = ["strict","friendly","balanced","diabetes","hypertension","respiratory","mental","soap","who_rag","evaluation"];

export default function PromptsPage(){
  const { user, loading } = useAuth();
  const router = useRouter();
  const [tone,setTone]=useState("diabetes");
  const [data,setData]=useState<any>(null);
  const [draft,setDraft]=useState("");
  const [desc,setDesc]=useState("");
  const [msg,setMsg]=useState("");
  const [dryQueries,setDryQueries]=useState("");
  const [dryRes,setDryRes]=useState<any>(null);
  const [dryLoading,setDryLoading]=useState(false);

  useEffect(()=>{ if(!loading && (!user || user.role!=="admin")) router.push("/"); },[user,loading]);
  const load=async(t=tone)=>{
    try{ const d=await adminGetPromptTone(t); setData(d); setDraft(d.active?.text || ""); }catch(e:any){ setMsg(e.message); }
  };
  useEffect(()=>{ if(user) load(); },[user]);

  const create=async()=>{
    try{ const r=await adminCreatePromptDraft(tone, draft, desc); setMsg(`Draft created ${r.version} — pending_approval`); load(); }catch(e:any){ setMsg(e.message); }
  };
  const approve=async(v:string)=>{
    try{ await adminApprovePrompt(tone, v); setMsg(`Approved ${v}`); load(); }catch(e:any){ setMsg(e.message); }
  };
  const reject=async(v:string)=>{
    try{ await adminRejectPrompt(tone, v); setMsg(`Rejected ${v}`); load(); }catch(e:any){ setMsg(e.message); }
  };
  const doDry=async()=>{
    setDryLoading(true);
    try{
      const qs = dryQueries.trim()? dryQueries.split("\n").map(s=>s.trim()).filter(Boolean) : [];
      const r=await adminDryRun(tone, draft, qs);
      setDryRes(r);
    }catch(e:any){ setMsg(e.message); }finally{ setDryLoading(false); }
  };

  if(loading) return <div>loading</div>;
  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold">Admin — Prompt Versioning (2-step approval + dry-run)</h1>
      <p className="text-xs text-slate-600">Mỗi tone có 1 active, các draft ở pending_approval. Cần dry-run 5 queries golden hoặc tự nhập trước khi Approve.</p>
      <div className="flex gap-2">
        {tones.map(t=> <button key={t} onClick={()=>{setTone(t); load(t);}} className={`rounded-full px-3 py-1 text-xs border ${tone===t?"bg-slate-900 text-white":"bg-white"}`}>{t}</button>)}
      </div>
      {data && (
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="rounded-2xl border bg-white p-4 shadow-sm">
            <h2 className="text-sm font-semibold">Active — {tone} ({data.active?.version})</h2>
            <pre className="mt-2 bg-slate-50 border p-3 rounded text-xs whitespace-pre-wrap max-h-64 overflow-auto">{data.active?.text || "—"}</pre>
            <div className="text-[11px] text-slate-500">Updated {data.active?.created_at?.slice(0,19)} by {data.active?.created_by?.slice(0,8)}</div>
          </div>
          <div className="rounded-2xl border bg-white p-4 shadow-sm">
            <h2 className="text-sm font-semibold">Tạo Draft (chỉnh sửa)</h2>
            <textarea value={draft} onChange={(e)=>setDraft(e.target.value)} rows={10} className="mt-2 w-full rounded border p-2 text-xs" />
            <input value={desc} onChange={(e)=>setDesc(e.target.value)} placeholder="description" className="mt-2 w-full rounded border p-2 text-xs" />
            <button onClick={create} className="mt-2 rounded-full bg-blue-600 text-white px-4 py-2 text-xs">Tạo Draft (pending_approval)</button>
            {msg && <div className="mt-2 text-xs bg-amber-50 border p-2 rounded">{msg}</div>}
          </div>
        </div>
      )}
      <div className="rounded-2xl border bg-white p-4 shadow-sm">
        <h2 className="text-sm font-semibold">Dry-run preview — 5 queries (golden hoặc tự nhập)</h2>
        <p className="text-xs text-slate-500">Nếu để trống queries, hệ thống dùng 5 golden queries của tone. Hoặc nhập mỗi dòng 1 query (tối đa 10).</p>
        <textarea value={dryQueries} onChange={(e)=>setDryQueries(e.target.value)} rows={5} placeholder={"Nhập query tùy ý, mỗi dòng 1 câu\nvd: Ngưỡng chẩn đoán đái tháo đường là bao nhiêu?"} className="mt-2 w-full rounded border p-2 text-xs" />
        <button onClick={doDry} disabled={dryLoading} className="mt-2 rounded-full bg-emerald-600 text-white px-4 py-2 text-xs disabled:opacity-50">{dryLoading?"Đang chạy 5 queries...":"Chạy dry-run 5 queries"}</button>
        {dryRes && (
          <div className="mt-3 space-y-2">
            {dryRes.results?.map((r:any,i:number)=>(
              <div key={i} className="rounded-lg border p-3 bg-slate-50">
                <div className="text-xs font-bold">{i+1}. {r.query}</div>
                <div className="text-xs mt-1 whitespace-pre-wrap">{r.answer?.slice(0,600) || r.error}</div>
                <div className="text-[11px] text-slate-500">trace {r.trace_id} · {r.status}</div>
              </div>
            ))}
            <div className="text-xs text-emerald-700">Dry-run xong — nếu hài lòng, bấm Approve draft ở bảng versions bên dưới.</div>
          </div>
        )}
      </div>
      <div className="rounded-2xl border bg-white p-4 shadow-sm">
        <h2 className="text-sm font-semibold">Versions — {tone}</h2>
        <table className="w-full text-xs mt-2">
          <thead className="bg-slate-50 border-b"><tr><th className="p-2 text-left">Version</th><th>Status</th><th>Created</th><th>Actions</th></tr></thead>
          <tbody>
            {data?.versions?.map((v:any)=>(
              <tr key={v.id} className="border-b">
                <td className="p-2 font-mono">{v.version} {v.is_active? "● active":""}</td>
                <td className="p-2"><span className={`rounded-full px-2 py-0.5 border text-[11px] ${v.status==="active"?"bg-emerald-50 border-emerald-200": v.status==="pending_approval"?"bg-amber-50 border-amber-200":"bg-slate-100"}`}>{v.status}</span></td>
                <td className="p-2 text-[11px]">{v.created_at?.slice(0,19).replace("T"," ")}</td>
                <td className="p-2 flex gap-1">
                  {v.status==="pending_approval" && <><button onClick={()=>approve(v.version)} className="rounded bg-emerald-600 text-white px-2 py-1">Approve (2nd step)</button><button onClick={()=>reject(v.version)} className="rounded border px-2 py-1">Reject</button></>}
                  {v.status==="archived" && <button onClick={()=>approve(v.version)} className="rounded border px-2 py-1">Rollback → active</button>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
